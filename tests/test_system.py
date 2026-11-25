"""Tests del sistema. Ejecutar: python -m pytest -q

Usan una BD temporal con clave nueva; no tocan db/ ni secrets/.
"""
import base64
import csv
from pathlib import Path

import numpy as np
import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from facerec import DBIntegrityError, FaceError, RateLimitError, SistemaBiometrico, TemplateError
from facerec import config as C

DATA = C.ROOT / "data"
A = next(DATA.glob("adele_2016.*"))
B = next(DATA.glob("keanu_reeves_2021.*"))
M = next(DATA.glob("zohran-mamdani.*"))


@pytest.fixture()
def s(tmp_path):
    return SistemaBiometrico(db_path=tmp_path / "t.csv", key=AESGCM.generate_key(256), crear_bd=True)


def test_nueve_metodos(s):
    for m in ["inscribir", "verificar", "identificar", "inscribir_sd", "verificar_sd",
              "identificar_sd", "inscribir_t", "verificar_t", "identificar_t"]:
        assert callable(getattr(s, m))


def test_flujo_normal(s):
    s.inscribir("adele", A)
    s.inscribir("keanu", B)
    assert s.verificar("adele", A) is True
    assert s.verificar("keanu", A) is False
    assert s.identificar(A) == "adele"
    assert s.identificar(M) is None          # no inscrito -> desconocido


def test_template_tiene_k_valores(s):
    t = s.extraer(A)
    assert t.shape == (s.model.k,) and np.all(np.isfinite(t))


def test_flujo_t(s):
    t = s.extraer(A)
    s.inscribir_t("adele", t)
    assert s.verificar_t("adele", t)
    assert s.identificar_t(",".join(map(str, t))) == "adele"


def test_flujo_sd(s):
    s.inscribir_sd("adele", A)
    assert s.verificar_sd("adele", A)
    assert s.identificar_sd(A) == "adele"


def test_no_sobrescribe(s):
    s.inscribir("adele", A)
    with pytest.raises(ValueError):
        s.inscribir("adele", B)


def test_sin_cara(s):
    with pytest.raises(FaceError):
        s.verificar("adele", np.full((200, 200, 3), 127, np.uint8))


@pytest.mark.parametrize("bad", [np.zeros(150), np.full(150, np.nan), np.ones(10),
                                 np.full(150, 1e9), "a,b,c"])
def test_template_invalido(s, bad):
    with pytest.raises(TemplateError):
        s.verificar_t("x", bad)


@pytest.mark.parametrize("bad_id", ["", "=1+1", "../x", "a" * 65, "__manifest__", "a,b"])
def test_id_invalido(s, bad_id):
    with pytest.raises(ValueError):
        s.inscribir_t(bad_id, np.ones(150))


def _rows(p):
    with open(p, newline="") as f:
        return list(csv.reader(f))


def _write(p, rows):
    with open(p, "w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)


def test_bd_cifrada_y_manipulacion(s):
    s.inscribir("adele", A)
    s.inscribir("keanu", B)
    p = s.db.path
    rows = _rows(p)
    assert rows[0] == ["id", "nonce", "data"]
    # el template no aparece en claro
    t = s.extraer(A).astype("<f4").tobytes()
    assert base64.b64encode(t).decode()[:20] not in p.read_text()
    # intercambiar templates -> detectado
    bad = [r[:] for r in rows]
    bad[1][1:], bad[2][1:] = bad[2][1:], bad[1][1:]
    _write(p, bad)
    with pytest.raises(DBIntegrityError):
        s.verificar("adele", A)
    # borrar una fila -> detectado
    _write(p, [rows[0], rows[2], rows[3]])
    with pytest.raises(DBIntegrityError):
        s.identificar(A)
    _write(p, rows)
    assert s.verificar("adele", A)


def test_clave_incorrecta(s, tmp_path):
    s.inscribir("adele", A)
    otro = SistemaBiometrico(db_path=s.db.path, key=AESGCM.generate_key(256))
    with pytest.raises(DBIntegrityError):
        otro.verificar("adele", A)


def test_borrar(s):
    s.inscribir("adele", A)
    with _admin("secreto"):
        s.borrar("adele", "secreto")
    assert s.inscritos() == []


# --------------------------------------------------------------------------
# Regresiones de la revisión de seguridad
# --------------------------------------------------------------------------
import contextlib
import os


@contextlib.contextmanager
def _admin(token):
    old = os.environ.get(C.ADMIN_TOKEN_ENV_VAR)
    if token is None:
        os.environ.pop(C.ADMIN_TOKEN_ENV_VAR, None)
    else:
        os.environ[C.ADMIN_TOKEN_ENV_VAR] = token
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(C.ADMIN_TOKEN_ENV_VAR, None)
        else:
            os.environ[C.ADMIN_TOKEN_ENV_VAR] = old


def test_borrar_exige_token(s):
    s.inscribir("adele", A)
    with _admin(None):                       # sin variable: deshabilitado
        with pytest.raises(PermissionError):
            s.borrar("adele", "lo-que-sea")
    with _admin("secreto"):
        with pytest.raises(PermissionError):
            s.borrar("adele")                # sin token
        with pytest.raises(PermissionError):
            s.borrar("adele", "otro")        # token incorrecto
    assert s.inscritos() == ["adele"]


def test_bd_vacia_o_borrada_no_es_bd_nueva(s):
    s.inscribir("adele", A)
    p = s.db.path
    p.write_bytes(b"")                       # truncada
    with pytest.raises(DBIntegrityError):
        s.inscritos()
    with pytest.raises(DBIntegrityError):
        s.inscribir("adele", M)              # no se puede re-inscribir sobre una BD vaciada
    p.unlink()                               # borrada
    with pytest.raises(DBIntegrityError):
        s.verificar("adele", A)


def test_crear_bd_solo_si_se_pide(tmp_path):
    key = AESGCM.generate_key(256)
    sin = SistemaBiometrico(db_path=tmp_path / "a.csv", key=key)
    with pytest.raises(DBIntegrityError):
        sin.inscritos()
    con = SistemaBiometrico(db_path=tmp_path / "b.csv", key=key, crear_bd=True)
    assert con.inscritos() == []


@pytest.mark.parametrize("bad_id", ["ana\n", "a" * 63 + "\n", "a" * 65, " ana", "ana\r"])
def test_id_con_salto_de_linea_o_largo(s, bad_id):
    with pytest.raises(ValueError):
        s.inscribir_t(bad_id, np.ones(150))


def test_bomba_de_descompresion_se_rechaza_sin_decodificar(tmp_path):
    import cv2
    ok, buf = cv2.imencode(".png", np.zeros((9000, 9000), np.uint8))   # 81 Mpx > 8000^2
    f = tmp_path / "bomba.png"
    f.write_bytes(buf.tobytes())
    assert f.stat().st_size < 1_000_000
    from facerec.detector import _peek_dims
    assert _peek_dims(np.frombuffer(f.read_bytes(), np.uint8)) == (9000, 9000)
    from facerec.detector import load_image
    with pytest.raises(FaceError):
        load_image(f)


def test_todas_las_fotos_de_data_se_cargan():
    """Detecta fotos que este entorno no sabe decodificar (p. ej. AVIF con extensión .jpg)."""
    from facerec.detector import load_image
    malas = []
    for p in sorted(DATA.iterdir()):
        try:
            load_image(p)
        except FaceError as e:
            malas.append(f"{p.name}: {e}")
    assert not malas, "no se pueden decodificar:\n" + "\n".join(malas)


def test_respaldo_pillow_cuando_opencv_no_decodifica():
    import facerec.detector as det
    avif = DATA / "aitana_etxebarria_(twin_melody)_2024.jpg"
    original = det.cv2.imdecode
    det.cv2.imdecode = lambda *a, **k: None          # simula un OpenCV sin AVIF
    try:
        img = det.load_image(avif)
    finally:
        det.cv2.imdecode = original
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8


# --------------------------------------------------------------------------
# Mejoras: defensa "lobo", límite de intentos, bloqueo de fichero, caras ambiguas
# --------------------------------------------------------------------------
@contextlib.contextmanager
def _rate_limit(n):
    old = os.environ.get(C.RATE_LIMIT_ENV_VAR)
    os.environ[C.RATE_LIMIT_ENV_VAR] = str(n)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(C.RATE_LIMIT_ENV_VAR, None)
        else:
            os.environ[C.RATE_LIMIT_ENV_VAR] = old


def test_defensa_lobo_rechaza_plantilla_que_se_parece_a_todos(s):
    rng = np.random.default_rng(0)
    base = rng.normal(size=150)
    for i in range(12):                                  # 12 ids casi idénticos entre sí
        s.inscribir_t(f"p{i}", base + 0.01 * rng.normal(size=150))
    with _rate_limit(0):
        assert s.verificar_t("p0", base) is False        # supera λ contra 12 ids > 8: fabricada
        otra = rng.normal(size=150)
        s.inscribir_t("distinto", otra)
        assert s.verificar_t("distinto", otra) is True   # una plantilla normal sigue pasando


def test_limite_de_intentos_fallidos(s):
    s.inscribir("adele", A)
    mala = np.random.default_rng(1).normal(size=150)
    with _rate_limit(3):
        for _ in range(3):
            assert s.verificar_t("adele", mala) is False
        with pytest.raises(RateLimitError):
            s.verificar_t("adele", mala)
        with pytest.raises(RateLimitError):              # ni siquiera el usuario legítimo, mientras dure
            s.verificar("adele", A)
        s.limiter.clock = lambda: __import__("time").time() + 120     # pasa la ventana
        assert s.verificar("adele", A) is True


def test_limite_desactivado_con_cero(s):
    s.inscribir("adele", A)
    mala = np.random.default_rng(2).normal(size=150)
    with _rate_limit(0):
        for _ in range(60):
            s.verificar_t("adele", mala)
    assert not s.limiter.path.exists()


def test_limite_identificar(s):
    mala = np.random.default_rng(3).normal(size=150)
    with _rate_limit(2):
        s.identificar_t(mala)
        s.identificar_t(mala)
        with pytest.raises(RateLimitError):
            s.identificar_t(mala)


def test_inscripciones_simultaneas_no_se_pierden(tmp_path):
    import threading
    key = AESGCM.generate_key(256)
    db = tmp_path / "c.csv"
    SistemaBiometrico(db_path=db, key=key, crear_bd=True)
    n = 6
    sistemas = [SistemaBiometrico(db_path=db, key=key) for _ in range(n)]
    barrera = threading.Barrier(n)
    errores = []

    def worker(i):
        try:
            barrera.wait()
            sistemas[i].inscribir_t(f"persona{i}", np.random.default_rng(i).normal(size=150))
        except Exception as e:  # noqa: BLE001
            errores.append(e)

    hilos = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    [h.start() for h in hilos]
    [h.join() for h in hilos]
    assert not errores
    assert len(SistemaBiometrico(db_path=db, key=key).inscritos()) == n


def test_bloqueo_huerfano_se_rompe(tmp_path):
    from facerec.filelock import FileLock
    p = tmp_path / "x.lock"
    p.write_text("1234")
    os.utime(p, (0, 0))                                  # bloqueo viejísimo: proceso muerto
    with FileLock(p, timeout=2, stale=10):
        pass
    assert not p.exists()


def test_dos_caras_parecidas_son_ambiguas_y_una_dominante_no(s):
    import cv2
    v = cv2.imread(str(DATA / "taylor_swift_2009.jpg"))

    def h(im, alto):
        return cv2.resize(im, (round(im.shape[1] * alto / im.shape[0]), alto), interpolation=cv2.INTER_AREA)

    a = cv2.imread(str(M))
    img = np.concatenate([h(v, 400), h(a, 400)], 1)
    with pytest.raises(FaceError):
        s.extraer(img)
    pv = cv2.copyMakeBorder(h(v, 700), 0, 0, 0, 0, cv2.BORDER_CONSTANT)
    pa = cv2.copyMakeBorder(h(a, 200), 0, 500, 0, 0, cv2.BORDER_CONSTANT)
    assert s.extraer(np.concatenate([pv, pa], 1)).shape == (150,)    # una cara domina: se usa
