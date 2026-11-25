"""Pruebas extra para hacer vosotras mismas (no necesita LFW ni matplotlib).

    python scripts/pruebas_extra.py              # todo
    python scripts/pruebas_extra.py --solo seguridad|reales|robustez

Usa una BD temporal con clave nueva: no toca db/ ni secrets/.

  reales    : falsos aceptados entre las propias fotos del profe (FAR real) y
              autoverificación / autoidentificación.
  robustez  : giros, escalas, JPEG, ruido, luz, espejo, oclusión sobre las fotos
              de inscripción. Cuenta aceptados y también identificaciones ERRÓNEAS.
  seguridad : regresiones de los fallos encontrados (CSV vacío, borrar sin token,
              id con salto de línea, bomba de descompresión, template maestro,
              límite de intentos, varias caras).
  concurrencia : 6 procesos inscribiendo a la vez (no deben perderse inscripciones).
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from facerec import DBIntegrityError, FaceError, RateLimitError, SistemaBiometrico  # noqa: E402
from facerec import config as C  # noqa: E402
from facerec.db import validate_id  # noqa: E402
from facerec.detector import load_image  # noqa: E402

DATA = C.ROOT / "data"
# La evaluación hace miles de comparaciones fallidas: se desactiva el límite de
# intentos (en el bloque de seguridad se activa a propósito para probarlo).
os.environ.setdefault(C.RATE_LIMIT_ENV_VAR, "0")


def montar(tmp: Path) -> SistemaBiometrico:
    s = SistemaBiometrico(db_path=tmp / "t.csv", key=AESGCM.generate_key(256), crear_bd=True)
    for p in sorted(DATA.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            try:
                s.inscribir(p.stem, p)
            except Exception as e:  # noqa: BLE001
                raise SystemExit(f"ERROR inscribiendo {p.name}: {type(e).__name__}: {e}")
    return s


def reales(s):
    print("\n== REALES: fotos del profe contra ellas mismas ==")
    ids = s.inscritos()
    D = {p.stem: p for p in DATA.iterdir()}
    v = sum(s.verificar(i, D[i]) for i in ids)
    n = sum(s.identificar(D[i]) == i for i in ids)
    print(f"misma foto: verificar {v}/{len(ids)}, identificar {n}/{len(ids)}")
    T = np.stack([s.extraer(D[i]) for i in ids]).astype(float)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    S = T @ T.T
    off = ~np.eye(len(ids), dtype=bool)
    acc = (S >= s.threshold) & off
    print(f"FAR real con λ={s.threshold:.4f}: {acc.sum() / off.sum():.2%} "
          f"(objetivo 1 %) | ids con algún impostor aceptado: {int(acc.any(1).sum())}/{len(ids)}")
    print(f"pares impostores sobre λ_id={s.threshold_ident:.4f}: {int(((S >= s.threshold_ident) & off).sum())}")
    iu = np.triu_indices(len(ids), 1)
    for k in np.argsort(-S[iu])[:5]:
        print(f"   {S[iu][k]:.3f}  {ids[iu[0][k]]}  <->  {ids[iu[1][k]]}")


def _rot(a):
    def f(im):
        h, w = im.shape[:2]
        return cv2.warpAffine(im, cv2.getRotationMatrix2D((w / 2, h / 2), a, 1), (w, h),
                              borderMode=cv2.BORDER_REPLICATE)
    return f


def _noise(sig):
    rng = np.random.default_rng(0)
    return lambda im: np.clip(im + rng.normal(0, sig, im.shape), 0, 255).astype(np.uint8)


def robustez(s):
    print("\n== ROBUSTEZ sobre las fotos de inscripción (sobre 50) ==")
    ids = s.inscritos()
    D = {p.stem: p for p in DATA.iterdir()}
    IM = {i: load_image(D[i]) for i in ids}
    tests = {
        "original": lambda im: im,
        "espejo horizontal": lambda im: im[:, ::-1].copy(),
        "rotación +5°": _rot(5), "rotación -10°": _rot(-10), "rotación +20°": _rot(20), "rotación 45°": _rot(45),
        "escala 0.5": lambda im: cv2.resize(im, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA),
        "escala 0.25": lambda im: cv2.resize(im, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA),
        "escala 0.1": lambda im: cv2.resize(im, None, fx=.1, fy=.1, interpolation=cv2.INTER_AREA),
        "JPEG q=10": lambda im: cv2.imdecode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 10])[1], 1),
        "ruido σ=15": _noise(15), "ruido σ=40": _noise(40),
        "oscura (γ=2.5)": lambda im: (255 * (im / 255.0) ** 2.5).astype(np.uint8),
        "clara (γ=0.4)": lambda im: (255 * (im / 255.0) ** 0.4).astype(np.uint8),
        "media cara negra": lambda im: np.concatenate([im[:, :im.shape[1] // 2], np.zeros_like(im[:, im.shape[1] // 2:])], 1),
    }
    print(f"{'transformación':22s} {'sin cara':>9s} {'verif OK':>9s} {'ident OK':>9s} {'ident ERRÓNEA':>14s}")
    for name, f in tests.items():
        nf = vo = io = iw = 0
        for i in ids:
            try:
                t = s.extraer(f(IM[i]))
            except FaceError:
                nf += 1
                continue
            vo += s.verificar_t(i, t)
            r = s.identificar_t(t)
            io += r == i
            iw += r is not None and r != i
        print(f"{name:22s} {nf:>9d} {vo:>9d} {io:>9d} {iw:>14d}")


def seguridad(s, tmp: Path):
    print("\n== SEGURIDAD (regresiones) ==")
    ok = lambda b: "OK " if b else "FALLO"   # noqa: E731
    D = {p.stem: p for p in DATA.iterdir()}
    victima, atacante = "taylor_swift_2009", D["zohran-mamdani"]
    copia = tmp / "copia.csv"
    copia.write_bytes(s.db.path.read_bytes())

    for nombre, accion in (("CSV de 0 bytes", lambda: s.db.path.write_bytes(b"")),
                           ("CSV borrado", lambda: s.db.path.unlink())):
        accion()
        try:
            s.inscribir(victima, atacante)
            print(f"[{ok(False)}] {nombre}: se pudo re-inscribir a la víctima")
        except DBIntegrityError:
            print(f"[{ok(True)}] {nombre}: detectado, no se acepta como BD vacía")
        s.db.path.write_bytes(copia.read_bytes())

    os.environ.pop(C.ADMIN_TOKEN_ENV_VAR, None)
    try:
        s.borrar(victima, "x")
        print(f"[{ok(False)}] borrar sin variable de administrador")
    except PermissionError:
        print(f"[{ok(True)}] borrar deshabilitado sin FACEREC_ADMIN_TOKEN")
    os.environ[C.ADMIN_TOKEN_ENV_VAR] = "abc"
    try:
        s.borrar(victima, "zzz")
        print(f"[{ok(False)}] borrar con token incorrecto")
    except PermissionError:
        print(f"[{ok(True)}] borrar con token incorrecto rechazado")
    os.environ.pop(C.ADMIN_TOKEN_ENV_VAR)

    for pid in ("ana\n", "a" * 65, "=1+1", "../x", "__manifest__", ""):
        try:
            validate_id(pid)
            print(f"[{ok(False)}] id {pid[:12]!r} aceptado")
        except ValueError:
            print(f"[{ok(True)}] id {pid[:12]!r} rechazado")

    f = tmp / "bomba.png"
    f.write_bytes(cv2.imencode(".png", np.zeros((9000, 9000), np.uint8))[1].tobytes())
    try:
        load_image(f)
        print(f"[{ok(False)}] PNG 9000x9000 aceptado")
    except FaceError:
        print(f"[{ok(True)}] PNG de {f.stat().st_size // 1024} KB que declara 9000x9000 rechazado")

    # template 'maestro': media de los inscritos (defensa "lobo")
    db = s.db.load()
    G = np.stack(list(db.values())).astype(float)
    Gn = G / np.linalg.norm(G, axis=1, keepdims=True)
    m = Gn.mean(0)
    sc = Gn @ (m / np.linalg.norm(m))
    acepta = sum(s.verificar_t(i, m) for i in db)
    print(f"[{ok(acepta == 0)}] template medio de la BD: supera λ en {(sc >= s.threshold).sum()}/{len(db)} ids, "
          f"verificar_t lo acepta en {acepta} (antes del parche: 19)")
    D = {p.stem: p for p in DATA.iterdir()}
    legit = sum(s.verificar(i, D[i]) for i in db)
    print(f"[{ok(legit == len(db))}] usuarios legítimos siguen entrando: {legit}/{len(db)}")

    # límite de intentos
    os.environ[C.RATE_LIMIT_ENV_VAR] = "5"
    rng = np.random.default_rng(1)
    try:
        bloqueado = False
        for _ in range(8):
            try:
                s.verificar_t(victima, rng.normal(size=s.model.k))
            except RateLimitError:
                bloqueado = True
                break
        print(f"[{ok(bloqueado)}] límite de intentos: tras 5 rechazos seguidos se bloquea el id")
    finally:
        os.environ[C.RATE_LIMIT_ENV_VAR] = "0"
    rl = s.limiter.path
    if rl.exists():
        rl.unlink()

    # fuerza bruta _t (ahora acotada por el límite; aquí se mide sin límite)
    R = np.random.default_rng(1).normal(size=(20000, s.model.k))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    hit = int(((R @ Gn.T)[:, list(db).index(victima)] >= s.threshold).sum())
    print(f"[info] fuerza bruta _t SIN límite: {hit}/20000 plantillas aleatorias pasan contra un id "
          f"(con el límite por defecto, 30 rechazos/min)")

    # varias caras
    v, a = cv2.imread(str(D[victima])), cv2.imread(str(atacante))

    def h(im, alto):
        return cv2.resize(im, (round(im.shape[1] * alto / im.shape[0]), alto), interpolation=cv2.INTER_AREA)
    try:
        s.extraer(np.concatenate([h(v, 400), h(a, 400)], 1))
        print(f"[{ok(False)}] dos caras de igual tamaño: se eligió una en silencio")
    except FaceError:
        print(f"[{ok(True)}] dos caras de igual tamaño: rechazada por ambigua")


def concurrencia(tmp: Path):
    print("\n== CONCURRENCIA: 6 procesos inscribiendo a la vez ==")
    import multiprocessing as mp
    key = AESGCM.generate_key(256)
    db = tmp / "conc.csv"
    SistemaBiometrico(db_path=db, key=key, crear_bd=True)
    n = 6
    with mp.Pool(n) as pool:
        pool.map(_worker_conc, [(str(db), key, i) for i in range(n)])
    hay = len(SistemaBiometrico(db_path=db, key=key).inscritos())
    print(f"[{'OK ' if hay == n else 'FALLO'}] inscripciones guardadas: {hay}/{n} (antes del parche: 2-4 de 6)")


def _worker_conc(args):
    db, key, i = args
    s = SistemaBiometrico(db_path=db, key=key)
    s.inscribir_t(f"persona{i}", np.random.default_rng(i).normal(size=s.model.k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", choices=["reales", "robustez", "seguridad", "concurrencia"])
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        s = montar(tmp)
        if a.solo in (None, "reales"):
            reales(s)
        if a.solo in (None, "robustez"):
            robustez(s)
        if a.solo in (None, "seguridad"):
            seguridad(s, tmp)
        if a.solo in (None, "concurrencia"):
            concurrencia(tmp)


if __name__ == "__main__":
    __import__("multiprocessing").freeze_support()
    main()
