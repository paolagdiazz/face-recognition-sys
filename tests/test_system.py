"""Tests del sistema. Ejecutar: python -m pytest -q

Usan una BD temporal con clave nueva; no tocan db/ ni secrets/.
"""
import base64
import csv
from pathlib import Path

import numpy as np
import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from facerec import DBIntegrityError, FaceError, SistemaBiometrico, TemplateError
from facerec import config as C

DATA = C.ROOT / "data"
A = next(DATA.glob("adele_2016.*"))
B = next(DATA.glob("keanu_reeves_2021.*"))
M = next(DATA.glob("zohran-mamdani.*"))


@pytest.fixture()
def s(tmp_path):
    return SistemaBiometrico(db_path=tmp_path / "t.csv", key=AESGCM.generate_key(256))


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
    s.borrar("adele")
    assert s.inscritos() == []
