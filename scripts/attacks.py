"""Batería de ataques contra NUESTRO sistema (sobre una copia de la BD).

    python scripts/attacks.py

Cubre los supuestos del enunciado:
  1. El atacante LEE la BD (sin clave / con la clave robada).
  2. El atacante ESCRIBE en la BD (manipula, mueve, borra, añade, rollback).
  3. El atacante se SALTA partes del sistema (_t con templates fabricados,
     _sd con imágenes que no son caras).
Genera docs/results_attacks.json y docs/figs/reconstruccion.png
"""
from __future__ import annotations

import base64
import csv
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from facerec import config as C  # noqa: E402
from facerec.db import DBIntegrityError, TemplateDB, load_key  # noqa: E402
from facerec.detector import align_face, load_image  # noqa: E402
from facerec.system import SistemaBiometrico, TemplateError  # noqa: E402

DATA = C.ROOT / "data"
os.environ.setdefault(C.RATE_LIMIT_ENV_VAR, "0")   # la batería hace muchas llamadas fallidas


def read_rows(p):
    with open(p, newline="") as f:
        return list(csv.reader(f))


def write_rows(p, rows):
    with open(p, "w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)


def try_load(sysm):
    try:
        sysm.db.load()
        return "NO detectado"
    except DBIntegrityError as e:
        return f"DETECTADO ({e})"


def main():
    key = load_key()
    work = Path(tempfile.mkdtemp(prefix="facerec_attacks_"))
    db_path = work / "templates.csv"
    shutil.copy(C.DEFAULT_DB_PATH, db_path)
    sysm = SistemaBiometrico(db_path=db_path, key=key)
    lam = sysm.threshold
    clean = read_rows(db_path)
    victim, other = "taylor_swift_2009", "keanu_reeves_2021"
    victim_photo = next(DATA.glob(victim + ".*"))
    attacker_photo = next(DATA.glob("zohran-mamdani.*"))
    R: dict = {}

    def reset():
        write_rows(db_path, clean)

    # ------------------------------------------------------------------ 1. lectura
    R["1_lectura_sin_clave"] = {
        "que_ve": "ids en claro; nonce y template cifrados (AES-256-GCM)",
        "ejemplo_fila": clean[1][:1] + [clean[1][1], clean[1][2][:40] + "..."],
        "ids_visibles": len(clean) - 2,
    }
    # con la clave robada: descifra y reconstruye el mapa de orientaciones
    db = sysm.db.load()
    t = db[victim]
    rec = sysm.model.reconstruct(t)[0]
    n = C.FACE_SIZE
    phi_rec = np.arctan2(rec[n * n:], rec[:n * n]).reshape(n, n)
    img = load_image(victim_photo)
    face = align_face(img, sysm.detector.best_face(img), n)
    from facerec.igo import igo_vector
    v = igo_vector(face)
    phi_true = np.arctan2(v[n * n:], v[:n * n]).reshape(n, n)
    corr = float(np.mean(np.cos(phi_true - phi_rec)))
    fig, ax = plt.subplots(1, 3, figsize=(7.5, 2.8))
    ax[0].imshow(face, cmap="gray"); ax[0].set_title("cara inscrita")
    ax[1].imshow(phi_true, cmap="hsv"); ax[1].set_title("orientaciones reales")
    ax[2].imshow(phi_rec, cmap="hsv"); ax[2].set_title("reconstruidas\ndesde el template")
    for a in ax:
        a.axis("off")
    fig.tight_layout(); fig.savefig(C.ROOT / "docs" / "figs" / "reconstruccion.png", dpi=120); plt.close(fig)
    R["1_lectura_con_clave_robada"] = {
        "templates_descifrados": len(db),
        "reconstruccion_orientaciones_cos_medio": round(corr, 3),
        "nota": "con la clave + el modelo PCA (público en el repo) se reconstruye el mapa de bordes "
                "de la cara inscrita; ver docs/figs/reconstruccion.png",
    }

    # ------------------------------------------------------------------ 2. escritura
    W = {}
    rows = [r[:] for r in clean]
    i = next(k for k, r in enumerate(rows) if r[0] == victim)
    raw = bytearray(base64.b64decode(rows[i][2])); raw[5] ^= 1
    rows[i][2] = base64.b64encode(bytes(raw)).decode()
    write_rows(db_path, rows); W["modificar 1 bit de un template"] = try_load(sysm); reset()

    rows = [r[:] for r in clean]
    j = next(k for k, r in enumerate(rows) if r[0] == other)
    rows[i][1:], rows[j][1:] = rows[j][1:], rows[i][1:]
    write_rows(db_path, rows); W["intercambiar templates entre dos ids"] = try_load(sysm); reset()

    rows = [r[:] for r in clean]
    rows[i][0] = "atacante"
    write_rows(db_path, rows); W["renombrar el id de una fila"] = try_load(sysm); reset()

    rows = [r for r in clean if r[0] != victim]
    write_rows(db_path, rows); W["borrar una fila"] = try_load(sysm); reset()

    rows = [r[:] for r in clean]
    rows.insert(2, rows[i][:])
    write_rows(db_path, rows); W["duplicar una fila"] = try_load(sysm); reset()

    rows = [r[:] for r in clean]
    fake = np.random.default_rng(0).normal(size=sysm.model.k).astype("<f4")
    rows.insert(2, ["atacante", base64.b64encode(b"\0" * 12).decode(),
                    base64.b64encode(fake.tobytes() + b"\0" * 16).decode()])
    write_rows(db_path, rows); W["añadir una fila fabricada (sin clave)"] = try_load(sysm); reset()

    rows = [r[:] for r in clean]
    rows[i][1:] = ["", ""]
    write_rows(db_path, rows); W["vaciar campos"] = try_load(sysm); reset()

    write_rows(db_path, [["id", "nonce", "data"]]); W["dejar solo la cabecera"] = try_load(sysm); reset()

    # rollback de fichero completo
    snapshot = db_path.read_bytes()
    os.environ[C.ADMIN_TOKEN_ENV_VAR] = "token-solo-para-este-script"
    sysm.borrar(victim, "token-solo-para-este-script")   # el admin da de baja a la víctima...
    db_path.write_bytes(snapshot)             # ...el atacante restaura la copia antigua
    W["rollback: restaurar una copia antigua completa del CSV"] = (
        try_load(sysm) + " -> la persona dada de baja vuelve a estar inscrita")
    reset()

    # CSV vacío / borrado (solo permiso de escritura, sin clave): ya NO es "BD vacía"
    db_path.write_bytes(b""); W["vaciar el CSV (0 bytes)"] = try_load(sysm); reset()
    db_path.unlink(); W["borrar el fichero CSV"] = try_load(sysm); reset()
    try:
        sysm.borrar(victim)
        W["borrar un id SIN token de administrador"] = "ACEPTADO (¡suplantación posible!)"
    except PermissionError as e:
        W["borrar un id SIN token de administrador"] = f"RECHAZADO ({e})"

    # con la clave robada: sustituir el template de la víctima por el del atacante
    att_t = sysm.extraer(attacker_photo)
    d = sysm.db.load(); d[victim] = att_t; sysm.db.save(d)
    W["con clave robada: poner mi template bajo el id de la víctima"] = (
        f"NO detectado (cifrado válido). verificar('{victim}', foto_atacante) = "
        f"{sysm.verificar(victim, attacker_photo)}")
    reset()
    R["2_escritura"] = W

    # ------------------------------------------------------------------ 3. saltarse partes
    T = {}
    leaked = db[victim]
    T["_t con el template filtrado de la víctima"] = f"verificar_t -> {sysm.verificar_t(victim, leaked)}"
    for name, bad in {
        "vector de ceros": np.zeros(sysm.model.k),
        "NaN": np.full(sysm.model.k, np.nan),
        "infinito": np.full(sysm.model.k, np.inf),
        "longitud 10": np.ones(10),
        "valores enormes (1e9)": np.full(sysm.model.k, 1e9),
        "texto": "hola,adios",
    }.items():
        try:
            T[f"_t con {name}"] = f"aceptado?? -> {sysm.verificar_t(victim, bad)}"
        except TemplateError as e:
            T[f"_t con {name}"] = f"RECHAZADO ({e})"

    ids = sorted(db)
    G = np.stack([db[k] for k in ids]).astype(np.float64)
    Gn = G / np.linalg.norm(G, axis=1, keepdims=True)

    # template "medio" (wolf): media de todos los inscritos
    mean_t = Gn.mean(0)
    s = Gn @ (mean_t / np.linalg.norm(mean_t))
    T["_t con el template medio de todos los inscritos"] = {
        "ids_que_superan_lambda": int((s >= lam).sum()),
        "ids_que_acepta_verificar_t_real": int(sum(sysm.verificar_t(k, mean_t) for k in ids)),
        "identificar_t": sysm.identificar_t(mean_t),
        "mejor_score": round(float(s.max()), 4)}

    # fuerza bruta con templates aleatorios (solo con respuesta sí/no)
    rng = np.random.default_rng(1)
    N = 200_000
    Rt = rng.normal(size=(N, sysm.model.k))
    Rt /= np.linalg.norm(Rt, axis=1, keepdims=True)
    S = Rt @ Gn.T
    T["_t fuerza bruta: 200.000 templates aleatorios"] = {
        "aciertos_contra_un_id_concreto": int((S[:, ids.index(victim)] >= lam).sum()),
        "aciertos_identificar_t_alguien": int((S.max(1) >= sysm.threshold_ident).sum()),
        "aciertos_verificar_t_contra_algun_id": int((S.max(1) >= lam).sum()),
        "max_score": round(float(S.max()), 4)}

    # hill-climbing si el atacante tuviera acceso al score (no lo damos en la CLI)
    x = rng.normal(size=sysm.model.k); x /= np.linalg.norm(x)
    target = Gn[ids.index(victim)]
    queries, best = 0, float(x @ target)
    while best < lam and queries < 100_000:
        y = x + 0.05 * rng.normal(size=x.size); y /= np.linalg.norm(y)
        queries += 1
        sy = float(y @ target)
        if sy > best:
            x, best = y, sy
    T["_t hill-climbing CON acceso al score"] = {
        "consultas_hasta_aceptar": queries, "score_final": round(best, 4),
        "nota": "por eso la CLI solo devuelve sí/no y no el score"}

    # imágenes sin cara por _sd y por la ruta normal
    rng = np.random.default_rng(0)
    nonfaces = {
        "gris uniforme": np.full((64, 64), 128, np.uint8),
        "ruido": rng.integers(0, 256, (64, 64), dtype=np.uint8),
        "tablero de ajedrez": (np.indices((64, 64)).sum(0) // 8 % 2 * 255).astype(np.uint8),
    }
    for name, im in nonfaces.items():
        T[f"identificar_sd con '{name}'"] = str(sysm.identificar_sd(im))
        try:
            T[f"identificar (con detector) con '{name}'"] = str(sysm.identificar(im))
        except Exception as e:  # noqa: BLE001
            T[f"identificar (con detector) con '{name}'"] = f"RECHAZADO ({e})"
    try:
        sysm.extraer_sd(np.zeros((10, 10), np.uint8))
        T["_sd con imagen de 10x10"] = "aceptada"
    except Exception as e:  # noqa: BLE001
        T["_sd con imagen de 10x10"] = f"RECHAZADO ({e})"

    # ids maliciosos
    for bad_id in ["=cmd|' /C calc'!A0", "../../etc/passwd", "", "a" * 100, "__manifest__", "ana\n"]:
        try:
            sysm.inscribir_t(bad_id, leaked)
            T[f"inscribir id {bad_id[:20]!r}"] = "ACEPTADO"
        except ValueError as e:
            T[f"inscribir id {bad_id[:20]!r}"] = f"RECHAZADO ({str(e)[:40]}...)"
    try:
        sysm.inscribir(victim, attacker_photo)
        T["re-inscribir un id existente con otra cara"] = "ACEPTADO (¡suplantación!)"
    except ValueError as e:
        T["re-inscribir un id existente con otra cara"] = f"RECHAZADO ({e})"
    T["verificar un id que no existe"] = str(sysm.verificar("no_existe", victim_photo))
    R["3_saltarse_partes"] = T

    shutil.rmtree(work)
    out = C.ROOT / "docs" / "results_attacks.json"
    out.write_text(json.dumps(R, indent=1, ensure_ascii=False))
    print(json.dumps(R, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
