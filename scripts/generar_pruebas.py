"""Genera pruebas SIMULADAS a partir de las fotos de data/ (no descarga nada).

    python scripts/generar_pruebas.py            # crea/rehace tests/*_simulada

Crea:
  tests/mamdani_simulada/       variantes de la foto de Mamdani de data/ (luz, giro, gafas,
                                mascarilla, desenfoque, ruido...). Prueba de ROBUSTEZ.
  tests/presentacion_simulada/  "foto a una pantalla" e "impresa" de 5 inscritos (perspectiva,
                                reflejo, moiré, bajo contraste, JPEG...).
  tests/varias_caras_simulada/  fotos con 2 caras (iguales y una dominante).

IMPORTANTE: son simulaciones. No sustituyen a fotos reales de otra sesión/edad/cámara
(Mamdani) ni a fotos hechas con un móvil a una pantalla o a papel. Usadlas para ver cómo
reacciona el sistema y para tener la estructura lista; para la memoria, completad con fotos reales.
Luego:  python scripts/probar_fotos.py --id zohran-mamdani tests/mamdani_simulada
        python scripts/probar_fotos.py tests/presentacion_simulada
        python scripts/probar_fotos.py tests/varias_caras_simulada
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from facerec import config as C  # noqa: E402
from facerec.detector import Detector, load_image  # noqa: E402

DATA, TESTS = C.ROOT / "data", C.ROOT / "tests"
PRESENT = ["taylor_swift_2009", "adele_2016", "keanu_reeves_2021", "lady_gaga_2009", "idris_elba_2022"]


def foto(stem, lado=1000):
    """Foto de data/ reducida a como mucho `lado` px (las pruebas pesan menos y la cara sigue siendo grande)."""
    im = load_image(next(DATA.glob(stem + ".*")))
    k = lado / max(im.shape[:2])
    return im if k >= 1 else cv2.resize(im, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)


def save(path, img, q=88):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, q])


def gamma(im, g):
    return (255 * (im / 255.0) ** g).clip(0, 255).astype(np.uint8)


def rot(im, a):
    h, w = im.shape[:2]
    return cv2.warpAffine(im, cv2.getRotationMatrix2D((w / 2, h / 2), a, 1), (w, h), borderMode=cv2.BORDER_REPLICATE)


def noise(im, sig, seed=0):
    return np.clip(im + np.random.default_rng(seed).normal(0, sig, im.shape), 0, 255).astype(np.uint8)


def luz_lateral(im):
    w = im.shape[1]
    g = np.linspace(0.35, 1.25, w)[None, :, None]
    return np.clip(im * g, 0, 255).astype(np.uint8)


def perspectiva(im, k=0.06, seed=0):
    h, w = im.shape[:2]
    r = np.random.default_rng(seed)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = (src + r.uniform(-k, k, src.shape) * [w, h]).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(im, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(25, 25, 25))


def gafas_y_mascarilla(img, det):
    f = det.best_face(img)
    el, er = f.eyes
    iod = np.linalg.norm(er - el)
    g, m = img.copy(), img.copy()
    for e in (el, er):
        cv2.ellipse(g, tuple(int(v) for v in e), (int(iod * .38), int(iod * .27)), 0, 0, 360, (15, 15, 15), -1)
    cv2.line(g, tuple(int(v) for v in el), tuple(int(v) for v in er), (15, 15, 15), max(2, int(iod * .08)))
    nose, mouth_l, mouth_r = f.landmarks[2], f.landmarks[3], f.landmarks[4]
    pts = np.array([nose + [-iod * .7, -iod * .05], nose + [iod * .7, -iod * .05],
                    mouth_r + [iod * .5, iod * .75], mouth_l + [-iod * .5, iod * .75]], np.int32)
    cv2.fillPoly(m, [pts], (205, 190, 150))
    return g, m


def pantalla(im):
    """Foto de un monitor/móvil: marco oscuro, perspectiva, reflejo, moiré, desenfoque, ruido, JPEG."""
    im = cv2.copyMakeBorder(im, *(int(0.12 * max(im.shape[:2])),) * 4, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    im = perspectiva(im, 0.05, 1)
    h, w = im.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    moire = 12 * np.sin(2 * np.pi * (xx * 0.43 + yy * 0.29))
    glare = 90 * np.exp(-(((xx - 0.75 * w) / (0.18 * w)) ** 2 + ((yy - 0.25 * h) / (0.12 * h)) ** 2))
    im = np.clip(im.astype(np.float32) * 1.08 + moire[..., None] + glare[..., None], 0, 255).astype(np.uint8)
    im = cv2.GaussianBlur(im, (0, 0), 1.3)
    return noise(im, 6, 2)


def impresa(im):
    """Foto de un papel: bajo contraste, tono cálido, sombra, trama, giro, desenfoque, JPEG."""
    im = cv2.copyMakeBorder(im, *(int(0.08 * max(im.shape[:2])),) * 4, cv2.BORDER_CONSTANT, value=(235, 240, 245))
    im = np.clip(im.astype(np.float32) * 0.68 + 55, 0, 255)
    im[..., 0] *= 0.9
    h, w = im.shape[:2]
    sombra = np.linspace(1.0, 0.7, h)[:, None, None]
    trama = 10 * np.sign(np.sin(np.arange(w) * 1.7))[None, :, None]
    im = np.clip(im * sombra + trama, 0, 255).astype(np.uint8)
    im = rot(perspectiva(im, 0.03, 3), 3)
    im = cv2.GaussianBlur(im, (0, 0), 1.6)
    return noise(im, 5, 4)


def lado_a_lado(a, b, ha, hb):
    def h(im, alto):
        return cv2.resize(im, (round(im.shape[1] * alto / im.shape[0]), alto), interpolation=cv2.INTER_AREA)
    H = max(ha, hb)
    pa = cv2.copyMakeBorder(h(a, ha), 0, H - ha, 0, 0, cv2.BORDER_CONSTANT)
    pb = cv2.copyMakeBorder(h(b, hb), 0, H - hb, 0, 0, cv2.BORDER_CONSTANT)
    return np.concatenate([pa, pb], 1)


def main():
    for d in ("mamdani_simulada", "presentacion_simulada", "varias_caras_simulada"):
        shutil.rmtree(TESTS / d, ignore_errors=True)
    det = Detector()

    m = foto("zohran-mamdani")
    gafas, mascarilla = gafas_y_mascarilla(m, det)
    variantes = {
        "clara": gamma(m, 0.5), "oscura": gamma(m, 2.2),
        "contraste_bajo": np.clip(m.astype(np.float32) * 0.45 + 70, 0, 255).astype(np.uint8),
        "luz_lateral": luz_lateral(m), "giro_mas8": rot(m, 8), "giro_menos8": rot(m, -8),
        "reducida_30": cv2.resize(m, None, fx=.3, fy=.3, interpolation=cv2.INTER_AREA),
        "desenfoque": cv2.GaussianBlur(m, (0, 0), 2.5), "ruido": noise(m, 20),
        "gafas_de_sol": gafas, "mascarilla": mascarilla, "espejo": m[:, ::-1].copy(),
    }
    for n, im in variantes.items():
        save(TESTS / "mamdani_simulada" / f"{n}.jpg", im)

    for pid in PRESENT:
        im = foto(pid)
        save(TESTS / "presentacion_simulada" / f"{pid}__pantalla1.jpg", pantalla(im), 80)
        save(TESTS / "presentacion_simulada" / f"{pid}__impresa1.jpg", impresa(im), 80)

    v, a = foto("taylor_swift_2009"), foto("zohran-mamdani")
    save(TESTS / "varias_caras_simulada" / "taylor_swift_2009__dos_caras_iguales.jpg", lado_a_lado(v, a, 400, 400))
    save(TESTS / "varias_caras_simulada" / "taylor_swift_2009__ella_domina.jpg", lado_a_lado(v, a, 700, 200))
    save(TESTS / "varias_caras_simulada" / "zohran-mamdani__el_domina.jpg", lado_a_lado(v, a, 200, 700))
    for d, txt in (("mamdani_simulada", "Variantes SIMULADAS de data/zohran-mamdani.jpg (robustez). No son fotos reales distintas."),
                   ("presentacion_simulada", "Capturas SIMULADAS de pantalla/papel. Las reales hay que hacerlas con un móvil."),
                   ("varias_caras_simulada", "Composiciones de dos fotos de data/. Las reales: fotos con dos personas.")):
        (TESTS / d / "LEEME.txt").write_text(txt + "\n", encoding="utf-8")
    print("generado en tests/: mamdani_simulada, presentacion_simulada, varias_caras_simulada")


if __name__ == "__main__":
    main()
