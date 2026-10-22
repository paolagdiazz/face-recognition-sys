"""Pasa LFW por NUESTRO pipeline (YuNet + alineación) y crea los splits.

Uso:
    python scripts/prepare_lfw.py --lfw /ruta/lfw_funneled --out work/lfw_aligned.npz

Guarda las caras alineadas a CANON_SIZE (128x128) para poder probar después
distintos tamaños sin volver a detectar.

Splits (por identidad, nunca se mezcla una persona entre splits):
  - overlap : famosos que también están en el dataset del profe (+ sus gemelos).
              Nunca se usan para entrenar ni calibrar. Sirven de pruebas reales.
  - train   : personas con 1 sola foto en LFW -> entrenamiento del PCA.
  - calib   : mitad de las personas con >=2 fotos -> elegir tamaño/k y calibrar λ.
  - test    : la otra mitad -> evaluación final (no se toca para decidir nada).
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from facerec import config as C  # noqa: E402
from facerec.detector import Detector, FaceError, align_face, load_image  # noqa: E402

# Personas de LFW que son la misma persona que alguien del dataset del profe.
OVERLAP = {
    "Christian_Bale": "christian_bale_2012",
    "Jackie_Chan": "jackie_chan_2000",
    "Keanu_Reeves": "keanu_reeves_2021",
    "Salma_Hayek": "salma_hayek_2012",
    "Mary-Kate_Olsen": "mary-kate_olsen_2011",
    "Michael_Jackson": "michael_jackson_1982",
    "James_Phelps": "james_phelps_(gemelo_weasley)_2011",
    "Serena_Williams": "serenna_williams_2017",
    "Queen_Elizabeth_II": "elisabeth_II_1945",
}
# Gemelos / familiares cercanos de personas inscritas: para el ataque de gemelos.
TWINS = {
    "Ashley_Olsen": "mary-kate_olsen_2011",
    "Oliver_Phelps": "james_phelps_(gemelo_weasley)_2011",
    "Venus_Williams": "serenna_williams_2017",  # hermana (no gemela)
}

_det = None


def _init():
    global _det
    import cv2
    cv2.setNumThreads(1)
    _det = Detector()


def _process(path: str):
    try:
        img = load_image(path)
        faces = _det.detect(img)
        if not faces:
            return path, None
        # En LFW la persona etiquetada es la cara central de la foto.
        h, w = img.shape[:2]
        c = np.array([w / 2, h / 2])
        face = min(faces, key=lambda f: np.linalg.norm(f.box[:2] + f.box[2:] / 2 - c))
        return path, align_face(img, face, C.CANON_SIZE)
    except FaceError:
        return path, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfw", required=True)
    ap.add_argument("--out", default="work/lfw_aligned.npz")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.lfw)
    paths = sorted(str(p) for p in root.glob("*/*.jpg"))
    print(f"{len(paths)} imágenes en LFW")

    with Pool(args.workers, initializer=_init) as pool:
        results = pool.map(_process, paths, chunksize=64)

    ok = [(p, a) for p, a in results if a is not None]
    print(f"detectadas y alineadas: {len(ok)}  /  fallos del detector: {len(results) - len(ok)}")

    names = np.array([Path(p).parent.name for p, _ in ok])
    files = np.array([Path(p).name for p, _ in ok])
    faces = np.stack([a for _, a in ok]).astype(np.uint8)

    # --- splits por identidad ---
    counts = {n: int((names == n).sum()) for n in np.unique(names)}
    special = set(OVERLAP) | set(TWINS)
    rng = np.random.default_rng(args.seed)
    multi = sorted(n for n, c in counts.items() if c >= 2 and n not in special)
    rng.shuffle(multi)
    half = len(multi) // 2
    calib_ids, test_ids = set(multi[:half]), set(multi[half:])

    split = np.empty(len(names), dtype="<U8")
    for i, n in enumerate(names):
        if n in OVERLAP:
            split[i] = "overlap"
        elif n in TWINS:
            split[i] = "twin"
        elif n in calib_ids:
            split[i] = "calib"
        elif n in test_ids:
            split[i] = "test"
        else:
            split[i] = "train"
    for s in ("train", "calib", "test", "overlap", "twin"):
        m = split == s
        print(f"  {s:8s}: {m.sum():6d} imágenes, {len(set(names[m])):5d} personas")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, faces=faces, names=names, files=files, split=split)
    print(f"guardado en {args.out}")


if __name__ == "__main__":
    main()
