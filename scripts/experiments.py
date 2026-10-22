"""Experimentos para elegir tamaño de cara, nº de componentes, whitening y blur.

Entrena el PCA con el split `train` de LFW y mide el EER en el split `calib`.
El split `test` NO se usa aquí (se reserva para la evaluación final).

    python scripts/experiments.py --data work/lfw_aligned.npz --out docs/experiments.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from facerec.igo import cosine_scores, fit_pca, igo_batch  # noqa: E402
from facerec.metrics import eer, pair_scores  # noqa: E402

MAX_PER_ID = 10


def resize_all(faces, size):
    if size == faces.shape[1]:
        return faces
    return np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in faces])


def cap_per_identity(names, max_per_id, seed=0):
    rng = np.random.default_rng(seed)
    keep = []
    for n in np.unique(names):
        idx = np.flatnonzero(names == n)
        if len(idx) > max_per_id:
            idx = rng.choice(idx, max_per_id, replace=False)
        keep.extend(idx.tolist())
    return np.array(sorted(keep))


def evaluate(feat_train, feat_calib, labels, size, ks, whiten):
    model = fit_pca(feat_train, max(ks), size, whiten)
    out = {}
    for k in ks:
        m = model.truncate(k)
        Y = m.project(feat_calib)
        g, i = pair_scores(cosine_scores(Y, Y), labels)
        out[k] = eer(g, i)[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="work/lfw_aligned.npz")
    ap.add_argument("--out", default="docs/experiments.json")
    args = ap.parse_args()

    z = np.load(args.data)
    faces, names, split = z["faces"], z["names"], z["split"]
    tr = split == "train"
    ca_idx = np.flatnonzero(split == "calib")
    ca_idx = ca_idx[cap_per_identity(names[ca_idx], MAX_PER_ID)]
    labels = names[ca_idx]
    print(f"train: {tr.sum()} imágenes | calib: {len(ca_idx)} imágenes, {len(set(labels))} personas")

    ks = [25, 50, 100, 150, 200, 300, 400]
    results = {"size_k_whiten": [], "blur": [], "baseline_pixels": []}

    # 1) tamaño x k x whitening (blur 0.8 en esta fase)
    for size in [32, 48, 64, 80]:
        ftr = igo_batch(resize_all(faces[tr], size), blur_sigma=0.8)
        fca = igo_batch(resize_all(faces[ca_idx], size), blur_sigma=0.8)
        for whiten in [False, True]:
            r = evaluate(ftr, fca, labels, size, ks, whiten)
            for k, e in r.items():
                results["size_k_whiten"].append({"size": size, "k": k, "whiten": whiten, "eer": e})
            print(f"size={size:3d} whiten={whiten!s:5s} " +
                  " ".join(f"k{k}={e*100:5.2f}%" for k, e in r.items()), flush=True)

    # 2) blur previo al Sobel (a 64x64)
    for sigma in [0.0, 0.5, 0.8, 1.2, 1.5, 2.0]:
        ftr = igo_batch(resize_all(faces[tr], 64), blur_sigma=sigma)
        fca = igo_batch(resize_all(faces[ca_idx], 64), blur_sigma=sigma)
        r = evaluate(ftr, fca, labels, 64, [150], True)
        results["blur"].append({"sigma": sigma, "eer": r[150]})
        print(f"blur sigma={sigma}: EER={r[150]*100:.2f}%", flush=True)

    # 3) línea base: PCA normal sobre píxeles (eigenfaces) a 64x64
    def pix(f):
        x = resize_all(f, 64).reshape(len(f), -1).astype(np.float32)
        return x
    for whiten in [False, True]:
        r = evaluate(pix(faces[tr]), pix(faces[ca_idx]), labels, 64, [150], whiten)
        results["baseline_pixels"].append({"whiten": whiten, "eer": r[150]})
        print(f"PCA píxeles (eigenfaces) whiten={whiten}: EER={r[150]*100:.2f}%", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"guardado en {args.out}")


if __name__ == "__main__":
    main()
