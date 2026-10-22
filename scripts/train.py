"""Entrena el PCA definitivo y calibra el umbral λ.

    python scripts/train.py --data work/lfw_aligned.npz

- PCA: split `train` de LFW (personas con 1 sola foto), parámetros de config.py.
- λ  : split `calib` de LFW, fijado para FAR = 1 % (se guarda también el del EER).
Salida: models/pca_igo.npz y models/threshold.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from facerec import config as C  # noqa: E402
from facerec.igo import cosine_scores, fit_pca, igo_batch  # noqa: E402
from facerec.metrics import eer, pair_scores, rates_at, threshold_at_far  # noqa: E402
from scripts.experiments import MAX_PER_ID, cap_per_identity, resize_all  # noqa: E402

N_COMPONENTS = 150
WHITEN = True
TARGET_FAR = 0.01
TARGET_FPIR = 0.01
GALLERY_SIZE = 51
TRIALS_IDENT = 20000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="work/lfw_aligned.npz")
    args = ap.parse_args()

    z = np.load(args.data)
    faces, names, split = z["faces"], z["names"], z["split"]

    tr = split == "train"
    X = igo_batch(resize_all(faces[tr], C.FACE_SIZE))
    model = fit_pca(X, N_COMPONENTS, C.FACE_SIZE, WHITEN)
    C.PCA_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(C.PCA_MODEL_PATH)
    var = model.eigvals.sum() / np.var(X.astype(np.float64), axis=0, ddof=1).sum()
    print(f"PCA entrenado con {tr.sum()} caras, k={model.k}, varianza explicada={var:.1%}")

    ca = np.flatnonzero(split == "calib")
    ca = ca[cap_per_identity(names[ca], MAX_PER_ID)]
    Y = model.project(igo_batch(resize_all(faces[ca], C.FACE_SIZE)))
    g, i = pair_scores(cosine_scores(Y, Y), names[ca])
    e, t_eer = eer(g, i)
    lam = threshold_at_far(i, TARGET_FAR)
    far, frr = rates_at(g, i, lam)
    print(f"calib: {len(g)} pares genuinos, {len(i)} impostores")
    print(f"EER={e:.2%} (umbral {t_eer:.4f})")
    print(f"λ={lam:.4f} -> FAR={far:.2%} FRR={frr:.2%}")

    # --- λ_id para identificar(): con 51 inscritos, un desconocido se compara
    # contra 51 templates, así que el FAR por comparación se acumula. Se calibra
    # para que FPIR (desconocido identificado como alguien) = 1 %.
    rng = np.random.default_rng(0)
    L = names[ca]
    S = cosine_scores(Y, Y)
    first = {}
    for k, n in enumerate(L):
        first.setdefault(n, k)
    reps = np.array(list(first.values()))          # una foto por persona
    rep_names = L[reps]
    imp_max, gen_ok = [], []
    for _ in range(TRIALS_IDENT):
        p = rng.integers(len(L))
        others = reps[rep_names != L[p]]
        gal = rng.choice(others, GALLERY_SIZE, replace=False)
        imp_max.append(S[p, gal].max())            # p no está inscrito
    imp_max = np.array(imp_max)
    lam_id = threshold_at_far(imp_max, TARGET_FPIR)
    # tasa de identificación correcta estimada con λ_id (p inscrito con otra foto)
    multi = [k for k in range(len(L)) if k not in set(reps)]
    for _ in range(TRIALS_IDENT):
        p = multi[rng.integers(len(multi))]
        own = first[L[p]]
        others = reps[rep_names != L[p]]
        gal = np.concatenate([[own], rng.choice(others, GALLERY_SIZE - 1, replace=False)])
        s = S[p, gal]
        gen_ok.append(s.argmax() == 0 and s.max() >= lam_id)
    fpir_lam = float((imp_max >= lam).mean())
    print(f"identificación (galería {GALLERY_SIZE}): con λ FPIR={fpir_lam:.1%}; "
          f"λ_id={lam_id:.4f} -> FPIR={TARGET_FPIR:.0%}, aciertos={np.mean(gen_ok):.1%}")

    info = {
        "lambda": lam,
        "lambda_ident": lam_id,
        "ident": {"gallery_size": GALLERY_SIZE, "target_fpir": TARGET_FPIR,
                  "fpir_if_using_lambda": fpir_lam, "correct_rate_calib": float(np.mean(gen_ok))},
        "metric": "cosine",
        "accept_if": "score >= lambda",
        "target_far": TARGET_FAR,
        "calib": {"far": far, "frr": frr, "eer": e, "eer_threshold": t_eer,
                  "n_genuine": int(len(g)), "n_impostor": int(len(i))},
        "pca": {"k": model.k, "whiten": WHITEN, "face_size": C.FACE_SIZE,
                "blur_sigma": C.BLUR_SIGMA, "explained_variance": float(var),
                "n_train": int(tr.sum()),
                "sha256": hashlib.sha256(C.PCA_MODEL_PATH.read_bytes()).hexdigest()},
    }
    C.THRESHOLD_PATH.write_text(json.dumps(info, indent=2))
    print(f"guardado {C.PCA_MODEL_PATH.name} y {C.THRESHOLD_PATH.name}")


if __name__ == "__main__":
    main()
