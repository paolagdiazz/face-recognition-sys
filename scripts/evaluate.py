"""Evaluación del sistema PCA GO.

    python scripts/evaluate.py --data work/lfw_aligned.npz

Necesita: models/ entrenados (scripts/train.py) y la BD inicializada
(python -m facerec init). Genera docs/results_eval.json y figuras en docs/figs/.

Bloques:
  A. LFW test (nunca usado para decidir nada): verificación y identificación.
  B. Sistema real (50 inscritos): fotos de LFW de 9 personas inscritas como
     genuinos; miles de desconocidos de LFW como impostores.
  C. Gemelos / familiares de inscritos.
  D. Robustez: luz, oclusiones, ruido, desenfoque. PCA GO vs PCA de píxeles.
  E. Ruta _sd (sin detector ni alineación).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from facerec import config as C  # noqa: E402
from facerec.detector import load_image  # noqa: E402
from facerec.igo import cosine_scores, fit_pca, igo_batch  # noqa: E402
from facerec.metrics import cmc, eer, far_frr, pair_scores, rates_at  # noqa: E402
from facerec.system import SistemaBiometrico  # noqa: E402
from scripts.experiments import MAX_PER_ID, cap_per_identity, resize_all  # noqa: E402
from scripts.prepare_lfw import OVERLAP, TWINS  # noqa: E402

FIGS = C.ROOT / "docs" / "figs"
LFW_ROOT_HINT = "lfw_funneled"


def pct(x):
    return round(100 * float(x), 2)


# ---------------------------------------------------------------------------
# A. LFW test
# ---------------------------------------------------------------------------
def block_lfw_test(sysm, faces, names, split, lam):
    te = np.flatnonzero(split == "test")
    te = te[cap_per_identity(names[te], MAX_PER_ID, seed=1)]
    L = names[te]
    Y = sysm.model.project(igo_batch(resize_all(faces[te], C.FACE_SIZE)))
    S = cosine_scores(Y, Y)
    g, i = pair_scores(S, L)
    e, t_eer = eer(g, i)
    far, frr = rates_at(g, i, lam)

    # DET + distribuciones
    t = np.linspace(-1, 1, 2001)
    FAR, FRR = far_frr(g, i, t)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(i, bins=100, alpha=.6, density=True, label="impostores")
    ax[0].hist(g, bins=100, alpha=.6, density=True, label="genuinos")
    ax[0].axvline(lam, color="k", ls="--", label=f"λ={lam:.3f}")
    ax[0].set_xlabel("similitud coseno"); ax[0].set_title("Distribución de scores (LFW test)")
    ax[0].legend()
    ax[1].plot(FAR * 100, FRR * 100)
    ax[1].plot([0, 100], [0, 100], "k:", lw=.8)
    ax[1].scatter([far * 100], [frr * 100], c="r", zorder=3, label=f"λ: FAR={far:.1%} FRR={frr:.1%}")
    ax[1].set_xscale("log"); ax[1].set_xlim(0.01, 100); ax[1].set_ylim(0, 100)
    ax[1].set_xlabel("FAR (%)"); ax[1].set_ylabel("FRR (%)"); ax[1].set_title(f"Curva DET — EER={e:.1%}")
    ax[1].legend(); fig.tight_layout(); fig.savefig(FIGS / "lfw_det.png", dpi=120); plt.close(fig)

    # Identificación: mitad de identidades inscritas (1 foto), la otra mitad desconocidas
    ids = np.unique(L)
    rng = np.random.default_rng(2)
    rng.shuffle(ids)
    enrolled = set(ids[: len(ids) // 2])
    gal_idx, probe_idx, unk_idx = [], [], []
    seen = set()
    for k, n in enumerate(L):
        if n in enrolled:
            if n not in seen:
                gal_idx.append(k); seen.add(n)
            else:
                probe_idx.append(k)
        else:
            unk_idx.append(k)
    G, P, U = Y[gal_idx], Y[probe_idx], Y[unk_idx]
    SG = cosine_scores(P, G)
    curve = cmc(SG, L[probe_idx], L[gal_idx], 20)
    best = SG.max(1)
    correct = L[gal_idx][SG.argmax(1)] == L[probe_idx]
    lam_id = sysm.threshold_ident
    dir_ = float(((best >= lam_id) & correct).mean())       # bien identificado
    miss = float((best < lam_id).mean())                    # "desconocido" siendo inscrito
    fpir = float((cosine_scores(U, G).max(1) >= lam_id).mean())  # desconocido identificado como alguien

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(range(1, 21), curve * 100, marker="o")
    ax.set_xlabel("rango"); ax.set_ylabel("tasa de identificación (%)")
    ax.set_title(f"CMC (LFW test, galería {len(gal_idx)})"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIGS / "lfw_cmc.png", dpi=120); plt.close(fig)

    return {
        "n_images": int(len(te)), "n_people": int(len(ids)),
        "n_genuine": int(len(g)), "n_impostor": int(len(i)),
        "eer": pct(e), "eer_threshold": round(t_eer, 4),
        "far_at_lambda": pct(far), "frr_at_lambda": pct(frr),
        "ident": {"gallery": len(gal_idx), "probes": len(probe_idx), "unknown_probes": len(unk_idx),
                  "rank1_closed": pct(curve[0]), "rank5_closed": pct(curve[4]),
                  "rank10_closed": pct(curve[9]),
                  "nota": "abierta con λ_id (calibrado para galería de 51; aquí la galería es mayor)",
                  "open_correct": pct(dir_), "open_said_unknown": pct(miss), "open_fpir": pct(fpir)},
    }, (Y, L)


# ---------------------------------------------------------------------------
# B. Sistema real con los 50 inscritos
# ---------------------------------------------------------------------------
def block_real_system(sysm, faces, names, split, lam):
    db = sysm.db.load()
    ids = sorted(db)
    G = np.stack([db[i] for i in ids])
    lam_id = sysm.threshold_ident
    out = {"enrolled": len(ids), "per_person": {}}

    # genuinos: fotos de LFW de famosos inscritos
    m = split == "overlap"
    P = sysm.model.project(igo_batch(resize_all(faces[m], C.FACE_SIZE)))
    true_ids = np.array([OVERLAP[n] for n in names[m]])
    S = cosine_scores(P, G)
    gidx = np.array([ids.index(t) for t in true_ids])
    gen = S[np.arange(len(P)), gidx]
    top = np.array(ids)[S.argmax(1)]
    for person in sorted(set(true_ids)):
        k = true_ids == person
        out["per_person"][person] = {
            "n": int(k.sum()),
            "verif_accept": pct((gen[k] >= lam).mean()),
            "ident_rank1": pct((top[k] == person).mean()),
            "ident_open_ok": pct(((top[k] == person) & (S[k].max(1) >= lam_id)).mean()),
            "mean_score": round(float(gen[k].mean()), 3),
        }
    out["genuine_total"] = {
        "n": int(len(P)), "verif_accept": pct((gen >= lam).mean()),
        "ident_rank1": pct((top == true_ids).mean()),
        "ident_open_ok": pct(((top == true_ids) & (S.max(1) >= lam_id)).mean()),
    }

    # impostores: todas las fotos del split test de LFW (gente no inscrita)
    te = split == "test"
    Q = sysm.model.project(igo_batch(resize_all(faces[te], C.FACE_SIZE)))
    SQ = cosine_scores(Q, G)
    best = SQ.max(1)
    who = np.array(ids)[SQ.argmax(1)]
    fp = best >= lam_id
    wolves = {}
    for w in who[fp]:
        wolves[w] = wolves.get(w, 0) + 1
    out["impostors"] = {
        "n": int(te.sum()),
        "fpir": pct(fp.mean()),                                  # identificar() devuelve a alguien (λ_id)
        "fpir_si_se_usara_lambda": pct((best >= lam).mean()),
        "verif_far_per_comparison": pct((SQ >= lam).mean()),     # verificar() contra un id concreto
        "most_matched": dict(sorted(wolves.items(), key=lambda x: -x[1])[:8]),
    }
    return out


# ---------------------------------------------------------------------------
# C. Gemelos
# ---------------------------------------------------------------------------
def block_twins(sysm, faces, names, split, lam):
    db = sysm.db.load()
    res = []
    for k in np.flatnonzero(split == "twin"):
        target = TWINS[names[k]]
        t = sysm.model.project(igo_batch(resize_all(faces[k:k + 1], C.FACE_SIZE)))[0]
        s = sysm.similitud(db[target], t)
        rank = [r[0] for r in sysm.ranking(t)].index(target) + 1
        res.append({"foto": f"{names[k]}", "objetivo": target, "score": round(s, 4),
                    "verificar_acepta": bool(s >= lam), "rango_del_objetivo": rank})
    return res


# ---------------------------------------------------------------------------
# D. Robustez
# ---------------------------------------------------------------------------
def _gamma(f, g):
    return np.clip(255 * (f / 255.0) ** g, 0, 255).astype(np.uint8)


def _side_light(f):
    h, w = f.shape
    ramp = np.linspace(0.25, 1.4, w)[None, :]
    return np.clip(f * ramp, 0, 255).astype(np.uint8)


def _box(f, y0, y1, x0=0.0, x1=1.0, val=0):
    g = f.copy()
    h, w = g.shape
    g[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = val
    return g


TRANSFORMS = {
    "original": lambda f: f,
    "oscura (γ=2.5)": lambda f: _gamma(f, 2.5),
    "clara (γ=0.4)": lambda f: _gamma(f, 0.4),
    "contraste x0.4": lambda f: (f * 0.4 + 30).astype(np.uint8),
    "luz lateral": _side_light,
    "mascarilla": lambda f: _box(f, 0.62, 1.0, 0.1, 0.9, 200),
    "gafas de sol": lambda f: _box(f, 0.25, 0.42, 0.12, 0.88, 10),
    "media cara tapada": lambda f: _box(f, 0.0, 1.0, 0.5, 1.0, 128),
    "ruido σ=15": lambda f: np.clip(f + np.random.default_rng(0).normal(0, 15, f.shape), 0, 255).astype(np.uint8),
    "desenfoque σ=2": lambda f: cv2.GaussianBlur(f, (0, 0), 2),
    "baja resolución (16px)": lambda f: cv2.resize(cv2.resize(f, (16, 16), interpolation=cv2.INTER_AREA),
                                                   f.shape[::-1], interpolation=cv2.INTER_LINEAR),
}


def block_robustness(sysm, faces, names, split, lam):
    """Genuinos: 1ª foto de cada persona (galería, intacta) vs resto (transformadas).
    Impostores: galería vs fotos transformadas de otras personas.
    Se compara PCA GO con PCA de píxeles (eigenfaces) entrenado igual."""
    te = np.flatnonzero(split == "test")
    te = te[cap_per_identity(names[te], 4, seed=3)]
    L = names[te]
    F64 = resize_all(faces[te], C.FACE_SIZE)
    first = {}
    for k, n in enumerate(L):
        first.setdefault(n, k)
    gal = np.array(sorted(first.values()))
    probe = np.array([k for k in range(len(L)) if k not in set(gal)])
    gl, pl = L[gal], L[probe]

    # baseline eigenfaces (mismo split de entrenamiento, k=150, whitening)
    tr = split == "train"
    pix_tr = resize_all(faces[tr], C.FACE_SIZE).reshape(tr.sum(), -1).astype(np.float32)
    pix_model = fit_pca(pix_tr, 150, C.FACE_SIZE, True)

    def scores(feat_g, feat_p):
        S = cosine_scores(feat_p, feat_g)
        same = pl[:, None] == gl[None, :]
        return S[same], S[~same]

    # λ de eigenfaces a FAR=1 % en las fotos originales (para comparar en igualdad)
    from facerec.metrics import threshold_at_far
    res = {}
    igo_g = sysm.model.project(igo_batch(F64[gal]))
    pix_g = pix_model.project(F64[gal].reshape(len(gal), -1).astype(np.float32))
    pix_lam = None
    examples = []
    for name, fn in TRANSFORMS.items():
        Fp = np.stack([fn(f) for f in F64[probe]])
        examples.append(Fp[0])
        g1, i1 = scores(igo_g, sysm.model.project(igo_batch(Fp)))
        g2, i2 = scores(pix_g, pix_model.project(Fp.reshape(len(Fp), -1).astype(np.float32)))
        if pix_lam is None:
            pix_lam = threshold_at_far(i2, 0.01)
        res[name] = {
            "pca_go": {"eer": pct(eer(g1, i1)[0]), "tar_at_lambda": pct((g1 >= lam).mean()),
                       "far_at_lambda": pct((i1 >= lam).mean())},
            "pca_pixeles": {"eer": pct(eer(g2, i2)[0]), "tar_at_lambda": pct((g2 >= pix_lam).mean()),
                            "far_at_lambda": pct((i2 >= pix_lam).mean())},
        }
    # figura de ejemplos
    fig, ax = plt.subplots(1, len(examples), figsize=(1.6 * len(examples), 2))
    for a, im, n in zip(ax, examples, TRANSFORMS):
        a.imshow(im, cmap="gray", vmin=0, vmax=255); a.set_title(n, fontsize=6); a.axis("off")
    fig.tight_layout(); fig.savefig(FIGS / "robustez_ejemplos.png", dpi=130); plt.close(fig)
    # figura comparativa
    fig, ax = plt.subplots(figsize=(9, 3.6))
    xs = np.arange(len(res))
    ax.bar(xs - .2, [r["pca_go"]["tar_at_lambda"] for r in res.values()], .4, label="PCA GO")
    ax.bar(xs + .2, [r["pca_pixeles"]["tar_at_lambda"] for r in res.values()], .4, label="PCA píxeles")
    ax.set_xticks(xs); ax.set_xticklabels(list(res), rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("aceptación de genuinos a FAR=1 % (%)"); ax.legend(); ax.grid(axis="y", alpha=.3)
    ax.set_title("Robustez: PCA GO vs PCA normal (LFW test)")
    fig.tight_layout(); fig.savefig(FIGS / "robustez.png", dpi=120); plt.close(fig)
    return {"n_gallery": int(len(gal)), "n_probes": int(len(probe)), "results": res}


# ---------------------------------------------------------------------------
# E. Ruta _sd
# ---------------------------------------------------------------------------
def block_sd(sysm, lfw_dir: Path, lam):
    """Para cada foto de LFW de un famoso inscrito: recorte del detector SIN alinear
    (lo que haría un atacante/cliente que llama a *_sd) y se pasa por verificar_sd.
    Además, ¿cuántas imágenes que no son caras identifica identificar_sd?"""
    res = {}
    if lfw_dir and lfw_dir.is_dir():
        acc_sd, acc_full, n = 0, 0, 0
        for lfw_name, pid in OVERLAP.items():
            for p in sorted((lfw_dir / lfw_name).glob("*.jpg")):
                img = load_image(p)
                f = sysm.detector.best_face(img)
                x, y, w, h = [int(round(v)) for v in f.box]
                crop = img[max(y, 0):y + h, max(x, 0):x + w]
                acc_sd += sysm.verificar_sd(pid, crop)
                acc_full += sysm.verificar(pid, img)
                n += 1
        res["genuinos_lfw"] = {"n": n, "acepta_con_detector": pct(acc_full / n),
                               "acepta_sd_recorte_sin_alinear": pct(acc_sd / n)}
    rng = np.random.default_rng(0)
    probes = {
        "gris uniforme": np.full((64, 64), 128, np.uint8),
        "negro": np.zeros((64, 64), np.uint8),
        "ruido": rng.integers(0, 256, (64, 64), dtype=np.uint8),
        "degradado horizontal": np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1)),
        "tablero de ajedrez": (np.indices((64, 64)).sum(0) // 8 % 2 * 255).astype(np.uint8),
    }
    nf = {}
    for name, im in probes.items():
        r = sysm.ranking(sysm.extraer_sd(im))
        nf[name] = {"identificar_sd": r[0][0] if r[0][1] >= sysm.threshold_ident else None,
                    "mas_parecido": r[0][0], "mejor_score": round(r[0][1], 4)}
    res["no_caras"] = nf
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="work/lfw_aligned.npz")
    ap.add_argument("--lfw", default=None, help="carpeta lfw_funneled (para el bloque _sd)")
    args = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)

    sysm = SistemaBiometrico()
    lam = sysm.threshold
    z = np.load(args.data)
    faces, names, split = z["faces"], z["names"], z["split"]

    out = {"lambda": lam, "lambda_ident": sysm.threshold_ident}
    out["A_lfw_test"], _ = block_lfw_test(sysm, faces, names, split, lam)
    print("A", json.dumps(out["A_lfw_test"], ensure_ascii=False), flush=True)
    out["B_sistema_real"] = block_real_system(sysm, faces, names, split, lam)
    print("B", json.dumps(out["B_sistema_real"], ensure_ascii=False), flush=True)
    out["C_gemelos"] = block_twins(sysm, faces, names, split, lam)
    print("C", json.dumps(out["C_gemelos"], ensure_ascii=False), flush=True)
    out["D_robustez"] = block_robustness(sysm, faces, names, split, lam)
    print("D", json.dumps(out["D_robustez"], ensure_ascii=False), flush=True)
    out["E_sd"] = block_sd(sysm, Path(args.lfw) if args.lfw else None, lam)
    print("E", json.dumps(out["E_sd"], ensure_ascii=False), flush=True)

    (C.ROOT / "docs" / "results_eval.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("guardado docs/results_eval.json")


if __name__ == "__main__":
    main()
