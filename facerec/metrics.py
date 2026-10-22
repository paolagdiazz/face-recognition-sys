"""Métricas de evaluación biométrica (scores = similitud, mayor = más parecido)."""
from __future__ import annotations

import numpy as np


def far_frr(genuine: np.ndarray, impostor: np.ndarray, thresholds: np.ndarray):
    """FAR(t) = P(impostor >= t), FRR(t) = P(genuino < t)."""
    g = np.sort(genuine)
    i = np.sort(impostor)
    frr = np.searchsorted(g, thresholds, side="left") / len(g)
    far = 1 - np.searchsorted(i, thresholds, side="left") / len(i)
    return far, frr


def eer(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """Devuelve (EER, umbral en el EER)."""
    t = np.unique(np.concatenate([genuine, impostor]))
    far, frr = far_frr(genuine, impostor, t)
    idx = int(np.argmin(np.abs(far - frr)))
    return float((far[idx] + frr[idx]) / 2), float(t[idx])


def threshold_at_far(impostor: np.ndarray, target_far: float) -> float:
    """Umbral más bajo con FAR <= target_far (aceptar si score >= umbral)."""
    i = np.sort(impostor)
    n = len(i)
    k = int(np.floor(target_far * n))          # nº de impostores que se permite aceptar
    # aceptar si score >= t ; con t justo por encima del (n-k)-ésimo impostor
    t = np.nextafter(i[n - k - 1], np.inf) if k < n else i[0]
    return float(t)


def rates_at(genuine, impostor, t: float) -> tuple[float, float]:
    far = float((np.asarray(impostor) >= t).mean())
    frr = float((np.asarray(genuine) < t).mean())
    return far, frr


def pair_scores(S: np.ndarray, labels: np.ndarray):
    """De una matriz de similitud NxN entre muestras -> (genuinos, impostores)
    usando solo el triángulo superior (cada par una vez)."""
    iu = np.triu_indices(len(labels), k=1)
    same = labels[iu[0]] == labels[iu[1]]
    s = S[iu]
    return s[same], s[~same]


def cmc(S: np.ndarray, probe_labels: np.ndarray, gallery_labels: np.ndarray, max_rank: int = 20):
    """Curva CMC. S: (n_probes, n_gallery). Solo probes cuya identidad está en la galería."""
    order = np.argsort(-S, axis=1)
    ranked = gallery_labels[order]
    hits = ranked == probe_labels[:, None]
    first = np.where(hits.any(1), hits.argmax(1), np.iinfo(int).max)
    return np.array([(first < r).mean() for r in range(1, max_rank + 1)])
