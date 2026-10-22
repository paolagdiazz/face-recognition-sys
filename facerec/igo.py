"""Extractor PCA GO (PCA on Image Gradient Orientations, Tzimiropoulos et al. 2012).

cara (gris, alineada) -> blur -> Sobel (Gx, Gy) -> φ = atan2(Gy, Gx)
-> [cos φ, sin φ] por píxel -> vector de 2·N² -> proyección PCA -> template
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import config as C


def igo_vector(face: np.ndarray, blur_sigma: float = C.BLUR_SIGMA,
               ksize: int = C.SOBEL_KSIZE) -> np.ndarray:
    """Cara gris (uint8, NxN) -> vector IGO [cos..., sin...] (float32, 2·N²)."""
    f = face.astype(np.float32)
    if blur_sigma > 0:
        f = cv2.GaussianBlur(f, (0, 0), blur_sigma)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=ksize)
    phi = np.arctan2(gy, gx)
    return np.concatenate([np.cos(phi).ravel(), np.sin(phi).ravel()]).astype(np.float32)


def igo_batch(faces: np.ndarray, **kw) -> np.ndarray:
    return np.stack([igo_vector(f, **kw) for f in faces])


@dataclass
class PCAModel:
    mean: np.ndarray        # (D,)
    components: np.ndarray  # (k, D), filas ortonormales
    eigvals: np.ndarray     # (k,)
    face_size: int
    whiten: bool

    @property
    def k(self) -> int:
        return self.components.shape[0]

    def project(self, X: np.ndarray) -> np.ndarray:
        Y = (np.atleast_2d(X) - self.mean) @ self.components.T
        if self.whiten:
            Y = Y / np.sqrt(self.eigvals)
        return Y.astype(np.float32)

    def reconstruct(self, Y: np.ndarray) -> np.ndarray:
        """Template -> vector IGO aproximado (se usa en el ataque de lectura de BD)."""
        Y = np.atleast_2d(Y)
        if self.whiten:
            Y = Y * np.sqrt(self.eigvals)
        return Y @ self.components + self.mean

    def truncate(self, k: int) -> "PCAModel":
        return PCAModel(self.mean, self.components[:k], self.eigvals[:k], self.face_size, self.whiten)

    # --- persistencia ---
    def save(self, path: Path) -> None:
        np.savez_compressed(path, mean=self.mean.astype(np.float32),
                            components=self.components.astype(np.float32),
                            eigvals=self.eigvals.astype(np.float64),
                            meta=json.dumps({"face_size": self.face_size, "whiten": self.whiten}))

    @classmethod
    def load(cls, path: Path) -> "PCAModel":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return cls(z["mean"], z["components"], z["eigvals"], meta["face_size"], meta["whiten"])


def fit_pca(X: np.ndarray, k: int, face_size: int, whiten: bool) -> PCAModel:
    """PCA por el truco de la matriz de Gram (N < D)."""
    X = X.astype(np.float64)
    mean = X.mean(0)
    Xc = X - mean
    n = Xc.shape[0]
    G = Xc @ Xc.T / (n - 1)
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1][:k]
    w, V = w[order], V[:, order]
    comps = (Xc.T @ V) / np.sqrt(w * (n - 1))   # (D, k), columnas unitarias
    return PCAModel(mean.astype(np.float32), comps.T.astype(np.float32), w, face_size, whiten)


def cosine_scores(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Similitud coseno entre filas de A y filas de B (matriz |A|x|B|)."""
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return A @ B.T
