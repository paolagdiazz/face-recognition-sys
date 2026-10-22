"""Carga de imágenes, detector de caras (YuNet) y alineación por los ojos."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import config as C


class FaceError(ValueError):
    """La imagen no es válida o no contiene una cara utilizable."""


# --------------------------------------------------------------------------
# Carga de imágenes
# --------------------------------------------------------------------------
def load_image(foto) -> np.ndarray:
    """Acepta ruta, bytes o array (BGR/gris). Devuelve imagen BGR uint8."""
    if isinstance(foto, np.ndarray):
        img = foto
    else:
        if isinstance(foto, (str, Path)):
            p = Path(foto)
            if not p.is_file():
                raise FaceError(f"no existe el fichero: {p}")
            if p.stat().st_size > C.MAX_IMAGE_BYTES:
                raise FaceError("imagen demasiado grande")
            data = np.fromfile(str(p), dtype=np.uint8)  # soporta rutas no ASCII
        elif isinstance(foto, (bytes, bytearray)):
            if len(foto) > C.MAX_IMAGE_BYTES:
                raise FaceError("imagen demasiado grande")
            data = np.frombuffer(foto, dtype=np.uint8)
        else:
            raise FaceError(f"tipo de foto no soportado: {type(foto).__name__}")
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise FaceError("no se pudo decodificar la imagen")
    if img.dtype != np.uint8:
        raise FaceError("la imagen debe ser uint8")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif not (img.ndim == 3 and img.shape[2] == 3):
        raise FaceError("formato de imagen no soportado")
    h, w = img.shape[:2]
    if h < 1 or w < 1 or max(h, w) > C.MAX_IMAGE_SIDE:
        raise FaceError(f"dimensiones de imagen no válidas: {w}x{h}")
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------
@dataclass
class Face:
    box: np.ndarray        # x, y, w, h (coordenadas de la imagen original)
    landmarks: np.ndarray  # 5x2: ojo, ojo, nariz, comisura, comisura
    score: float

    @property
    def area(self) -> float:
        return float(self.box[2] * self.box[3])

    @property
    def eyes(self) -> tuple[np.ndarray, np.ndarray]:
        """(ojo a la izquierda de la imagen, ojo a la derecha de la imagen)."""
        a, b = self.landmarks[0], self.landmarks[1]
        return (a, b) if a[0] <= b[0] else (b, a)


def _check_model(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"falta el modelo del detector: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != C.YUNET_SHA256:
        raise RuntimeError("el modelo del detector no coincide con el hash esperado")


class Detector:
    def __init__(self, model_path: Path = C.YUNET_PATH,
                 score_threshold: float = C.DET_SCORE_THRESHOLD):
        _check_model(Path(model_path))
        self.score_threshold = score_threshold
        self._net = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), score_threshold, C.DET_NMS_THRESHOLD, 5000)

    def detect(self, img: np.ndarray) -> list[Face]:
        h, w = img.shape[:2]
        scale = min(1.0, C.DET_MAX_SIDE / max(h, w))
        small = img if scale == 1.0 else cv2.resize(
            img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        if small.ndim == 2:
            small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        sh, sw = small.shape[:2]
        self._net.setInputSize((sw, sh))
        _, out = self._net.detect(small)
        faces = []
        if out is not None:
            for r in out:
                score = float(r[14])
                if score < self.score_threshold:
                    continue
                box = r[0:4] / scale
                lms = r[4:14].reshape(5, 2) / scale
                faces.append(Face(box.astype(np.float64), lms.astype(np.float64), score))
        return faces

    def best_face(self, img: np.ndarray) -> Face:
        """Política: de todas las caras válidas se coge la de mayor área."""
        faces = self.detect(img)
        if not faces:
            raise FaceError("no se ha detectado ninguna cara")
        return max(faces, key=lambda f: f.area)


# --------------------------------------------------------------------------
# Alineación
# --------------------------------------------------------------------------
def _similarity(src_l, src_r, dst_l, dst_r) -> np.ndarray:
    """Matriz 2x3 (rotación + escala + traslación) que lleva src -> dst."""
    sv = np.asarray(src_r, float) - np.asarray(src_l, float)
    dv = np.asarray(dst_r, float) - np.asarray(dst_l, float)
    s = np.linalg.norm(dv) / max(np.linalg.norm(sv), 1e-6)
    ang = np.arctan2(dv[1], dv[0]) - np.arctan2(sv[1], sv[0])
    c, si = s * np.cos(ang), s * np.sin(ang)
    R = np.array([[c, -si], [si, c]])
    t = np.asarray(dst_l, float) - R @ np.asarray(src_l, float)
    return np.hstack([R, t[:, None]])


def align_face(img: np.ndarray, face: Face, size: int = C.FACE_SIZE) -> np.ndarray:
    """Recorta y alinea la cara: ojos horizontales y en posiciones fijas.

    Se alinea a un lienzo canónico de CANON_SIZE y después se reduce a `size`
    con INTER_AREA, para no meter aliasing al reducir caras grandes.
    """
    gray = to_gray(img)
    el, er = face.eyes
    iod = float(np.linalg.norm(er - el))
    if iod < 4:
        raise FaceError("cara demasiado pequeña")
    canon = C.CANON_SIZE
    dst_l = np.array(C.EYE_LEFT) * canon
    dst_r = np.array(C.EYE_RIGHT) * canon
    target_iod = float(np.linalg.norm(dst_r - dst_l))

    # Si la cara es mucho mayor que el lienzo, se reduce antes (anti-aliasing).
    pre = target_iod / iod
    if pre < 0.5:
        # recorta una zona alrededor de la cara para no reducir la imagen entera
        cx, cy = (el + er) / 2
        half = 4 * iod
        x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
        x1, y1 = int(min(cx + half, gray.shape[1])), int(min(cy + half, gray.shape[0]))
        crop = gray[y0:y1, x0:x1]
        f = pre * 1.5
        crop = cv2.resize(crop, (max(1, round(crop.shape[1] * f)), max(1, round(crop.shape[0] * f))),
                          interpolation=cv2.INTER_AREA)
        el = (el - [x0, y0]) * f
        er = (er - [x0, y0]) * f
        gray = crop

    M = _similarity(el, er, dst_l, dst_r)
    aligned = cv2.warpAffine(gray, M, (canon, canon), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    if size != canon:
        interp = cv2.INTER_AREA if size < canon else cv2.INTER_LINEAR
        aligned = cv2.resize(aligned, (size, size), interpolation=interp)
    return aligned


def crop_without_detector(img: np.ndarray, size: int = C.FACE_SIZE) -> np.ndarray:
    """Ruta _sd: se asume que la imagen YA es la cara. Sin alineación posible."""
    gray = to_gray(img)
    h, w = gray.shape
    if min(h, w) < C.MIN_SD_SIDE:
        raise FaceError(f"imagen _sd demasiado pequeña (mínimo {C.MIN_SD_SIDE}px)")
    interp = cv2.INTER_AREA if min(h, w) >= size else cv2.INTER_LINEAR
    return cv2.resize(gray, (size, size), interpolation=interp)
