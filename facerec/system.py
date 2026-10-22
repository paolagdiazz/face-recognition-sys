"""Sistema de reconocimiento facial PCA GO con los 9 métodos del enunciado.

    from facerec import SistemaBiometrico
    s = SistemaBiometrico()
    s.inscribir("ana", "foto.jpg")
    s.verificar("ana", "otra.jpg")      -> True / False
    s.identificar("otra.jpg")           -> "ana" / None

Variantes:
    *_sd : se salta el detector (la foto debe ser ya la cara recortada).
    *_t  : se salta detector y extractor (se pasa el template directamente).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import config as C
from .db import DBError, TemplateDB, load_key, validate_id
from .detector import Detector, FaceError, align_face, crop_without_detector, load_image
from .igo import PCAModel, igo_vector

__all__ = ["SistemaBiometrico", "FaceError", "DBError", "TemplateError"]

TEMPLATE_ABS_MAX = 1e3   # los templates (whitened) tienen componentes ~N(0,1)


class TemplateError(ValueError):
    pass


class SistemaBiometrico:
    def __init__(self, db_path: str | Path | None = None, key: bytes | None = None,
                 threshold: float | None = None):
        info = json.loads(C.THRESHOLD_PATH.read_text())
        digest = hashlib.sha256(C.PCA_MODEL_PATH.read_bytes()).hexdigest()
        if digest != info["pca"]["sha256"]:
            raise RuntimeError("models/pca_igo.npz no es el modelo con el que se calibró λ "
                               "(hash distinto). Vuelve a ejecutar scripts/train.py")
        self.model = PCAModel.load(C.PCA_MODEL_PATH)
        # λ    : verificar()   (FAR = 1 % por comparación)
        # λ_id : identificar() (FPIR = 1 % con 51 inscritos)
        self.threshold = float(info["lambda"] if threshold is None else threshold)
        self.threshold_ident = float(info.get("lambda_ident", self.threshold)
                                     if threshold is None else threshold)
        self.detector = Detector()
        db_path = Path(db_path or os.environ.get(C.DB_PATH_ENV_VAR) or C.DEFAULT_DB_PATH)
        self.db = TemplateDB(db_path, key if key is not None else load_key(), self.model.k)

    # ------------------------------------------------------------------
    # Extracción
    # ------------------------------------------------------------------
    def _template_from_face(self, face: np.ndarray) -> np.ndarray:
        return self.model.project(igo_vector(face))[0]

    def extraer(self, foto) -> np.ndarray:
        """foto -> detector -> alineación -> PCA GO -> template."""
        img = load_image(foto)
        face = self.detector.best_face(img)
        return self._template_from_face(align_face(img, face, self.model.face_size))

    def extraer_sd(self, foto) -> np.ndarray:
        """foto (ya recortada) -> PCA GO -> template. Sin detector ni alineación."""
        img = load_image(foto)
        return self._template_from_face(crop_without_detector(img, self.model.face_size))

    def validar_template(self, template) -> np.ndarray:
        if isinstance(template, str):
            try:
                template = [float(x) for x in template.replace(";", ",").split(",") if x.strip()]
            except ValueError as e:
                raise TemplateError("template con valores no numéricos") from e
        try:
            t = np.asarray(template, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise TemplateError("template no numérico") from e
        if t.ndim != 1 or t.shape[0] != self.model.k:
            raise TemplateError(f"el template debe tener exactamente {self.model.k} valores")
        if not np.all(np.isfinite(t)):
            raise TemplateError("el template contiene NaN o infinitos")
        if np.max(np.abs(t)) > TEMPLATE_ABS_MAX:
            raise TemplateError("valores del template fuera de rango")
        if np.linalg.norm(t) < 1e-6:
            raise TemplateError("template nulo")
        return t.astype(np.float32)

    # ------------------------------------------------------------------
    # Comparación
    # ------------------------------------------------------------------
    @staticmethod
    def similitud(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, np.float64)
        b = np.asarray(b, np.float64)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    def puntuar(self, pid: str, template: np.ndarray) -> float:
        """Score coseno contra el template inscrito (solo para evaluación)."""
        db = self.db.load()
        if pid not in db:
            raise KeyError(f"id no inscrito: {pid!r}")
        return self.similitud(db[pid], template)

    def ranking(self, template: np.ndarray) -> list[tuple[str, float]]:
        """Todos los inscritos ordenados por similitud (solo para evaluación)."""
        db = self.db.load()
        if not db:
            return []
        ids = list(db)
        G = np.stack([db[i] for i in ids]).astype(np.float64)
        t = np.asarray(template, np.float64)
        s = G @ t / (np.linalg.norm(G, axis=1) * np.linalg.norm(t) + 1e-12)
        order = np.argsort(-s)
        return [(ids[i], float(s[i])) for i in order]

    # ------------------------------------------------------------------
    # Operaciones genéricas
    # ------------------------------------------------------------------
    def _inscribir(self, pid: str, template: np.ndarray) -> np.ndarray:
        validate_id(pid)
        db = self.db.load()
        if pid in db:
            raise ValueError(f"el id {pid!r} ya está inscrito (bórralo antes con `borrar`)")
        db[pid] = template
        self.db.save(db)
        return template

    def _verificar(self, pid: str, template: np.ndarray) -> bool:
        validate_id(pid)
        db = self.db.load()
        if pid not in db:
            return False          # id desconocido = rechazo (no se distingue del fallo)
        return self.similitud(db[pid], template) >= self.threshold

    def _identificar(self, template: np.ndarray) -> str | None:
        r = self.ranking(template)
        if not r or r[0][1] < self.threshold_ident:
            return None           # identificación abierta: "desconocido"
        return r[0][0]

    # ------------------------------------------------------------------
    # Los 9 métodos del enunciado
    # ------------------------------------------------------------------
    def inscribir(self, id: str, foto) -> np.ndarray:
        return self._inscribir(id, self.extraer(foto))

    def verificar(self, id: str, foto) -> bool:
        return self._verificar(id, self.extraer(foto))

    def identificar(self, foto) -> str | None:
        return self._identificar(self.extraer(foto))

    def inscribir_sd(self, id: str, foto) -> np.ndarray:
        return self._inscribir(id, self.extraer_sd(foto))

    def verificar_sd(self, id: str, foto) -> bool:
        return self._verificar(id, self.extraer_sd(foto))

    def identificar_sd(self, foto) -> str | None:
        return self._identificar(self.extraer_sd(foto))

    def inscribir_t(self, id: str, template) -> np.ndarray:
        return self._inscribir(id, self.validar_template(template))

    def verificar_t(self, id: str, template) -> bool:
        return self._verificar(id, self.validar_template(template))

    def identificar_t(self, template) -> str | None:
        return self._identificar(self.validar_template(template))

    # ------------------------------------------------------------------
    # Extras
    # ------------------------------------------------------------------
    def borrar(self, id: str) -> None:
        validate_id(id)
        db = self.db.load()
        if id not in db:
            raise KeyError(f"id no inscrito: {id!r}")
        del db[id]
        self.db.save(db)

    def inscritos(self) -> list[str]:
        return sorted(self.db.load())
