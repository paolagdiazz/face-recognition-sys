"""Sistema de reconocimiento facial con PCA GO (PCA on Image Gradient Orientations)."""
import os as _os

# Límite de píxeles que OpenCV acepta al decodificar. Debe fijarse ANTES de que
# se importe cv2 (por eso va aquí, antes de importar nada del paquete). Evita
# que un PNG diminuto que declara 30000x30000 px reserve gigas de memoria.
from . import config as _C
_os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(_C.MAX_IMAGE_SIDE ** 2))

from .system import SistemaBiometrico, TemplateError  # noqa: E402,F401
from .detector import FaceError  # noqa: E402,F401
from .db import DBError, DBIntegrityError  # noqa: E402,F401
from .ratelimit import RateLimitError  # noqa: E402,F401
