"""Parámetros del sistema. Todo lo que se decidió está aquí, en un único sitio."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# --- Detector (YuNet, incluido en OpenCV) ---------------------------------
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
DET_SCORE_THRESHOLD = 0.7   # confianza mínima de una cara
DET_NMS_THRESHOLD = 0.3
DET_MAX_SIDE = 1280         # se reduce la imagen antes de detectar si es mayor

# --- Alineación ----------------------------------------------------------
# Recorte canónico grande (se alinea aquí y luego se reduce con INTER_AREA).
CANON_SIZE = 128
# Posición de los ojos en coordenadas normalizadas [0,1] del recorte final.
# Recorte ajustado: de cejas a barbilla, sin pelo ni fondo.
EYE_LEFT = (0.30, 0.33)    # ojo que aparece a la izquierda de la imagen
EYE_RIGHT = (0.70, 0.33)

# --- Extractor PCA GO (IGO-PCA) -------------------------------------------
FACE_SIZE = 64              # lado de la cara alineada (vector = 2*64*64)
BLUR_SIGMA = 1.2            # blur gaussiano previo al Sobel
SOBEL_KSIZE = 3

# --- Ficheros del modelo ---------------------------------------------------
PCA_MODEL_PATH = MODELS_DIR / "pca_igo.npz"       # media + componentes
THRESHOLD_PATH = MODELS_DIR / "threshold.json"    # λ calibrado

# --- Base de datos -----------------------------------------------------
DEFAULT_DB_PATH = ROOT / "db" / "templates.csv"
DEFAULT_KEY_PATH = ROOT / "secrets" / "db.key"
KEY_ENV_VAR = "FACEREC_DB_KEY"        # clave en base64 (32 bytes)
KEY_FILE_ENV_VAR = "FACEREC_DB_KEY_FILE"
DB_PATH_ENV_VAR = "FACEREC_DB_PATH"
ADMIN_TOKEN_ENV_VAR = "FACEREC_ADMIN_TOKEN"   # sin él, `borrar` está deshabilitado

# Varias caras: si la 2ª cara tiene al menos esta fracción del área de la mayor,
# la foto es ambigua y se rechaza (en vez de elegir una en silencio).
MULTI_FACE_RATIO = 0.5

# Defensa "lobo": una plantilla que supera λ contra más de max(WOLF_MIN_IDS,
# WOLF_FRACTION*N) ids a la vez no es una cara, es una plantilla fabricada.
# (Medido: una plantilla legítima acepta como mucho 5 ids; la media de la BD, 19.)
WOLF_MIN_IDS = 8
WOLF_FRACTION = 0.15

# Límite de intentos fallidos (ver facerec/ratelimit.py)
RATE_LIMIT_ENV_VAR = "FACEREC_RATE_LIMIT"      # 0 = desactivado
RATE_LIMIT_DEFAULT = 30                        # rechazos por ventana y clave
RATE_WINDOW_S = 60.0

# Bloqueo de fichero de la BD (ver facerec/filelock.py)
LOCK_TIMEOUT_S = 15.0
LOCK_STALE_S = 60.0

# Límites de validación de entradas
MAX_ID_LEN = 64
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_SIDE = 8000
MIN_SD_SIDE = 32           # una imagen _sd más pequeña se rechaza
