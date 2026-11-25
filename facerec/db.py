"""Base de datos de templates en CSV, cifrada y autenticada con AES-256-GCM.

Formato (`db/templates.csv`):

    id,nonce,data
    adele_2016,<nonce b64>,<AES-GCM(template) b64>
    ...
    __manifest__,<nonce b64>,<AES-GCM(sha256 de todas las filas) b64>

- Cada template (150 float32) se cifra con AES-GCM; el `id` va como dato
  asociado (AAD). Así:
    * leer el CSV no revela los templates (confidencialidad);
    * modificar un byte se detecta (integridad);
    * mover el template de un id a otro se detecta (el AAD no cuadra).
- La fila `__manifest__` autentica el conjunto de filas: borrar, añadir o
  duplicar filas sin la clave se detecta.
- Un CSV que no existe o está vacío NO se interpreta como "BD vacía": es un
  error de integridad (fail closed). Solo `create_empty()` (usado por `init`)
  crea una BD nueva. Así, quien solo puede escribir en db/ no puede vaciarla y
  re-inscribirse con el id de otra persona.
- Lo que NO protege (documentado): restaurar una copia ANTIGUA completa del
  CSV (rollback) y que los ids se ven en claro.

Ante cualquier fallo de integridad el sistema se niega a operar (fail closed).
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import re
import secrets
import tempfile
from pathlib import Path

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from . import config as C
from .filelock import FileLock

MANIFEST_ID = "__manifest__"
AAD_PREFIX = b"facerec-pcago-v1|"
# Se usa fullmatch: con match + "$", un "\n" final se colaba (el $ casa antes de él).
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _\-().']{0,%d}" % (C.MAX_ID_LEN - 1))
HEADER = ["id", "nonce", "data"]


class DBError(RuntimeError):
    pass


class DBIntegrityError(DBError):
    """El CSV ha sido manipulado o la clave no es la correcta."""


def validate_id(pid: str) -> str:
    if not isinstance(pid, str) or not ID_RE.fullmatch(pid) or pid == MANIFEST_ID:
        raise ValueError("id no válido: 1-64 caracteres [A-Za-z0-9 _-().'], empezando por letra o número")
    return pid


# --------------------------------------------------------------------------
# Clave
# --------------------------------------------------------------------------
def generate_key_file(path: Path) -> bytes:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = AESGCM.generate_key(bit_length=256)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(base64.b64encode(key).decode())
    return key


def load_key(key_path: Path | None = None) -> bytes:
    """Orden: variable FACEREC_DB_KEY (base64) > FACEREC_DB_KEY_FILE > secrets/db.key."""
    raw = os.environ.get(C.KEY_ENV_VAR)
    if raw is None:
        p = Path(os.environ.get(C.KEY_FILE_ENV_VAR) or key_path or C.DEFAULT_KEY_PATH)
        if not p.is_file():
            raise DBError(f"no hay clave de la BD ({p}). Ejecuta `python -m facerec init`.")
        raw = p.read_text().strip()
    try:
        key = base64.b64decode(raw, validate=True)
    except ValueError as e:
        raise DBError("la clave no es base64 válido") from e
    if len(key) != 32:
        raise DBError("la clave debe tener 32 bytes (AES-256)")
    return key


# --------------------------------------------------------------------------
# BD
# --------------------------------------------------------------------------
class TemplateDB:
    def __init__(self, path: Path, key: bytes, dim: int):
        self.path = Path(path)
        self._aes = AESGCM(key)
        self.dim = dim

    def lock(self) -> FileLock:
        """Bloqueo exclusivo para operaciones leer-modificar-escribir (evita perder inscripciones)."""
        return FileLock(self.path.with_name(self.path.name + ".lock"))

    # -- cifrado de una fila --
    def _enc(self, pid: str, payload: bytes) -> tuple[str, str]:
        nonce = secrets.token_bytes(12)
        ct = self._aes.encrypt(nonce, payload, AAD_PREFIX + pid.encode())
        return base64.b64encode(nonce).decode(), base64.b64encode(ct).decode()

    def _dec(self, pid: str, nonce_b64: str, data_b64: str) -> bytes:
        try:
            nonce = base64.b64decode(nonce_b64, validate=True)
            ct = base64.b64decode(data_b64, validate=True)
            return self._aes.decrypt(nonce, ct, AAD_PREFIX + pid.encode())
        except (InvalidTag, ValueError) as e:
            raise DBIntegrityError(f"fila manipulada o clave incorrecta (id={pid!r})") from e

    @staticmethod
    def _rows_digest(rows: list[list[str]]) -> bytes:
        h = hashlib.sha256()
        for r in sorted(rows):
            h.update(("\x1f".join(r) + "\x1e").encode())
        return h.digest()

    # -- lectura / escritura --
    def load(self) -> dict[str, np.ndarray]:
        """Lee y verifica TODO el CSV. Lanza DBIntegrityError si algo no cuadra."""
        if not self.path.is_file():
            raise DBIntegrityError(f"no existe la BD ({self.path}); créala con `init`")
        with open(self.path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if not rows:
            raise DBIntegrityError("la BD está vacía (0 bytes): borrada o truncada")
        if rows[0] != HEADER:
            raise DBIntegrityError("cabecera del CSV no válida")
        body = rows[1:]
        manifest = [r for r in body if r and r[0] == MANIFEST_ID]
        data = [r for r in body if r and r[0] != MANIFEST_ID]
        if len(manifest) != 1:
            raise DBIntegrityError("falta el manifiesto o está duplicado")
        if any(len(r) != 3 for r in body):
            raise DBIntegrityError("fila con nº de columnas incorrecto")
        expected = self._dec(MANIFEST_ID, manifest[0][1], manifest[0][2])
        if expected != self._rows_digest(data):
            raise DBIntegrityError("el conjunto de filas no coincide con el manifiesto "
                                   "(filas añadidas, borradas o duplicadas)")
        out: dict[str, np.ndarray] = {}
        for pid, n, d in data:
            if pid in out:
                raise DBIntegrityError(f"id duplicado: {pid!r}")
            raw = self._dec(pid, n, d)
            if len(raw) != 4 * self.dim:
                raise DBIntegrityError(f"template de longitud incorrecta (id={pid!r})")
            out[pid] = np.frombuffer(raw, dtype="<f4").copy()
        return out

    def create_empty(self) -> None:
        """Crea una BD nueva y válida (solo cabecera + manifiesto). Falla si ya existe."""
        if self.path.exists():
            raise DBError(f"la BD ya existe: {self.path}")
        self.save({})

    def save(self, templates: dict[str, np.ndarray]) -> None:
        rows = []
        for pid in sorted(templates):
            t = np.asarray(templates[pid], dtype="<f4")
            if t.shape != (self.dim,):
                raise DBError("dimensión de template incorrecta")
            rows.append([pid, *self._enc(pid, t.tobytes())])
        manifest = [MANIFEST_ID, *self._enc(MANIFEST_ID, self._rows_digest(rows))]
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(HEADER)
        w.writerows(rows)
        w.writerow(manifest)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # escritura atómica: o se escribe todo o no se toca el fichero
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".tmp_", suffix=".csv")
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            f.write(buf.getvalue())
        os.replace(tmp, self.path)
