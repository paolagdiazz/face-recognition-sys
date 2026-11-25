"""Límite de intentos fallidos (frena la fuerza bruta sobre verificar/identificar).

Cada rechazo se anota con su hora en `<bd>.ratelimit.json`. Si una clave (un id,
o "*identificar") acumula `limit` rechazos dentro de la ventana, las siguientes
llamadas dan `RateLimitError` hasta que caduquen. Es persistente porque la CLI
lanza un proceso nuevo en cada llamada.

    FACEREC_RATE_LIMIT=30   rechazos por ventana (por defecto 30; 0 = desactivado)

Limitaciones (documentadas): quien pueda escribir en db/ puede borrar el fichero;
y un atacante puede bloquear temporalmente a un usuario legítimo provocando
rechazos de su id (denegación de servicio acotada a la ventana).
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from . import config as C
from .filelock import FileLock


class RateLimitError(RuntimeError):
    """Demasiados intentos fallidos recientes."""


def configured_limit() -> int:
    raw = os.environ.get(C.RATE_LIMIT_ENV_VAR)
    if raw is None or raw.strip() == "":
        return C.RATE_LIMIT_DEFAULT
    try:
        return max(0, int(raw))
    except ValueError:
        return C.RATE_LIMIT_DEFAULT


class RateLimiter:
    def __init__(self, path: str | Path, window: float = C.RATE_WINDOW_S, clock=time.time):
        self.path = Path(path)
        self.window = window
        self.clock = clock
        self._lock = FileLock(self.path.with_name(self.path.name + ".lock"))

    # el límite se lee en cada llamada, así se puede cambiar con la variable de entorno
    @property
    def limit(self) -> int:
        return configured_limit()

    def _read(self) -> dict[str, list[float]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return {k: [float(t) for t in v] for k, v in data.items()}
        except (OSError, ValueError, AttributeError, TypeError):
            return {}

    def _write(self, data: dict[str, list[float]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".tmp_rl_", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, self.path)

    def _recent(self, data, key, now):
        return [t for t in data.get(key, []) if now - t < self.window]

    def check(self, key: str) -> None:
        """Lanza RateLimitError si la clave ya agotó sus intentos fallidos."""
        limit = self.limit
        if limit <= 0:
            return
        with self._lock:
            now = self.clock()
            recent = self._recent(self._read(), key, now)
        if len(recent) >= limit:
            wait = int(self.window - (now - min(recent))) + 1
            raise RateLimitError(f"demasiados intentos fallidos; espera ~{wait} s")

    def fail(self, key: str) -> None:
        """Anota un rechazo."""
        if self.limit <= 0:
            return
        with self._lock:
            now = self.clock()
            data = {k: [t for t in v if now - t < self.window] for k, v in self._read().items()}
            data = {k: v for k, v in data.items() if v}
            data.setdefault(key, []).append(now)
            self._write(data)
