"""Bloqueo de fichero sencillo y multiplataforma (sin dependencias).

Se crea un fichero `.lock` con O_CREAT|O_EXCL (operación atómica en Windows y
Linux). Si un proceso muere sin soltarlo, el bloqueo se considera caducado
pasados `stale` segundos y se rompe.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from . import config as C


class LockTimeout(RuntimeError):
    """No se pudo obtener el bloqueo de la BD a tiempo."""


class FileLock:
    def __init__(self, path: str | Path, timeout: float = C.LOCK_TIMEOUT_S,
                 stale: float = C.LOCK_STALE_S):
        self.path = Path(path)
        self.timeout = timeout
        self.stale = stale
        self._fd: int | None = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self._fd, str(os.getpid()).encode())
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > self.stale:
                        self.path.unlink()          # bloqueo huérfano (proceso muerto)
                        continue
                except OSError:
                    pass                            # lo soltaron justo ahora: reintentar
                if time.monotonic() > deadline:
                    raise LockTimeout(f"no se pudo obtener el bloqueo {self.path.name} "
                                      f"en {self.timeout:.0f}s")
                time.sleep(0.02)
            except PermissionError:
                # Windows: el fichero se está borrando; reintentar
                if time.monotonic() > deadline:
                    raise LockTimeout(f"no se pudo obtener el bloqueo {self.path.name}")
                time.sleep(0.02)

    def __exit__(self, *exc):
        if self._fd is not None:
            try:
                os.close(self._fd)
            finally:
                self._fd = None
                try:
                    self.path.unlink()
                except OSError:
                    pass
        return False
