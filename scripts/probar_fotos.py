"""Prueba el sistema con VUESTRAS fotos reales (Mamdani, foto impresa, pantalla...).

    python scripts/probar_fotos.py tests/mamdani
    python scripts/probar_fotos.py tests/presentacion --real
    python scripts/probar_fotos.py --id zohran-mamdani foto1.jpg foto2.jpg

Qué id se espera para cada foto:
  1. el de --id, si se indica;
  2. si no, lo que hay en el nombre antes de "__":  taylor_swift_2009__pantalla.jpg -> taylor_swift_2009
  3. si no, la carpeta/nombre que coincida con un inscrito (p. ej. zohran-mamdani.jpg).

Para cada foto da: ¿se detecta cara?, score contra su id, posición en el ranking,
resultado de verificar() y de identificar(). Al final, un resumen por persona.
Por defecto usa una BD temporal con las fotos de data/ (no toca db/ ni secrets/);
con --real usa vuestra BD de verdad (hay que haber hecho `python -m facerec init`).

Ideas de pruebas (guardad las fotos en tests/, que no se sube a data/):
  tests/mamdani/            5-10 fotos de la persona nº 51: luz, gafas, edad, expresión.
  tests/presentacion/       fotos hechas a una foto impresa o a la pantalla de un móvil,
                            nombradas  <id>__impresa1.jpg, <id>__pantalla1.jpg ...
  tests/varias_caras/       fotos con 2+ personas (deben dar error de cara ambigua).
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from facerec import config as C  # noqa: E402

os.environ.setdefault(C.RATE_LIMIT_ENV_VAR, "0")
from facerec import FaceError, SistemaBiometrico  # noqa: E402
from scripts.pruebas_extra import montar  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}


def expandir(items):
    for it in items:
        p = Path(it)
        if p.is_dir():
            yield from sorted(f for f in p.rglob("*") if f.suffix.lower() in EXT)
        elif p.is_file():
            yield p


def esperado(p: Path, forzado, inscritos):
    if forzado:
        return forzado
    if "__" in p.stem:
        return p.stem.split("__")[0]
    for cand in (p.stem, p.parent.name):
        if cand in inscritos:
            return cand
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fotos", nargs="+", help="fotos o carpetas")
    ap.add_argument("--id", help="id esperado para todas las fotos")
    ap.add_argument("--real", action="store_true", help="usar la BD real (db/) en vez de una temporal")
    a = ap.parse_args()

    tmp = tempfile.TemporaryDirectory()
    s = SistemaBiometrico() if a.real else montar(Path(tmp.name))
    ids = s.inscritos()
    print(f"BD: {'real' if a.real else 'temporal (data/)'} | {len(ids)} inscritos | "
          f"λ verificar = {s.threshold:.4f} | λ identificar = {s.threshold_ident:.4f}\n")
    print(f"{'foto':34s} {'esperado':22s} {'score':>6s} {'rank':>4s} {'verificar':>9s}  identificar")
    res = defaultdict(list)
    for p in expandir(a.fotos):
        pid = esperado(p, a.id, ids)
        if pid not in ids:
            print(f"{p.name[:34]:34s} {str(pid)[:22]:22s}   (id sin inscribir o sin determinar: usa --id)")
            continue
        try:
            t = s.extraer(p)
        except FaceError as e:
            print(f"{p.name[:34]:34s} {pid[:22]:22s}   sin resultado: {e}")
            res[pid].append(None)
            continue
        rk = s.ranking(t)
        score = dict(rk)[pid]
        rank = [n for n, _ in rk].index(pid) + 1
        ver = s.verificar_t(pid, t)
        ide = s.identificar_t(t)
        res[pid].append((score, rank, ver, ide))
        print(f"{p.name[:34]:34s} {pid[:22]:22s} {score:6.3f} {rank:4d} {'ACEPTA' if ver else 'rechaza':>9s}  "
              f"{ide if ide else 'desconocido'}{'  <-- OTRA PERSONA' if ide and ide != pid else ''}")

    print("\n== RESUMEN POR PERSONA ==")
    for pid, rows in res.items():
        ok = [r for r in rows if r]
        if not ok:
            print(f"{pid}: ninguna foto procesada ({len(rows)} errores)")
            continue
        n = len(ok)
        acept = sum(r[2] for r in ok)
        r1 = sum(r[1] == 1 for r in ok)
        mal = sum(1 for r in ok if r[3] and r[3] != pid)
        print(f"{pid}: {n} fotos | verificar acepta {acept}/{n} | rank-1 {r1}/{n} | score medio "
              f"{np.mean([r[0] for r in ok]):.3f} | identificadas como OTRA persona: {mal} | sin cara/ambiguas: {len(rows) - n}")
    tmp.cleanup()


if __name__ == "__main__":
    main()
