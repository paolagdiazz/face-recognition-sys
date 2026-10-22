"""Interfaz de línea de comandos.

    python -m facerec init                       # crea clave + inscribe data/
    python -m facerec inscribir   <id> <foto>
    python -m facerec verificar   <id> <foto>
    python -m facerec identificar <foto>
    python -m facerec inscribir_sd <id> <foto>   (y verificar_sd, identificar_sd)
    python -m facerec inscribir_t  <id> <template>  (y verificar_t, identificar_t)
    python -m facerec extraer [--sd] <foto>      # imprime el template (150 valores)
    python -m facerec borrar <id>
    python -m facerec listar

<template> puede ser un fichero (.npy, .csv/.txt con valores separados por comas)
o directamente los valores: "0.12,-1.3,...".

Códigos de salida: 0 = aceptado / OK, 1 = rechazado / desconocido, 2 = error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import config as C


def _read_template(arg: str):
    p = Path(arg)
    if p.is_file():
        if p.suffix == ".npy":
            return np.load(p, allow_pickle=False)
        return p.read_text()
    return arg


def _out(obj, code=0):
    print(json.dumps(obj, ensure_ascii=False))
    return code


def cmd_init(args):
    from .db import generate_key_file, load_key
    from .system import SistemaBiometrico

    key_path = Path(args.key)
    if key_path.exists():
        print(f"clave existente: {key_path}")
    else:
        generate_key_file(key_path)
        print(f"clave nueva generada en {key_path} (no la subas al repo)")
    db_path = Path(args.db)
    if db_path.exists() and not args.force:
        print(f"la BD {db_path} ya existe. Usa --force para recrearla.")
        return 2
    if db_path.exists():
        db_path.unlink()
    s = SistemaBiometrico(db_path=db_path, key=load_key(key_path))
    fotos = sorted(p for p in Path(args.data).iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    ok = 0
    for p in fotos:
        try:
            s.inscribir(p.stem, p)
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {p.name}: {e}")
    print(f"inscritas {ok}/{len(fotos)} personas en {db_path}")
    return 0 if ok == len(fotos) else 2


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m facerec", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=None, help="ruta del CSV (por defecto db/templates.csv)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init")
    p.add_argument("--data", default=str(C.ROOT / "data"))
    p.add_argument("--key", default=str(C.DEFAULT_KEY_PATH))
    p.add_argument("--force", action="store_true")

    for name in ("inscribir", "verificar", "inscribir_sd", "verificar_sd"):
        q = sub.add_parser(name)
        q.add_argument("id")
        q.add_argument("foto")
    for name in ("identificar", "identificar_sd"):
        sub.add_parser(name).add_argument("foto")
    for name in ("inscribir_t", "verificar_t"):
        q = sub.add_parser(name)
        q.add_argument("id")
        q.add_argument("template")
    sub.add_parser("identificar_t").add_argument("template")
    q = sub.add_parser("extraer")
    q.add_argument("foto")
    q.add_argument("--sd", action="store_true")
    q.add_argument("--out", help="guardar el template en este fichero .npy")
    sub.add_parser("borrar").add_argument("id")
    sub.add_parser("listar")

    args = ap.parse_args(argv)

    from .system import SistemaBiometrico

    try:
        if args.cmd == "init":
            if args.db is None:
                args.db = str(C.DEFAULT_DB_PATH)
            return cmd_init(args)
        s = SistemaBiometrico(db_path=args.db)
        c = args.cmd
        if c in ("inscribir", "inscribir_sd"):
            getattr(s, c)(args.id, args.foto)
            return _out({"ok": True, "id": args.id})
        if c == "inscribir_t":
            s.inscribir_t(args.id, _read_template(args.template))
            return _out({"ok": True, "id": args.id})
        if c in ("verificar", "verificar_sd"):
            r = getattr(s, c)(args.id, args.foto)
            return _out({"verificado": r}, 0 if r else 1)
        if c == "verificar_t":
            r = s.verificar_t(args.id, _read_template(args.template))
            return _out({"verificado": r}, 0 if r else 1)
        if c in ("identificar", "identificar_sd"):
            r = getattr(s, c)(args.foto)
            return _out({"id": r}, 0 if r else 1)
        if c == "identificar_t":
            r = s.identificar_t(_read_template(args.template))
            return _out({"id": r}, 0 if r else 1)
        if c == "extraer":
            t = s.extraer_sd(args.foto) if args.sd else s.extraer(args.foto)
            if args.out:
                np.save(args.out, t)
            print(",".join(f"{v:.6f}" for v in t))
            return 0
        if c == "borrar":
            s.borrar(args.id)
            return _out({"ok": True})
        if c == "listar":
            return _out({"inscritos": s.inscritos()})
    except Exception as e:  # noqa: BLE001
        return _out({"error": f"{type(e).__name__}: {e}"}, 2)
    return 2


if __name__ == "__main__":
    sys.exit(main())
