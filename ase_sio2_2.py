#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
disp-* ディレクトリにある POSCAR を LAMMPS data に一括変換
その後、other.data があれば disp.data の末尾に 1 行空けて追記
"""

import argparse
import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


def append_other_data(target_data: str, other_file: str):
    if not os.path.isfile(other_file):
        print(f"[SKIP] other.data not found: {other_file}")
        return

    with open(target_data, "rb+") as fout:
        fout.seek(0, os.SEEK_END)
        pos = fout.tell()
        if pos > 0:
            fout.seek(pos - 1)
            last = fout.read(1)
            if last != b"\n":
                fout.write(b"\n")  # 最終行に改行がなければ付与

        fout.write(b"\n")  # 必ず空行を 1 行入れる

        with open(other_file, "rb") as fin:
            data = fin.read()
            fout.write(data)

        if not data.endswith(b"\n"):
            fout.write(b"\n")  # other.dataが改行無しで終わる場合のケア

    print(f"[APPENDED] other.data -> {target_data}")


def convert_one(d, out_name, atom_style, units, wrap, other):
    from ase.io import read, write
    poscar = os.path.join(d, "POSCAR")
    if not os.path.isfile(poscar):
        return (d, False, "POSCAR not found")

    try:
        # POSCAR → data
        atoms = read(poscar, format="vasp")
        atoms.set_pbc([True, True, True])
        if wrap:
            atoms.wrap()

        out_path = os.path.join(d, out_name)
        write(out_path, atoms, format="lammps-data",
              atom_style=atom_style, units=units)
        msg = f"Wrote {out_path}"

        # other.data があれば追記
        if other and os.path.isfile(other):
            append_other_data(out_path, other)
            msg += f" + other.data appended"

        return (d, True, msg)
    except Exception as e:
        return (d, False, str(e))


def main():
    ap = argparse.ArgumentParser(description="Convert disp-*/POSCAR to data + other.data復元")
    ap.add_argument("--glob", type=str, default="disp-*", help="対象ディレクトリのglob")
    ap.add_argument("--out", type=str, default="disp.data", help="出力dataファイル名")
    ap.add_argument("--atom-style", type=str, default="atomic", help="LAMMPS atom_style")
    ap.add_argument("--units", type=str, default="metal", help="LAMMPS units")
    ap.add_argument("--workers", type=int, default=4, help="並列数")
    ap.add_argument("--no-wrap", action="store_true", help="wrapを行わない")
    ap.add_argument("--other", type=str, default="../other.data", help="追記するother.dataのパス")
    args = ap.parse_args()

    # ASE check
    try:
        import ase  # noqa: F401
    except Exception:
        raise SystemExit("ASE が必要です: pip install ase")

    dirs = sorted([d for d in glob.glob(args.glob) if os.path.isdir(d)])
    if not dirs:
        raise SystemExit(f"対象ディレクトリが見つかりません: {args.glob}")

    print(f"[INFO] Converting {len(dirs)} dirs → '{args.out}' (append other.data: {args.other})")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(convert_one,
                          d, args.out, args.atom_style,
                          args.units, not args.no_wrap, args.other)
                for d in dirs]
        for fu in as_completed(futs):
            results.append(fu.result())

    ok = sum(1 for r in results if r[1])
    ng = len(results) - ok
    for d, success, info in sorted(results):
        status = "OK" if success else "NG"
        print(f"[{status}] {d}: {info}")

    print(f"\n✅ Done. success={ok}, failed={ng}")


if __name__ == "__main__":
    main()
