#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Usage
# python step3_prepare_disp_runs.py --dim 1 1 1 --ddist 0.01 --in-template in.force_template.lmp --in-name in.force_template.lmp 

import argparse
import glob
import os
import re
import shutil
import subprocess
from typing import List

def run_cmd(cmd: str):
    print(f"[RUN] {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")

def ensure_dir(path: str, force: bool = False):
    if os.path.isdir(path):
        if force:
            shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)

def list_poscar_variants() -> List[str]:
    files = sorted(glob.glob("POSCAR-*"))
    if not files:
        raise FileNotFoundError("POSCAR-* が見つかりません。phonopyで変位生成に失敗している可能性があります。")
    return files

def tag_from_poscar(name: str) -> str:
    m = re.fullmatch(r"POSCAR-(\d+)", os.path.basename(name))
    if not m:
        raise ValueError(f"Unexpected file name: {name}")
    return m.group(1)

def main():
    p = argparse.ArgumentParser(
        description="POSCAR から phonopy で変位生成 → disp-*** ディレクトリ作成 → POSCAR と LAMMPS入力を配置（実行はしない）"
    )
    p.add_argument("--dim", type=int, nargs=3, default=[2, 2, 2],
                   help="スーパーセル寸法 (例: --dim 2 2 2)")
    p.add_argument("--ddist", type=float, default=0.03,
                   help="変位距離 [Å] (例: --ddist 0.03)")
    p.add_argument("--phonopy", type=str, default="phonopy",
                   help="phonopy コマンド（パス指定可）")
    p.add_argument("--in-template", type=str, required=True,
                   help="各ディレクトリに配置する LAMMPS 入力テンプレート (例: in_template.lmp)")
    p.add_argument("--in-name", type=str, default="in.lmp",
                   help="テンプレートを配置する先のファイル名 (既定: in.lmp)")
    p.add_argument("--move", action="store_true",
                   help="POSCAR-*** をコピーではなく移動する")
    p.add_argument("--force", action="store_true",
                   help="既存の disp-*** ディレクトリを作り直す")
    p.add_argument("--extras", type=str, nargs="*", default=None,
                   help="各ディレクトリへ一緒にコピーする追加ファイル（ポテンシャル等）")
    p.add_argument("--skip-gen", action="store_true",
                   help="phonopy による変位生成をスキップ（既に POSCAR-* がある場合）")

    args = p.parse_args()

    # 0) 前提チェック
    if not os.path.exists("POSCAR") and not args.skip_gen:
        raise FileNotFoundError("POSCAR がありません。基準構造の POSCAR を用意してください。")

    # 1) phonopy で変位構造を作る（必要な場合）
    if not args.skip_gen:
        dim_str = f"{args.dim[0]} {args.dim[1]} {args.dim[2]}"
        cmd = f'{args.phonopy} -d --dim="{dim_str}" --amplitude {args.ddist} -c POSCAR'
        run_cmd(cmd)
        if not os.path.exists("SPOSCAR"):
            raise FileNotFoundError("SPOSCAR が生成されていません。phonopy 実行を確認してください。")

    # 2) POSCAR-* を列挙
    poscars = list_poscar_variants()
    print(f"[INFO] Found {len(poscars)} displacement structures.")

    # 3) disp-*** ディレクトリを作成し、POSCAR と in.lmp を配置
    for pos in poscars:
        tag = tag_from_poscar(pos)           # "001" など
        out_dir = f"disp-{tag}"
        ensure_dir(out_dir, force=args.force)

        # POSCAR を配置（ファイル名は 'POSCAR'）
        dst_poscar = os.path.join(out_dir, "POSCAR")
        if args.move:
            shutil.move(pos, dst_poscar)
            action = "Moved"
        else:
            shutil.copy2(pos, dst_poscar)
            action = "Copied"
        print(f"[INFO] {action} {pos} -> {dst_poscar}")

        # LAMMPS 入力テンプレートを配置（例: in.lmp）
        in_dst = os.path.join(out_dir, args.in_name)
        shutil.copy2(args.in_template, in_dst)
        print(f"[INFO] Copied {args.in_template} -> {in_dst}")

        # 追加ファイル
        if args.extras:
            for extra in args.extras:
                dst_extra = os.path.join(out_dir, os.path.basename(extra))
                shutil.copy2(extra, dst_extra)
                print(f"[INFO] Copied {extra} -> {dst_extra}")

    print("\n準備完了：disp-*** に POSCAR と入力ファイルを配置しました（計算は未実行）。")

if __name__ == "__main__":
    main()
