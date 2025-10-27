from ase.io.lammpsdata import read_lammps_data
from ase.io import write

# LAMMPS data → ASE Atoms
# style は atom_style に合わせて: 'atomic' / 'charge' / 'full' など
# 単位系は metal が一般的（Å, eV）
atoms = read_lammps_data(
    "aSi.data2",
    style="atomic",
    units="metal",
    # LAMMPSの "type番号 → 原子番号(Z)" の対応を指定
    # 例: 1=Si(14)
    Z_of_type={1: 14}
)

# PBCを確実に設定＆基本セル内へwrap（好みで）
atoms.set_pbc([True, True, True])
atoms.wrap()

# VASP POSCAR を出力（vasp5形式, 直交/斜方どちらでもOK）
# direct=True なら分率座標出力、False ならCartesian
write("POSCAR", atoms, vasp5=True, direct=True)
print("✅ Wrote POSCAR")
