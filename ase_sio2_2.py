from ase.io import read, write
from ase.build import make_supercell
import numpy as np

# 1. Read alpha-quartz from CIF
structure = read("SiO2.cif")  # CIF file path

# 2. Make a supercell (e.g., 3x3x3)
P = np.diag([5, 5, 5])
supercell = make_supercell(structure, P)

# 3. Write LAMMPS data file (atomic style, no charges for Tersoff)
write(
    "quartz_supercell.data",
    supercell,
    format="lammps-data",
    atom_style="atomic",  # Tersoff 用
)

