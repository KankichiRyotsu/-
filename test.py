import numpy as np
import matplotlib.pyplot as plt
from ase.io.lammpsrun import read_lammps_dump
from pwtools import pydos
from pwtools.signal import smooth
from scipy.signal.windows import hann

# 1. LAMMPSのdumpを読む（全部読む）
images = read_lammps_dump("dump.lammpstrj", index=slice(None))

# 2. 速度を (nstep, natom, 3) にまとめる
vel = np.array([img.get_velocities() for img in images])  # Å/fs 前提
print("vel shape:", vel.shape)

# 3. 解析パラメータ
dt = 5  # fs  ← dumpを何ステップごとに出したかに合わせて変える
natom = vel.shape[1]
mass = np.ones(natom) * 28.0855  # 全部Siならこれ

# 4. direct方式でVDOS
freq, vdos = pydos.pdos(
    vel,
    dt=dt,
    m=mass,
    method="direct",
    npad=15,
)

# 5. 平滑化（Hann窓）
kernel_size = 401
k = hann(kernel_size)
vdos_smooth = smooth(vdos, k)

# 6. 周波数軸を meV に変換
# 1 THz = 4.135667696 meV
freq_thz = freq * 1000.0  # もともと freq は fs^-1 単位 → THz
freq_mev = freq_thz * 4.135667696

# 7. プロット（meVに換算）
plt.plot(freq_mev, vdos, label="raw", alpha=0.4)
plt.plot(freq_mev, vdos_smooth, label="smoothed", linewidth=2)
plt.xlabel("Energy (meV)")
plt.ylabel("VDOS (arb. units)")
plt.title("VDOS from LAMMPS dump (direct, pwtools)")
plt.xlim(0, 90)  # Siなら 0–80 meV くらい
plt.legend()
plt.tight_layout()
plt.show()
