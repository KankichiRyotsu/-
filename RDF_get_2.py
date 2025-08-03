import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF
import matplotlib.pyplot as plt
import numpy as np

u = mda.Universe("dump.lammpstrj", format="LAMMPSDUMP", atom_style="id type x y z mass")

Si = u.select_atoms("type 2")
O = u.select_atoms("type 1")

# RDF 計算
rdf_SiO = InterRDF(Si, O, nbins=200, range=(0.0, 6.0))
rdf_OO = InterRDF(O, O, nbins=200, range=(0.0, 6.0), exclude="self")
rdf_SiSi = InterRDF(Si, Si, nbins=200, range=(0.0, 6.0), exclude="self")

rdf_SiO.run()
rdf_OO.run()
rdf_SiSi.run()

bins = rdf_SiO.results.bins
g_SiO = rdf_SiO.results.rdf
g_OO = rdf_OO.results.rdf
g_SiSi = rdf_SiSi.results.rdf

# total g(r): 濃度重み付き合成
N_Si = len(Si)
N_O = len(O)
N_total = N_Si + N_O
c_SiSi = (N_Si / N_total) ** 2
c_OO = (N_O / N_total) ** 2
c_SiO = (N_Si / N_total) * (N_O / N_total)
g_total = c_SiSi * g_SiSi + 2 * c_SiO * g_SiO + c_OO * g_OO

# r=0 近傍のアーチファクトを避ける（任意）
mask = bins > 0.1

plt.figure()
plt.plot(bins[mask], g_SiO[mask], label="Si-O")
plt.plot(bins[mask], g_OO[mask], label="O-O")
plt.plot(bins[mask], g_SiSi[mask], label="Si-Si")
plt.xlabel("r (Å)")
plt.ylabel("g(r)")
plt.legend()
plt.tight_layout()
plt.savefig("rdf_all_with_total.png")

# 既存の combined plot を保存済みなら、Totalのみ別出力
plt.figure()
plt.plot(bins[mask], g_total[mask], label="Total", color="k")
plt.xlabel("r (Å)")
plt.ylabel("g(r)")
plt.legend()
plt.tight_layout()
plt.savefig("rdf_total_only.png")

