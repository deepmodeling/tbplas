#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import tbplas as tb


# In this tutorial we show how to calculate the band structure and DOS of
# primitive cell. First, we import graphene primitive cell in diamond shape
# from the material repository.
dia_cell = tb.make_graphene_diamond()

# Then we create a path in the 1st Brillouin zone.
k_points = np.array([
    [0.0, 0.0, 0.0],
    [1. / 2, 0.0, 0.0],
    [2. / 3, 1. / 3, 0.0],
    [0.0, 0.0, 0.0],
])
k_label = ["G", "M", "K", "G"]
k_path, k_idx = tb.gen_kpath(k_points, [40, 40, 40])

# Then we evaluate the band structure.
k_len, bands = dia_cell.calc_bands(k_path)

# The band structure can be plotted as following:
num_bands = bands.shape[1]
for i in range(num_bands):
    plt.plot(k_len, bands[:, i], color="r", linewidth=1.0)
for idx in k_idx:
    plt.axvline(k_len[idx], color='k', linewidth=1.0)
plt.xlim((0, np.amax(k_len)))
plt.xticks(k_len[k_idx], k_label)
plt.xlabel("k / (1/nm)")
plt.ylabel("Energy (eV)")
plt.tight_layout()
plt.show()

# Alternatively, we can use the 'Visualizer' class.
vis = tb.Visualizer()
vis.plot_bands(k_len, bands, k_idx, k_label)

# Now we calculate DOS. We need to create a mesh-grid in 1st Brillouin zone
# first.
k_mesh = tb.gen_kmesh((120, 120, 1))
energies, dos = dia_cell.calc_dos(k_mesh)
vis.plot_dos(energies, dos)
