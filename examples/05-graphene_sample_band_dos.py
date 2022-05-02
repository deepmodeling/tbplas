#! /usr/bin/env python

import numpy as np

import tbplas as tb


# In this tutorial we show to calculate the band structure and DOS of the
# sample using exact diagonalizing. We will take periodic graphene sample
# as the model since only periodic structures can have band structure.
prim_cell = tb.make_graphene_diamond()
super_cell_pbc = tb.SuperCell(prim_cell, dim=(6, 6, 1), pbc=(True, True, True))
sample_pbc = tb.Sample(super_cell_pbc)

# Evaluation of band structure is similar to that of primitive cell.
# NOTE: since we are using a sample containing 6*6*1 supercell, the Dirac
# cone at Gamma point will be folded to K point.
k_points = np.array([
    [0.0, 0.0, 0.0],
    [1. / 2, 0.0, 0.0],
    [2. / 3, 1. / 3, 0.0],
    [0.0, 0.0, 0.0],
])
k_label = ["G", "M", "K", "G"]
k_path, k_idx = tb.gen_kpath(k_points, [40, 40, 40])
k_len, bands = sample_pbc.calc_bands(k_path)
vis = tb.Visualizer()
vis.plot_bands(k_len, bands, k_idx, k_label)

# Evaluation of DOS
# NOTE: since we are already using a 6*6*1 sample, we need 20*20*1 k_mesh
# to attain the same accuracy as that in example01, where 120*120*1 k_mesh
# is employed.
k_mesh = tb.gen_kmesh((20, 20, 1))
energies, dos = sample_pbc.calc_dos(k_mesh)
vis.plot_dos(energies, dos)
