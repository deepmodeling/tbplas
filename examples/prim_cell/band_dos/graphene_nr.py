#! /usr/bin/env python

import numpy as np

import tbplas as tb


# In this tutorial we show how to calculate the band structure of graphene
# nano-ribbon with armchair and zigzag edges. Firstly, we import the rectangular
# primitive cell.
rect_cell = tb.make_graphene_rect()

# Make AM and ZZ nano-ribbons
gnr_am = tb.extend_prim_cell(rect_cell, dim=(3, 3, 1))
gnr_am.apply_pbc(pbc=(False, True, False))
gnr_am.plot()

gnr_zz = tb.extend_prim_cell(rect_cell, dim=(3, 3, 1))
gnr_zz.apply_pbc(pbc=(True, False, False))
gnr_zz.plot()

# Now we evaluate their band structures. It is well known in the literature that
# armchair nano-ribbon usually has finite band gap, while zigzag nano-ribbon is
# always metallic.

# Armchair nano-ribbon
k_points = np.array([
    [0.0, -0.5, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0],
])
k_label = ["X", "G", "X"]
k_path, k_idx = tb.gen_kpath(k_points, [40, 40])
k_len, bands = gnr_am.calc_bands(k_path)
vis = tb.Visualizer()
vis.plot_bands(k_len, bands, k_idx, k_label)

# Zigzag nano-ribbon
k_points = np.array([
    [-0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0],
])
k_label = ["X", "G", "X"]
k_path, k_idx = tb.gen_kpath(k_points, [40, 40])
k_len, bands = gnr_zz.calc_bands(k_path)
vis.plot_bands(k_len, bands, k_idx, k_label)
