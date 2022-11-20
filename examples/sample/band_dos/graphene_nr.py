#! /usr/bin/env python
"""
Example for calculating band structure and DOS for graphene nano-ribbon at
Sample level.
"""
import numpy as np

import tbplas as tb


# Make samples.
rect_cell = tb.make_graphene_rect()
gnr_am = tb.Sample(tb.SuperCell(rect_cell, dim=(3, 3, 1),
                                pbc=(False, True, False)))
gnr_zz = tb.Sample(tb.SuperCell(rect_cell, dim=(3, 3, 1),
                   pbc=(True, False, False)))

# Now we evaluate their band structures. The results should be the same as
# that of primitive cell.

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
