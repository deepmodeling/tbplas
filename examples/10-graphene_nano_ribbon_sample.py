#! /usr/bin/env python

import numpy as np

import tbplas as tb


# In example03 we have demonstrated how to build graphene nano-ribbons in
# armchair and zigzag configuration at primitive cell level using
# 'extend_prim_cell' and 'apply_pbc'. However, these functions are intended
# to small cells. For large cells, the 'SuperCell' and 'Sample' classes are
# recommended, which will be shown in this tutorial.

# Just as in example03, we need the rectangular cell to build nano-ribbons.
rect_cell = tb.make_graphene_rect()

# Creating armchair graphene nano-ribbon is as easy as
gnr_am = tb.Sample(tb.SuperCell(rect_cell, dim=(3, 3, 1),
                                pbc=(False, True, False)))
gnr_am.plot()

# Similar for zigzag nano-ribbon
gnr_zz = tb.Sample(tb.SuperCell(rect_cell, dim=(3, 3, 1),
                   pbc=(True, False, False)))
gnr_zz.plot()

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
