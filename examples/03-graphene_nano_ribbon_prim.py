#! /usr/bin/env python

import numpy as np

import tbplas as tb

# To build graphene nano-ribbons we need the rectangular cell.
# We import it from the material repository. Alternatively, we
# can reuse the model we created in example01.
rect_cell = tb.make_graphene_rect()

# Then we extend the rectangular cell by calling 'extend_prim_cell'
gnr_am = tb.extend_prim_cell(rect_cell, dim=(3, 3, 1))

# We make an armchair graphene nano-ribbon by removing hopping terms along x
# direction.
gnr_am.apply_pbc(pbc=(False, True, False))
gnr_am.plot()

# Similarly, we can make a zigzag nano-ribbon by removing hopping terms along
# y direction.
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

# Note that 'extend_prim_cell' and 'apply_pbc' are intended for exact
# diagonalizing of primitive cell with small size. For large cells, we
# recommend the 'SuperCell' and 'Sample' classes instead.
