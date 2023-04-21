#! /usr/bin/env python

import math

import numpy as np

import tbplas as tb


# In this example we show how to create a rectangular graphene primitive cell.
# There are two approaches. In the first approach we build it from scratch.

# Generate lattice vectors
sqrt3 = math.sqrt(3)
a = 2.46
cc_bond = sqrt3 / 3 * a
vectors = tb.gen_lattice_vectors(sqrt3 * cc_bond, 3 * cc_bond)

# Create cell and add orbitals
rect_cell = tb.PrimitiveCell(vectors)
rect_cell.add_orbital((0, 0))
rect_cell.add_orbital((0, 2 / 3.))
rect_cell.add_orbital((1 / 2., 1 / 6.))
rect_cell.add_orbital((1 / 2., 1 / 2.))

# Add hopping terms
rect_cell.add_hopping((0, 0), 0, 2, -2.7)
rect_cell.add_hopping((0, 0), 2, 3, -2.7)
rect_cell.add_hopping((0, 0), 3, 1, -2.7)
rect_cell.add_hopping((0, 1), 1, 0, -2.7)
rect_cell.add_hopping((1, 0), 3, 1, -2.7)
rect_cell.add_hopping((1, 0), 2, 0, -2.7)

# Again we can view the cell by calling 'plot'.
rect_cell.plot()

# In the second approach, we create the rectangular cell by reshaping diamond
# cell with a1' = a1 and a2' = -a1 + 2a2. For convenience, we will import the
# diamond-shaped cell from the materials repository with 'make_graphene_diamond'.
diamond_cell = tb.make_graphene_diamond()
lat_sc = np.array([[1, 0, 0], [-1, 2, 0], [0, 0, 1]])
rect_cell2 = tb.reshape_prim_cell(diamond_cell, lat_sc)

# We will see both approaches yield the same cell.
rect_cell2.plot()
