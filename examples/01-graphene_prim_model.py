#! /usr/bin/env python

import math

import numpy as np

import tbplas as tb


# In this tutorial we show how to create primitive cell. First, we need to
# generate lattice vectors. Lengths are in Angstroms and angles are in degrees.
vectors = tb.gen_lattice_vectors(a=2.46, b=2.46, gamma=60)

# Then we create a primitive cell and add orbitals. Note that coordinates of
# orbitals should be fractional. Since we are lattice vectors with the angle
# between a1 and a2 being 60 degrees, the fractional coordinates are (0, 0, 0)
# and (1/3, 1/3, 0).
cell = tb.PrimitiveCell(vectors)
cell.add_orbital([0.0, 0.0], 0.0)
cell.add_orbital([1. / 3, 1. / 3], 0.0)

# Then we add hopping terms. Conjugate relation <i,0|H|j,R> = <j,0|H|i,-R> are
# handled automatically. So we need to add only half of all the hopping terms.
cell.add_hopping([0, 0], 0, 1, -2.7)
cell.add_hopping([1, 0], 1, 0, -2.7)
cell.add_hopping([0, 1], 1, 0, -2.7)

# Now we can view the primitive cell just created.
cell.plot()

# In the example above we have just created a graphene primitive cell in
# diamond shape. Now we will create a rectangular cell. There are two
# approaches. In the first approach we build it from scratch.

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
rect_cell.add_hopping([0, 0], 0, 2, -2.7)
rect_cell.add_hopping([0, 0], 2, 3, -2.7)
rect_cell.add_hopping([0, 0], 3, 1, -2.7)
rect_cell.add_hopping([0, 1], 1, 0, -2.7)
rect_cell.add_hopping([1, 0], 3, 1, -2.7)
rect_cell.add_hopping([1, 0], 2, 0, -2.7)

# Again we can view the cell by calling 'plot'.
rect_cell.plot()

# In the second approach, we create the rectangular cell by reshaping diamond
# cell with a1' = a1 and a2' = -a1 + 2a2.
lat_sc = np.array([[1, 0, 0], [-1, 2, 0], [0, 0, 1]])
rect_cell2 = tb.reshape_prim_cell(cell, lat_sc)

# We will see both approaches yield the same cell.
rect_cell2.plot()
