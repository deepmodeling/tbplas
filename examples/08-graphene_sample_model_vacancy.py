#! /usr/bin/env python

import numpy as np

import tbplas as tb


# In this tutorial we will show how to build complex samples with vacancies
# and deformation. First we build a 3*3*1 graphene sample with two orbitals
# removed.
prim_cell = tb.make_graphene_diamond()
vacancies = [(1, 1, 0, 0), (1, 1, 0, 1)]
super_cell = tb.SuperCell(prim_cell, dim=(3, 3, 1), pbc=(False, False, False),
                          vacancies=vacancies)
sample = tb.Sample(super_cell)
sample.plot()


# Then we build a larger 24*24*1 sample with 4 holes located at
# (2.101, 1.361, 0.0), (3.101, 3.361, 0.0), (5.84, 3.51, 0.0) and
# (4.82, 1.11, 0.0) with radius of 0.5 (all units are NM).
# We begin by creating a sample without holes, and get the coordinates of
# all orbitals.
super_cell = tb.SuperCell(prim_cell, dim=(24, 24, 1), pbc=(False, False, False))
sample = tb.Sample(super_cell)
sample.init_orb_pos()
positions = sample.orb_pos


# Then we determine the indices of orbitals to remove.
def add_hole(center, radius=0.5):
    for i, id_pc in enumerate(super_cell.orb_id_pc):
        if np.linalg.norm(positions[i] - center) <= radius:
            holes.append(tuple(id_pc))


holes = []
add_hole([2.101, 1.361, 0.0])
add_hole([3.101, 3.361, 0.0])
add_hole([5.84, 3.51, 0.0])
add_hole([4.82, 1.11, 0.0])

# Then we create a new sample with vacancies.
super_cell = tb.SuperCell(prim_cell, dim=(24, 24, 1), vacancies=holes)
sample = tb.Sample(super_cell)
sample.plot(with_cells=False)

# You may find some dangling orbitals in the sample, i.e. with only one
# associated hopping term. These orbitals and hopping terms can be removed
# by calling 'trim' method of 'SuperCell' class.
super_cell.unlock()
super_cell.trim()
sample = tb.Sample(super_cell)
sample.plot(with_cells=False)