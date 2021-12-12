#! /usr/bin/env python

import numpy as np

import tbplas as tb


# In this tutorial we will show how to build complex samples with deformation.
# Firstly, we define the deformation: a Gaussian bump by shifting z-coordinate
# of orbitals.
def _make_pos_mod(c0):
    def _pos_mod(orb_pos):
        x, y = orb_pos[:, 0], orb_pos[:, 1]
        orb_pos[:, 2] += np.exp(-(x - c0[0]) ** 2 - (y - c0[1]) ** 2)
    return _pos_mod


# Then we create a 24*24*1 supercell with the orb_pos_modifier argument.
prim_cell = tb.make_graphene_diamond()
super_cell = tb.SuperCell(prim_cell, dim=(24, 24, 1), pbc=(False, False, False),
                          orb_pos_modifier=_make_pos_mod([4.17, 2.70]))
sample = tb.Sample(super_cell)

# Then we have a look at the sample. We need to switch to front view since
# the bump is at z-axis.
sample.plot(with_cells=False, hop_as_arrows=False, view="bc")
