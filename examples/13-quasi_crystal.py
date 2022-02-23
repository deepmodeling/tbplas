#! /usr/bin/env python

import math

import numpy as np

import tbplas as tb


def rotate(center, angle):
    def _rotate(orb_pos):
        pos_shift = orb_pos - center
        pos_shift = tb.rotate_coord(pos_shift, angle=angle/180*math.pi)
        pos_shift = pos_shift + center
        orb_pos[:, :] = pos_shift
    return _rotate


def make_quasi_crystal(prim_cell, dim, center_pc):
    center_sc = np.array([dim[0]//2, dim[1]//2, 0]) + center_pc
    center_sc = np.matmul(center_sc, prim_cell.lat_vec)
    sc_fixed = tb.SuperCell(prim_cell, dim)
    sc_rotated = tb.SuperCell(prim_cell, dim,
                              orb_pos_modifier=rotate(center_sc, 30.0))
    sample = tb.Sample(sc_fixed, sc_rotated)
    return sample


def main():
    
    prim_cell = tb.make_graphene_diamond()
    dim = (33, 33, 1)
    center = (2./3, 2./3, 0)
    sample = make_quasi_crystal(prim_cell, dim, center)
    sample.plot( with_cells=False, with_orbitals=False, hop_as_arrows=False)

    prim_cell = tb.make_graphene_rect()
    dim = (36, 24, 1)
    center = (0, 1./3, 0)
    sample = make_quasi_crystal(prim_cell, dim, center)
    sample.plot(with_cells=False, with_orbitals=False, hop_as_arrows=False)


if __name__ == "__main__":
    main()
