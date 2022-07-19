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


def make_quasi_crystal_pc(prim_cell, dim, center):
    # Build bottom and top layers
    bottom_layer = tb.extend_prim_cell(prim_cell, dim=dim)
    top_layer = tb.extend_prim_cell(prim_cell, dim=dim)

    # Get the Cartesian coordinates of rotation center
    center = np.array([dim[0]//2, dim[1]//2, 0]) + center
    center = np.matmul(center, prim_cell.lat_vec)

    # Get rotated lattice vectors of top layer
    angle = 30 / 180 * math.pi
    end_points = np.vstack((np.zeros(3), top_layer.lat_vec)) - center
    end_points = tb.rotate_coord(end_points, angle)
    lat_vec = end_points[1:] - end_points[0]

    # Get rotated orbital positions of top layer
    orb_pos = top_layer.orb_pos_nm - center
    orb_pos = tb.rotate_coord(orb_pos, angle) + center
    orb_pos = tb.cart2frac(lat_vec, orb_pos)

    # Rotate top layer
    top_layer.lat_vec = lat_vec
    for i in range(top_layer.num_orb):
        top_layer.set_orbital(i, orb_pos[i])

    # Reshape top layer
    conv_mat = np.matmul(bottom_layer.lat_vec, np.linalg.inv(top_layer.lat_vec))
    top_layer = tb.reshape_prim_cell(top_layer, conv_mat)

    # Merge bottom and top layers
    final_cell = tb.merge_prim_cell(top_layer, bottom_layer)

    # Remove unnecessary orbitals
    idx_remove = []
    orb_pos = final_cell.orb_pos_nm
    for i, pos in enumerate(orb_pos):
        if np.linalg.norm(pos - center) > 3.0:
            idx_remove.append(i)
    final_cell.remove_orbitals(idx_remove)
    return final_cell


def main():
    prim_cell = tb.make_graphene_diamond()
    dim = (33, 33, 1)
    center = (2./3, 2./3, 0)
    cell = make_quasi_crystal_pc(prim_cell, dim, center)
    cell.plot(with_cells=False, with_orbitals=False, hop_as_arrows=False)
    sample = make_quasi_crystal(prim_cell, dim, center)
    sample.plot(with_cells=False, with_orbitals=False, hop_as_arrows=False)

    prim_cell = tb.make_graphene_rect()
    dim = (36, 24, 1)
    center = (0, 1./3, 0)
    sample = make_quasi_crystal(prim_cell, dim, center)
    sample.plot(with_cells=False, with_orbitals=False, hop_as_arrows=False)


if __name__ == "__main__":
    main()
