#! /usr/bin/env python

import math

import numpy as np

import tbplas as tb


def make_deform(centers, sigma=1.0, scale_xy=0.5, scale_z=1.0):
    def _deform(orb_pos):
        x, y, z = orb_pos[:, 0], orb_pos[:, 1], orb_pos[:, 2]
        norm_factor = 1.0 / (sigma * math.sqrt(2 * math.pi))
        for c0 in centers:
            dr_x, dr_y = x - c0[0], y - c0[1]
            damp = norm_factor * np.exp(-(dr_x**2 + dr_y**2) / (2 * sigma**2))
            x += dr_x * damp * scale_xy
            y += dr_y * damp * scale_xy
            z += damp * scale_z
    return _deform


def make_rand_centers(num_center, xmax, ymax):
    rand_x = np.random.rand(num_center) * xmax
    rand_y = np.random.rand(num_center) * ymax
    zero_z = np.zeros(num_center)
    centers = np.vstack((rand_x, rand_y, zero_z))
    return centers.T


def main():
    # Shared variables
    prim_cell = tb.make_graphene_rect()
    dim = (60, 36, 1)
    pbc = (False, False, False)

    # Make pristine supercell and sample
    super_cell = tb.SuperCell(prim_cell, dim, pbc)
    sample = tb.Sample(super_cell)
    sample.plot(with_cells=False, hop_as_arrows=False, view="ab",
                with_orbitals=False)
    sample.plot(with_cells=False, hop_as_arrows=False, view="bc",
                with_orbitals=False)

    # Get orbital positions
    sample.init_orb_pos()
    positions = sample.orb_pos
    xmax = positions[:, 0].max()
    ymax = positions[:, 1].max()

    # Sample with gaussian bump along z-direction
    centers = 0.5 * np.array([[xmax, ymax, 0.0]])
    deform = make_deform(centers, sigma=0.5, scale_xy=0.0)
    super_cell = tb.SuperCell(prim_cell, dim, pbc, orb_pos_modifier=deform)
    sample = tb.Sample(super_cell)
    sample.plot(with_cells=False, hop_as_arrows=False, view="bc",
                with_orbitals=False)

    # Sample with deformation in xOy-plane
    centers = make_rand_centers(100, xmax, ymax)
    deform = make_deform(centers, scale_z=0.0)
    super_cell = tb.SuperCell(prim_cell, dim, pbc, orb_pos_modifier=deform)
    sample = tb.Sample(super_cell)
    sample.plot(with_cells=False, hop_as_arrows=False, view="ab",
                with_orbitals=False)


if __name__ == "__main__":
    main()
