#! /usr/bin/env python

import math

import numpy as np
from numpy.linalg import norm
from scipy.spatial import KDTree

import tbplas as tb


def calc_hop(rij: np.ndarray):
    """
    Calculate hopping parameter according to Slater-Koster relation.
    See ref. [2] for the formulae.

    :param rij: (3,) array
        displacement vector between two orbitals in NM
    :return: hop: float
        hopping parameter
    """
    a0 = 0.1418
    a1 = 0.3349
    r_c = 0.6140
    l_c = 0.0265
    gamma0 = 2.7
    gamma1 = 0.48
    decay = 22.18
    q_pi = decay * a0
    q_sigma = decay * a1
    dr = norm(rij).item()
    n = rij.item(2) / dr
    v_pp_pi = - gamma0 * math.exp(q_pi * (1 - dr / a0))
    v_pp_sigma = gamma1 * math.exp(q_sigma * (1 - dr / a1))
    fc = 1 / (1 + math.exp((dr - r_c) / l_c))
    hop = (n**2 * v_pp_sigma + (1 - n**2) * v_pp_pi) * fc
    return hop


def extend_hop(prim_cell: tb.PrimitiveCell, max_distance=0.75):
    """
    Extend the hopping terms in primitive cell up to cutoff distance.

    :param prim_cell: tb.PrimitiveCell
        primitive cell to extend
    :param max_distance: cutoff distance in NM
    :return: None. Incoming primitive cell is modified
    """
    prim_cell.sync_array()
    pos_r0 = prim_cell.orb_pos_nm
    tree_r0 = KDTree(pos_r0)
    neighbors = [(0, 0, 0)]
    for rn in neighbors:
        pos_rn = pos_r0 + np.matmul(rn, prim_cell.lat_vec)
        tree_rn = KDTree(pos_rn)
        dist_matrix = tree_r0.sparse_distance_matrix(tree_rn,
                                                     max_distance=max_distance)
        for index, distance in dist_matrix.items():
            if distance > 0.0:
                rij = pos_rn[index[1]] - pos_r0[index[0]]
                prim_cell.add_hopping(rn, index[0], index[1], calc_hop(rij))


def make_quasi_crystal_pc(prim_cell, dim, angle, center, radius=3.0, shift=0.3):
    # Build bottom and top layers
    bottom_layer = tb.extend_prim_cell(prim_cell, dim=dim)
    top_layer = tb.extend_prim_cell(prim_cell, dim=dim)

    # Get the Cartesian coordinates of rotation center
    center = np.array([dim[0]//2, dim[1]//2, 0]) + center
    center = np.matmul(center, prim_cell.lat_vec)

    # Rotate and reshape top layer
    tb.spiral_prim_cell(top_layer, angle=angle, center=center, shift=shift)
    conv_mat = np.matmul(bottom_layer.lat_vec, np.linalg.inv(top_layer.lat_vec))
    top_layer = tb.reshape_prim_cell(top_layer, conv_mat)

    # Merge bottom and top layers
    final_cell = tb.merge_prim_cell(top_layer, bottom_layer)

    # Remove unnecessary orbitals
    idx_remove = []
    orb_pos = final_cell.orb_pos_nm
    for i, pos in enumerate(orb_pos):
        if np.linalg.norm(pos[:2] - center[:2]) > radius:
            idx_remove.append(i)
    final_cell.remove_orbitals(idx_remove)

    # Extend hopping terms
    extend_hop(final_cell)
    final_cell.sync_array()
    return final_cell


def cutoff_sc(super_cell, center, radius=3.0):
    idx_remove = []
    orb_pos = super_cell.get_orb_pos()
    for i, pos in enumerate(orb_pos):
        if norm(pos[:2] - center[:2]) > radius:
            idx_remove.append(i)
    idx_remove = np.array(idx_remove, dtype=np.int64)
    super_cell.set_vacancies(super_cell.orb_id_sc2pc_array(idx_remove))


def extend_hop_sc(super_cell, max_distance=0.75):
    pos_r0 = super_cell.get_orb_pos()
    tree_r0 = KDTree(pos_r0)
    neighbors = [(0, 0, 0)]
    for rn in neighbors:
        pos_rn = pos_r0 + np.matmul(rn, super_cell.sc_lat_vec)
        tree_rn = KDTree(pos_rn)
        dist_matrix = tree_r0.sparse_distance_matrix(tree_rn,
                                                     max_distance=max_distance)
        for index, distance in dist_matrix.items():
            if distance > 0.0:
                rij = pos_rn[index[1]] - pos_r0[index[0]]
                super_cell.add_hopping(rn, index[0], index[1], calc_hop(rij))


def make_inter_hop(sc0, sc1, max_distance=0.75):
    inter_hop = tb.SCInterHopping(sc0, sc1)
    pos_r0 = sc0.get_orb_pos()
    pos_r1 = sc1.get_orb_pos()
    tree_r0 = KDTree(pos_r0)
    neighbors = [(0, 0, 0)]
    for rn in neighbors:
        pos_rn = pos_r1 + np.matmul(rn, sc1.sc_lat_vec)
        tree_rn = KDTree(pos_rn)
        dist_matrix = tree_r0.sparse_distance_matrix(tree_rn,
                                                     max_distance=max_distance)
        for index, distance in dist_matrix.items():
            if distance > 0.0:
                rij = pos_rn[index[1]] - pos_r0[index[0]]
                inter_hop.add_hopping(rn, index[0], index[1], calc_hop(rij))
    return inter_hop


def make_quasi_crystal_sample(pc_fixed, pc_twisted, dim, angle, center,
                              radius=3.0, shift=0.3):
    # Calculate the Cartesian coordinates of rotation center
    center = np.array([dim[0]//2, dim[1]//2, 0]) + center
    center = np.matmul(center, pc_fixed.lat_vec)

    # Rotate and shift top layer
    tb.spiral_prim_cell(pc_twisted, angle=angle, center=center, shift=shift)

    # Make the sample
    sc_fixed = tb.SuperCell(pc_fixed, dim)
    sc_twisted = tb.SuperCell(pc_twisted, dim)

    # Remove unnecessary orbitals
    cutoff_sc(sc_fixed, center, radius=radius)
    cutoff_sc(sc_twisted, center, radius=radius)

    # Build inter-cell hopping terms
    inter_hop = make_inter_hop(sc_fixed, sc_twisted, max_distance=0.75)

    # Extend hopping terms
    extend_hop_sc(sc_fixed, max_distance=0.75)
    extend_hop_sc(sc_twisted, max_distance=0.75)
    sample = tb.Sample(sc_fixed, sc_twisted, inter_hop)
    return sample


def main():
    timer = tb.Timer()
    prim_cell = tb.make_graphene_diamond()
    dim = (33, 33, 1)
    angle = 30 / 180 * math.pi
    center = (2./3, 2./3, 0)
    timer.tic("pc")
    cell = make_quasi_crystal_pc(prim_cell, dim, angle, center)
    timer.toc("pc")
    cell.plot(with_cells=False, with_orbitals=False, hop_as_arrows=False,
              hop_eng_cutoff=0.5)

    pc_fixed = tb.make_graphene_diamond()
    pc_twisted = tb.make_graphene_diamond()
    timer.tic("sample_diamond")
    sample = make_quasi_crystal_sample(pc_fixed, pc_twisted, dim, angle, center)
    timer.toc("sample_diamond")
    sample.plot(with_cells=False, with_orbitals=False, hop_as_arrows=False,
                sc_colors=['r', 'b'], hop_colors=['g'], hop_eng_cutoff=0.5)

    pc_fixed = tb.make_graphene_rect()
    pc_twisted = tb.make_graphene_rect()
    dim = (36, 24, 1)
    center = (0, 1./3, 0)
    timer.tic("sample_rect")
    sample = make_quasi_crystal_sample(pc_fixed, pc_twisted, dim, angle, center)
    timer.toc("sample_rect")
    sample.plot(with_cells=False, with_orbitals=False, hop_as_arrows=False,
                sc_colors=['b', 'r'], hop_colors=['g'], hop_eng_cutoff=0.5)

    timer.report_total_time()


if __name__ == "__main__":
    main()
