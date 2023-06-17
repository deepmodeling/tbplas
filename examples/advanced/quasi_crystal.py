#! /usr/bin/env python
"""
Example for constructing quasi-crystal at primitive cell and sample levels.
"""

import math
from typing import Union, Tuple

import numpy as np
from numpy.linalg import norm

import tbplas as tb


def calc_hop(rij: np.ndarray) -> float:
    """
    Calculate hopping parameter according to Slater-Koster relation.

    :param rij: (3,) array, displacement vector between two orbitals in NM
    :return: hopping parameter in eV
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


def extend_hop(cell: Union[tb.PrimitiveCell, tb.SuperCell],
               max_distance=0.75) -> None:
    """
    Extend the hopping terms in primitive or supercell up to cutoff distance.

    :param cell: primitive cell or supercell to extend
    :param max_distance: cutoff distance in NM
    :return: None. Incoming cell is modified.
    """
    neighbors = tb.find_neighbors(cell, a_max=0, b_max=0,
                                  max_distance=max_distance)
    for term in neighbors:
        i, j = term.pair
        cell.add_hopping(term.rn, i, j, calc_hop(term.rij))


def cutoff_pc(prim_cell: tb.PrimitiveCell, center: np.ndarray,
              radius: float = 3.0) -> None:
    """
    Cutoff primitive cell up to given radius with respect to given center.

    :param prim_cell: supercell to cut
    :param center: Cartesian coordinate of center in nm
    :param radius: cutoff radius in nm
    :return: None. The incoming supercell is modified.
    """
    idx_remove = []
    orb_pos = prim_cell.orb_pos_nm
    for i, pos in enumerate(orb_pos):
        if norm(pos[:2] - center[:2]) > radius:
            idx_remove.append(i)
    prim_cell.remove_orbitals(idx_remove)
    prim_cell.trim()


def cutoff_sc(super_cell: tb.SuperCell, center: np.ndarray,
              radius: float = 3.0) -> None:
    """
    Cutoff supercells up to given radius with respect to given center.

    :param super_cell: supercell to cut
    :param center: Cartesian coordinate of center in nm
    :param radius: cutoff radius in nm
    :return: None. The incoming supercell is modified.
    """
    idx_remove = []
    orb_pos = super_cell.get_orb_pos()
    for i, pos in enumerate(orb_pos):
        if norm(pos[:2] - center[:2]) > radius:
            idx_remove.append(i)
    idx_remove = np.array(idx_remove, dtype=np.int64)
    super_cell.set_vacancies(super_cell.orb_id_sc2pc_array(idx_remove))
    super_cell.trim()


def make_quasi_crystal_pc(prim_cell: tb.PrimitiveCell,
                          dim: Tuple[int, int, int],
                          angle: float, center: np.ndarray,
                          radius: float = 3.0,
                          shift: float = 0.3,
                          algo: int = 0) -> tb.PrimitiveCell:
    """
    Create quasi-crystal at primitive level.

    :param prim_cell: primitive cell from which the quasi-crystal is built
    :param dim: dimension of the extended primitive cell
    :param angle: twisting angle in RADIAN
    :param center: fractional coordinate of the twisting center in the primitive
        cell
    :param radius: radius of quasi-crystal in nm
    :param shift: distance of shift along z-axis in NANOMETER
    :param algo: algorithm to build the quasi-crystal
    :return: the quasi-crystal
    """
    # Get the Cartesian coordinate of rotation center
    center = np.array([dim[0]//2, dim[1]//2, 0]) + center
    center = np.matmul(center, prim_cell.lat_vec)

    # Build fixed and twisted layers
    layer_fixed = tb.extend_prim_cell(prim_cell, dim=dim)
    layer_twisted = tb.extend_prim_cell(prim_cell, dim=dim)

    # We have 2 approaches to build the quasi-crystal
    if algo == 0:
        # Rotate and shift twisted layer
        tb.spiral_prim_cell(layer_twisted, angle=angle, center=center,
                            shift=shift)

        # Remove unnecessary orbitals
        cutoff_pc(layer_fixed, center=center, radius=radius)
        cutoff_pc(layer_twisted, center=center, radius=radius)

        # Reset the lattice of twisted layer
        layer_twisted.reset_lattice(layer_fixed.lat_vec, layer_fixed.origin,
                                    unit=tb.NM, fix_orb=True)
    else:
        # Remove unnecessary orbitals
        cutoff_pc(layer_fixed, center=center, radius=radius)
        cutoff_pc(layer_twisted, center=center, radius=radius)

        # Rotate and shift twisted layer
        orb_pos = tb.rotate_coord(layer_twisted.orb_pos_nm, angle,
                                  center=center)
        orb_pos += np.array([0, 0, shift])
        orb_pos = tb.cart2frac(layer_twisted.lat_vec, orb_pos)
        for i, pos in enumerate(orb_pos):
            layer_twisted.set_orbital(i, position=pos)

    # Build inter-cell hopping terms
    inter_hop = tb.PCInterHopping(layer_fixed, layer_twisted)
    neighbors = tb.find_neighbors(layer_fixed, layer_twisted,
                                  a_max=0, b_max=0, max_distance=0.75)
    for term in neighbors:
        i, j = term.pair
        inter_hop.add_hopping(term.rn, i, j, calc_hop(term.rij))

    # Extend hopping terms
    extend_hop(layer_fixed)
    extend_hop(layer_twisted)
    final_cell = tb.merge_prim_cell(layer_fixed, layer_twisted, inter_hop)
    return final_cell


def make_quasi_crystal_sample(pc_fixed: tb.PrimitiveCell,
                              pc_twisted: tb.PrimitiveCell,
                              dim: Tuple[int, int, int],
                              angle: float, center: np.ndarray,
                              radius: float = 3.0,
                              shift: float = 0.3,
                              algo: int = 0) -> tb.Sample:
    """
    Create quasi-crystal at primitive level.

    :param pc_fixed: primitive cell of fixed layer
    :param pc_twisted: primitive cell of twisted layer
    :param dim: dimension of the extended primitive cell
    :param angle: twisting angle in RADIAN
    :param center: fractional coordinate of the twisting center in the primitive
        cell
    :param radius: radius of quasi-crystal in nm
    :param shift: distance of shift along z-axis in NANOMETER
    :param algo: algorithm to build the quasi-crystal
    :return: the quasi-crystal
    """
    # Calculate the Cartesian coordinate of rotation center
    center = np.array([dim[0]//2, dim[1]//2, 0]) + center
    center = np.matmul(center, pc_fixed.lat_vec)

    # We have 2 approaches to build the quasi-crystal
    if algo == 0:
        # Rotate and shift primitive cell of twisted layer
        tb.spiral_prim_cell(pc_twisted, angle=angle, center=center, shift=shift)

        # Make the supercells
        sc_fixed = tb.SuperCell(pc_fixed, dim)
        sc_twisted = tb.SuperCell(pc_twisted, dim)

        # Remove unnecessary orbitals
        cutoff_sc(sc_fixed, center, radius=radius)
        cutoff_sc(sc_twisted, center, radius=radius)
    else:
        # Make the supercells
        sc_fixed = tb.SuperCell(pc_fixed, dim)
        sc_twisted = tb.SuperCell(pc_twisted, dim)

        # Remove unnecessary orbitals
        cutoff_sc(sc_fixed, center, radius=radius)
        cutoff_sc(sc_twisted, center, radius=radius)

        # Rotate and shift twisted layer
        def _modifier(orb_pos):
            orb_pos[:, :] = tb.rotate_coord(orb_pos, angle, center=center)
            orb_pos[:, 2] += shift
        sc_twisted.set_orb_pos_modifier(_modifier)

    # Build inter-cell hopping terms
    inter_hop = tb.SCInterHopping(sc_fixed, sc_twisted)
    neighbors = tb.find_neighbors(sc_fixed, sc_twisted, a_max=0, b_max=0,
                                  max_distance=0.75)
    for term in neighbors:
        i, j = term.pair
        inter_hop.add_hopping(term.rn, i, j, calc_hop(term.rij))

    # Extend hopping terms
    extend_hop(sc_fixed, max_distance=0.75)
    extend_hop(sc_twisted, max_distance=0.75)
    sample = tb.Sample(sc_fixed, sc_twisted, inter_hop)
    return sample


def main():
    timer = tb.Timer()
    prim_cell = tb.make_graphene_diamond()
    dim = (33, 33, 1)
    angle = 30 / 180 * math.pi
    center = np.array((2./3, 2./3, 0))
    timer.tic("pc")
    cell = make_quasi_crystal_pc(prim_cell, dim, angle, center)
    timer.toc("pc")
    cell.plot(with_cells=False, with_orbitals=False, hop_as_arrows=False,
              hop_eng_cutoff=0.3)

    pc_fixed = tb.make_graphene_diamond()
    pc_twisted = tb.make_graphene_diamond()
    timer.tic("sample_diamond")
    sample = make_quasi_crystal_sample(pc_fixed, pc_twisted, dim, angle, center)
    timer.toc("sample_diamond")
    sample.plot(with_cells=False, with_orbitals=False, hop_as_arrows=False,
                sc_hop_colors=["r", "b"], inter_hop_colors=["g"], hop_eng_cutoff=0.3)

    pc_fixed = tb.make_graphene_rect()
    pc_twisted = tb.make_graphene_rect()
    dim = (36, 24, 1)
    center = np.array((0, 1./3, 0))
    timer.tic("sample_rect")
    sample = make_quasi_crystal_sample(pc_fixed, pc_twisted, dim, angle, center)
    timer.toc("sample_rect")
    sample.plot(with_cells=False, with_orbitals=False, hop_as_arrows=False,
                sc_hop_colors=["b", "r"], inter_hop_colors=["g"], hop_eng_cutoff=0.3)

    timer.report_total_time()


if __name__ == "__main__":
    main()
