#! /usr/bin/env python
"""Example for constructing graphene sample with vacancies."""

from typing import Tuple, List

import numpy as np

import tbplas as tb


def get_vacancies(super_cell: tb.SuperCell,
                  orb_pos: np.ndarray,
                  centers: np.ndarray,
                  radius: float = 0.5) -> List[Tuple[int, int, int, int]]:
    """
    Get indices of vacancies according to given center and radius.

    :param super_cell: supercell to which to add vacancies
    :param orb_pos: Cartesian coordinates of orbitals in nm
    :param centers: Cartesian coordinates of centers in nm
    :param radius: radius of holes in nm
    :return: indices of vacancies in primitive cell representation
    """
    vacancies = []
    for i, id_pc in enumerate(super_cell.orb_id_pc):
        for c0 in centers:
            if np.linalg.norm(orb_pos[i] - c0) <= radius:
                vacancies.append(tuple(id_pc))
    return vacancies


def main():
    # In this tutorial we will show how to build complex samples with vacancies.
    # First we build a 3*3*1 graphene sample with two orbitals removed.
    prim_cell = tb.make_graphene_diamond()
    vacancies = [(1, 1, 0, 0), (1, 1, 0, 1)]
    super_cell = tb.SuperCell(prim_cell, dim=(3, 3, 1),
                              pbc=(False, False, False), vacancies=vacancies)
    sample = tb.Sample(super_cell)
    sample.plot()

    # Then we build a larger 24*24*1 sample with 4 holes located at
    # (2.101, 1.361, 0.0), (3.101, 3.361, 0.0), (5.84, 3.51, 0.0) and
    # (4.82, 1.11, 0.0) with radius of 0.5 (all units are NM).
    # We begin by creating a sample without holes, and get the coordinates of
    # all orbitals. Then we get the indices of orbitals to remove.
    super_cell = tb.SuperCell(prim_cell, dim=(24, 24, 1),
                              pbc=(False, False, False))
    sample = tb.Sample(super_cell)
    sample.init_orb_pos()
    positions = sample.orb_pos
    centers = np.array([[2.101, 1.361, 0.0],
                        [3.101, 3.361, 0.0],
                        [5.84, 3.51, 0.0],
                        [4.82, 1.11, 0.0]])
    vacancies = get_vacancies(super_cell, positions, centers)

    # Then we create a new sample with vacancies.
    super_cell = tb.SuperCell(prim_cell, dim=(24, 24, 1), vacancies=vacancies)
    sample = tb.Sample(super_cell)
    sample.plot(with_cells=False)

    # You may find some dangling orbitals in the sample, i.e. with only one
    # associated hopping term. These orbitals and hopping terms can be removed
    # by calling 'trim' method of 'SuperCell' class.
    super_cell.unlock()
    super_cell.trim()
    sample = tb.Sample(super_cell)
    sample.plot(with_cells=False)


if __name__ == "__main__":
    main()
