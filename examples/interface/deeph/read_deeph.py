#! /usr/bin/env python
"""
This example demonstrates how to read the Hamiltonian and overlap matrices
produced by DeepH and calculate the band structure of carbon nanotube with
100 atoms. The data files are not shipped with tbplas. They can be generated
following the instructions at
https://github.com/deepmodeling/DeepH-pack/files/9526304/demo_abacus.zip

CAUTION: The Hamiltonian matrix elements produced by DeepH do not strictly
follow the conjugate relation <i,0|H|j,R> = <j,0|H|i,-R>*. However, this
relation is enthusiastically utilized by TBPLaS to reduce memory and CPU cost.
This consistency may introduce errors in the Hamiltonian matrix in Bloch basis
as well as the eigenvalues and eigen-vectors. Unfortunately, the errors are
strongly case-dependent and not predictable. So, test carefully before
production calculations.
"""
from typing import Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt

import tbplas as tb


def read_deep_h(data_dir: str = ".",
                hop_eng_cutoff: float = 1.0e-5,
                overlap_cutoff: float = 1.0e-5) -> Tuple[tb.PrimitiveCell, tb.Overlap]:
    """
    Read primitive cell and overlap from output of DeepH.

    :param data_dir: directory name containing the data files
    :param hop_eng_cutoff: cutoff for hopping terms in eV
    :param overlap_cutoff: cutoff for overlap terms
    :return: (prim_cell, overlap)
    """
    # Read lattice vectors in Angstrom and create the primitive cell
    lat_vec = np.loadtxt(f"{data_dir}/lat.dat").T
    prim_cell = tb.PrimitiveCell(lat_vec, unit=tb.ANG)

    # Read atom positions in Angstrom and convert to fractional coordinates
    atom_pos = np.loadtxt(f"{data_dir}/site_positions.dat").T
    atom_pos = tb.cart2frac(lat_vec, atom_pos)

    # Get the number of orbitals on each atom and starting orbital indices
    num_orb = []
    with open(f"{data_dir}/orbital_types.dat", "r") as infile:
        content = infile.readlines()
        for line in content:
            s = 0
            for v in line.split():
                s += (2 * int(v) + 1)
            num_orb.append(s)
    ind_start = [sum(num_orb[:_]) for _ in range(len(num_orb))]

    # Add orbitals to the primitive cell
    for i, pos in enumerate(atom_pos):
        for j in range(num_orb[i]):
            prim_cell.add_orbital(tuple(pos))

    # Read Hamiltonian
    fid = h5py.File(f"{data_dir}/hamiltonians_pred.h5", "r")
    for key, value in fid.items():
        # Parse rn and atom indices
        key = key.lstrip("[").rstrip("]").split(",")
        rn = (int(key[0]), int(key[1]), int(key[2]))
        atom_i = int(key[3]) - 1
        atom_j = int(key[4]) - 1

        # Loop over orbitals on atoms to add hopping terms
        for i in range(num_orb[atom_i]):
            orb_i = ind_start[atom_i] + i
            for j in range(num_orb[atom_j]):
                orb_j = ind_start[atom_j] + j
                energy = value[i, j]
                if abs(energy) > hop_eng_cutoff:
                    try:
                        prim_cell.add_hopping(rn, orb_i, orb_j, energy)
                    except tb.PCHopDiagonalError:
                        prim_cell.set_orbital(orb_i, energy=energy.real)
    fid.close()

    # Read overlap matrix
    overlap = tb.Overlap(prim_cell)
    fid = h5py.File(f"{data_dir}/overlaps.h5", "r")
    for key, value in fid.items():
        # Parse rn and atom indices
        key = key.lstrip("[").rstrip("]").split(",")
        rn = (int(key[0]), int(key[1]), int(key[2]))
        atom_i = int(key[3]) - 1
        atom_j = int(key[4]) - 1

        # Loop over orbitals on atoms to add hopping terms
        for i in range(num_orb[atom_i]):
            orb_i = ind_start[atom_i] + i
            for j in range(num_orb[atom_j]):
                orb_j = ind_start[atom_j] + j
                olap = value[i, j]
                if abs(olap) > overlap_cutoff:
                    try:
                        overlap.add_offsite(rn, orb_i, orb_j, olap)
                    except tb.PCHopDiagonalError:
                        overlap.set_onsite(orb_i, overlap=olap.real)
    fid.close()
    return prim_cell, overlap


def main():
    prim_cell, overlap = read_deep_h()

    # Evaluate band structure
    fermi_energy = -2.4
    k_points = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5],
    ])
    k_path, k_idx = tb.gen_kpath(k_points, [40])
    solver = tb.DiagSolver(prim_cell, overlap, enable_mpi=True)
    k_len, bands = solver.calc_bands(k_path, solver="lapack",
                                     convention=2)[:2]
    # k_len, bands = solver.calc_bands(k_path, solver="arpack", convention=2,
    #                                  k=600, largest=False)[:2]
    bands -= fermi_energy

    # Plot on master process
    if solver.is_master:
        num_bands = bands.shape[1]
        for i in range(num_bands):
            plt.plot(k_len, bands[:, i], color="red", linewidth=1.0)
        plt.xlim(k_len.min(), k_len.max())
        plt.ylim(-4, 4)
        plt.show()


if __name__ == "__main__":
    main()
