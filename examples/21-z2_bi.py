#! /usr/bin/env python
"""
In this tutorial we show how to evaluate the Z2 topological invariant using the
Z2 class. We will plot the evolution of WF centers with respect to k_b and count
the number of crossing. The effects of SOC can be investigated by varying the
intensity of SOC coupling.

References:
[1] https://journals.aps.org/prb/abstract/10.1103/PhysRevB.84.075119
[2] https://journals.aps.org/prb/abstract/10.1103/PhysRevB.52.1566
[3] https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.97.236805
"""

from math import sqrt, pi

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

import tbplas as tb
import tbplas.builder.exceptions as exc


def calc_hop(sk, rij, label_i, label_j):
    # Range-dependent slater-Koster parameters from ref. 2
    dict1 = {"v_sss": -0.608, "v_sps": 1.320, "v_pps": 1.854, "v_ppp": -0.600}
    dict2 = {"v_sss": -0.384, "v_sps": 0.433, "v_pps": 1.396, "v_ppp": -0.344}
    dict3 = {"v_sss": 0.0, "v_sps": 0.0, "v_pps": 0.156, "v_ppp": 0.0}
    r_norm = norm(rij)
    if abs(r_norm - 0.30628728) < 1.0e-5:
        data = dict1
    elif abs(r_norm - 0.35116131) < 1.0e-5:
        data = dict2
    else:
        data = dict3

    lm_i = label_i.split(":")[1]
    lm_j = label_j.split(":")[1]

    return sk.eval(r=rij, label_i=lm_i, label_j=lm_j,
                   v_sss=data["v_sss"], v_sps=data["v_sps"],
                   v_pps=data["v_pps"], v_ppp=data["v_ppp"])


def make_cell():
    # Lattice constants from ref. 2
    a = 4.5332
    c = 11.7967
    mu = 0.2341

    # Lattice vectors of bulk from ref. 2
    a1 = np.array([-0.5*a, -sqrt(3)/6*a, c/3])
    a2 = np.array([0.5*a, -sqrt(3)/6*a, c/3])
    a3 = np.array([0, sqrt(3)/3*a, c/3])

    # Lattice vectors and atomic positions of bilayer from ref. 2 & 3
    a1_2d = a2 - a1
    a2_2d = a3 - a1
    a3_2d = np.array([0, 0, c])
    lat_vec = np.array([a1_2d, a2_2d, a3_2d])
    atom_position = np.array([[0, 0, 0], [1/3, 1/3, 2*mu-1/3]])

    # Create cell and add orbitals with energies from ref. 2
    cell = tb.PrimitiveCell(lat_vec, unit=tb.ANG)
    atom_label = ("Bi1", "Bi2")
    e_s, e_p = -10.906, -0.486
    orbital_energy = {"s": e_s, "px": e_p, "py": e_p, "pz": e_p}
    for i, pos in enumerate(atom_position):
        for orbital, energy in orbital_energy.items():
            label = f"{atom_label[i]}:{orbital}"
            cell.add_orbital(pos, label=label, energy=energy)

    # Add hopping terms
    neighbors = tb.find_neighbors(cell, a_max=5, b_max=5, max_distance=0.454)
    sk = tb.SK()
    for term in neighbors:
        i, j = term.pair
        label_i = cell.get_orbital(i).label
        label_j = cell.get_orbital(j).label
        hop = calc_hop(sk, term.rij, label_i, label_j)
        cell.add_hopping(term.rn, i, j, hop)
    return cell


def add_soc(cell):
    # Double the orbitals and hopping terms
    cell = tb.merge_prim_cell(cell, cell)

    # Add spin notations to the orbitals
    num_orb_half = cell.num_orb // 2
    num_orb_total = cell.num_orb
    for i in range(num_orb_half):
        label = cell.get_orbital(i).label
        cell.set_orbital(i, label=f"{label}:up")
    for i in range(num_orb_half, num_orb_total):
        label = cell.get_orbital(i).label
        cell.set_orbital(i, label=f"{label}:down")

    # Add SOC terms
    soc_lambda = 1.5  # ref. 2
    soc = tb.SOC()
    for i in range(num_orb_total):
        label_i = cell.get_orbital(i).label.split(":")
        atom_i, lm_i, spin_i = label_i

        for j in range(i+1, num_orb_total):
            label_j = cell.get_orbital(j).label.split(":")
            atom_j, lm_j, spin_j = label_j

            if atom_j == atom_i:
                soc_intensity = soc.eval(label_i=lm_i, spin_i=spin_i,
                                         label_j=lm_j, spin_j=spin_j)
                soc_intensity *= soc_lambda
                if abs(soc_intensity) >= 1.0e-15:
                    try:
                        energy = cell.get_hopping((0, 0, 0), i, j)
                    except exc.PCHopNotFoundError:
                        energy = 0.0
                    energy += soc_intensity
                    cell.add_hopping((0, 0, 0), i, j, soc_intensity)
    return cell


def main():
    # Create cell and add soc
    cell = make_cell()
    cell = add_soc(cell)

    # Evaluate Z2
    ka_array = np.linspace(-0.5, 0.5, 200)
    kb_array = np.linspace(0.0, 0.5, 200)
    kc = 0.0
    z2 = tb.Z2(cell, num_occ=10, enable_mpi=False)
    timer = tb.Timer()
    timer.tic("phase")
    kb_array, phases = z2.calc_phases(ka_array, kb_array, kc)
    timer.toc("phase")
    timer.report_total_time()

    # Count crossing number
    reorder_phases = False
    if reorder_phases:
        phases = z2.reorder_phases(phases)
        num_crossing = z2.count_crossing(phases, phase_ref=0.2)
        if z2.is_master:
            print(f"Number of crossing: {num_crossing}")

    # Regular plot
    if z2.is_master:
        fig, ax = plt.subplots()
        for i in range(z2.num_occ):
            if reorder_phases:
                ax.plot(kb_array, phases[:, i] / pi, c="r")
            else:
                ax.scatter(kb_array, phases[:, i] / pi, s=1, c="r")
        ax.grid()
        plt.show()
        plt.close()

    # Polar plot
    if z2.is_master:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        for i in range(z2.num_occ):
            if reorder_phases:
                ax.plot(phases[:, i], kb_array, c="r")
            else:
                ax.scatter(phases[:, i], kb_array, s=1, c="r")
        plt.show()
        plt.close()

    # Visualizer
    vis = tb.Visualizer()
    scatter = not reorder_phases
    vis.plot_phases(kb_array, phases, scatter=scatter)
    vis.plot_phases(kb_array, phases, scatter=scatter, polar=True)


if __name__ == '__main__':
    main()
