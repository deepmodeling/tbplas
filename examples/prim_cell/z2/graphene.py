#! /usr/bin/env python
"""
Example for reproducing the Z2 invariant of monolayer graphene in the reference:
https://journals.aps.org/prb/abstract/10.1103/PhysRevB.84.075119
"""
from math import pi

import numpy as np
import matplotlib.pyplot as plt

import tbplas as tb


def main():
    # Whether to reorder the phases for improving continuity and smoothness
    # CAUTION: this operation may fail!
    reorder_phases = False

    # Set up primitive cell
    prim_cell = tb.make_graphene_soc(is_qsh=True)

    # Plot model
    prim_cell.plot(hop_as_arrows=False)

    # Evaluate band structure
    k_points = np.array([
        [0.0, 0.0, 0.0],
        [1. / 2, 0.0, 0.0],
        [2. / 3, 1. / 3, 0.0],
        [0.0, 0.0, 0.0],
    ])
    k_label = ["G", "M", "K", "G"]
    k_path, k_idx = tb.gen_kpath(k_points, [40, 40, 40])
    k_len, bands = prim_cell.calc_bands(k_path)
    vis = tb.Visualizer()
    vis.plot_bands(k_len, bands, k_idx, k_label)

    # Get phases
    ka_array = np.linspace(-0.5, 0.5, 200)
    kb_array = np.linspace(0.0, 0.5, 200)
    kc = 0.0
    z2 = tb.Z2(prim_cell, num_occ=2)
    kb_array, phases = z2.calc_phases(ka_array, kb_array, kc)
    if reorder_phases:
        phases = z2.reorder_phases(phases)
        num_crossing = z2.count_crossing(phases, phase_ref=0.2)
        print(f"Number of crossing: {num_crossing}")

    # Regular plot
    fig, ax = plt.subplots()
    for i in range(2):
        if reorder_phases:
            ax.plot(kb_array, phases[:, i] / pi)
        else:
            ax.scatter(kb_array, phases[:, i] / pi, s=1, c="r")
    ax.grid()
    plt.show()
    plt.close()

    # Polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    for i in range(2):
        if reorder_phases:
            ax.plot(phases[:, i], kb_array)
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
