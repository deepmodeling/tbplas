#! /usr/bin/env python
"""
Example for calculating quasi-eigenstates and comparison with exact eigenstates.
"""

import numpy as np
import tbplas as tb


def make_gnr() -> tb.Sample:
    prim_cell = tb.make_graphene_rect()
    super_cell = tb.SuperCell(prim_cell, dim=(18, 12, 1),
                              pbc=(True, False, False))
    sample = tb.Sample(super_cell)
    return sample


def calc_bands(sample) -> None:
    k_points = np.array([
        [-0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
    ])
    k_label = ["X", "G", "X"]
    k_path, k_idx = tb.gen_kpath(k_points, [40, 40])
    k_len, bands = sample.calc_bands(k_path)
    vis = tb.Visualizer()
    vis.plot_bands(k_len, bands, k_idx, k_label)


def wfc_diag(sample: tb.Sample) -> np.ndarray:
    k_points = np.array([
        [-0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
    ])
    solver = tb.DiagSolver(model=sample)
    bands, states = solver.calc_states(k_points)

    i_k = 0
    i_b = sample.num_orb_tot // 2
    wfc = np.abs(states[i_k, i_b])**2
    print(bands[i_k, i_b])

    wfc /= wfc.max()
    return wfc


def wfc_tbpm(sample: tb.Sample) -> np.ndarray:
    sample.rescale_ham(9.0)

    config = tb.Config()
    config.generic['nr_random_samples'] = 16
    config.generic['nr_time_steps'] = 1024
    config.quasi_eigenstates['energies'] = [0.0]

    solver = tb.Solver(sample, config)
    qs = solver.calc_quasi_eigenstates()
    wfc = np.abs(qs[0])**2
    wfc /= wfc.max()
    return wfc


def main():
    sample = make_gnr()
    wfc_eigen = wfc_diag(sample)
    wfc_quasi_eigen = wfc_tbpm(sample)
    vis = tb.Visualizer()
    vis.plot_wfc(sample, wfc_eigen, scatter=True,
                 site_size=wfc_eigen*100, site_color="r")
    vis.plot_wfc(sample, wfc_quasi_eigen, scatter=True,
                 site_size=wfc_quasi_eigen*100, site_color="b")


if __name__ == "__main__":
    main()
