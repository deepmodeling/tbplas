#! /usr/bin/env python
"""
Example for calculating quasi-eigenstates and comparison with exact eigenstates.
"""

import numpy as np
import tbplas as tb


def make_sample() -> tb.Sample:
    """
    Make graphene sample with a single vacancy.

    :return: sample with vacancy
    """
    prim_cell = tb.make_graphene_diamond()
    super_cell = tb.SuperCell(prim_cell, dim=(17, 17, 1),
                              pbc=(True, True, False))
    super_cell.add_vacancies([(8, 8, 0, 0)])
    sample = tb.Sample(super_cell)
    return sample


def calc_bands(sample, enable_mpi=False) -> None:
    """
    Calculate the band structure of graphene sample.

    :param sample: graphene sample
    :param enable_mpi: whether to enable MPI parallelization
    :return: None
    """
    k_points = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [2./3, 1./3, 0.0],
        [0.0, 0.0, 0.0],
    ])
    k_label = ["G", "M", "K", "G"]
    k_path, k_idx = tb.gen_kpath(k_points, [40, 40, 40])
    k_len, bands = sample.calc_bands(k_path, enable_mpi=enable_mpi)
    vis = tb.Visualizer(enable_mpi=enable_mpi)
    vis.plot_bands(k_len, bands, k_idx, k_label)


def calc_dos(sample, enable_mpi=False) -> None:
    """
    Calculate the DOS of graphene sample.

    :param sample: graphene sample
    :param enable_mpi: whether to enable MPI parallelization
    :return: None
    """
    k_mesh = tb.gen_kmesh((1, 1, 1))
    energies, dos = sample.calc_dos(k_mesh, enable_mpi=enable_mpi)
    vis = tb.Visualizer(enable_mpi=enable_mpi)
    vis.plot_dos(energies, dos)


def wfc_diag(sample: tb.Sample, enable_mpi=False) -> None:
    """
    Calculate wave function using exact diagonalization.

    :param sample: graphene sample
    :param enable_mpi: whether to enable MPI parallelization
    :return: None
    """
    k_points = np.array([[0.0, 0.0, 0.0]])
    solver = tb.DiagSolver(model=sample, enable_mpi=enable_mpi)
    bands, states = solver.calc_states(k_points)

    i_b = sample.num_orb // 2
    wfc = np.abs(states[0, i_b])**2
    print(bands[0, i_b])

    wfc /= wfc.max()
    vis = tb.Visualizer(enable_mpi=enable_mpi)
    vis.plot_wfc(sample, wfc, scatter=True, site_size=wfc*100, site_color="r",
                 with_model=True, model_style={"alpha": 0.25, "color": "gray"},
                 fig_name="diag.png", fig_size=(6, 4.5), fig_dpi=100)


def wfc_tbpm(sample: tb.Sample, enable_mpi=False) -> None:
    """
    Calculate wave function using TBPM.

    :param sample: graphene sample
    :param enable_mpi: whether to enable MPI parallelization
    :return: None
    """
    sample.rescale_ham()
    config = tb.Config()
    config.generic['nr_random_samples'] = 1
    config.generic['nr_time_steps'] = 1024
    config.quasi_eigenstates['energies'] = [0.0]

    solver = tb.Solver(sample, config, enable_mpi=enable_mpi)
    qs = solver.calc_quasi_eigenstates()
    wfc = np.abs(qs[0])**2
    wfc /= wfc.max()
    vis = tb.Visualizer(enable_mpi=enable_mpi)
    vis.plot_wfc(sample, wfc, scatter=True, site_size=wfc*100, site_color="b",
                 with_model=True, model_style={"alpha": 0.25, "color": "gray"},
                 fig_name="tbpm.png", fig_size=(6, 4.5), fig_dpi=100)


def main():
    sample = make_sample()
    # calc_bands(sample, enable_mpi=True)
    # calc_dos(sample, enable_mpi=True)
    wfc_diag(sample, enable_mpi=True)
    wfc_tbpm(sample, enable_mpi=True)


if __name__ == "__main__":
    main()
