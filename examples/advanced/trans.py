#! /usr/bin/env python
"""Example for calculating transmission."""

import math
from collections import defaultdict

import numpy as np

import tbplas as tb

from strain_fields import init_wfc_gaussian, init_wfc_pw, add_efield


def integrate(sample: tb.Sample, wfc: np.ndarray) -> np.ndarray:
    """
    Integrate wave function along y-direction.

    :param sample: sample under investigation
    :param wfc: wave function to integrate
    :return: integrated wave function
    """
    sample.init_orb_pos()
    orb_pos = sample.orb_pos
    wfc_x = defaultdict(float)
    for i, pos in enumerate(orb_pos):
        x_i, wfc_i = pos.item(0), wfc.item(i)
        wfc_x[x_i] += abs(wfc_i)**2
    wfc_x = np.array(sorted([(x, y) for x, y in wfc_x.items()]))
    return wfc_x


def main():
    # Fundamental parameters
    prim_cell = tb.make_graphene_rect()
    dim = (100, 20, 1)
    pbc = (False, True, False)

    # Geometries
    x_max = prim_cell.lat_vec[0, 0] * dim[0]
    y_max = prim_cell.lat_vec[1, 1] * dim[1]
    g_x = 2 * math.pi / x_max

    # Perturbation
    deform_center = np.array([x_max * 0.5, y_max * 0.5])
    deform_extent = (1.0, 0.0)
    v_pot = 8.0

    # Initial wave function
    wfc_center = np.array([x_max * 0.1, y_max * 0.5])
    wfc_extent = (1.0, 0.0)
    wfc_kpt = np.array([50*g_x, 0, 0])

    # Propagation
    num_step = 512

    # Make sample
    sample = tb.Sample(tb.SuperCell(prim_cell, dim, pbc))
    add_efield(sample, center=deform_center, extent=deform_extent, v_pot=v_pot)

    # Initialize wave function
    psi_gau = init_wfc_gaussian(sample, center=wfc_center, extent=wfc_extent)
    psi_pw = init_wfc_pw(sample, wfc_kpt)
    psi0 = psi_gau * psi_pw

    # Propagate wave function
    config = tb.Config()
    config.generic['nr_time_steps'] = num_step
    time_log = np.array([_ for _ in range(num_step)])
    sample.rescale_ham()
    solver = tb.Solver(sample, config)
    psi_t = solver.calc_psi_t(psi0, time_log)

    # Integrate the wave function over y
    time, pos, wfc = [], [], []
    for i, t in enumerate(time_log):
        wfc_x = integrate(sample, psi_t[i])
        time.append(t * np.ones(wfc_x.shape[0]))
        pos.append(wfc_x[:, 0])
        wfc.append(wfc_x[:, 1])
    time = np.concatenate(time)
    pos = np.concatenate(pos)
    wfc = np.concatenate(wfc)

    # Visualize wave function
    vis = tb.Visualizer()
    vis.plot_scalar(pos / x_max, time / num_step, wfc)


if __name__ == "__main__":
    main()
