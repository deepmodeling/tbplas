#! /usr/bin/env python

from math import cos, sin, exp
import numpy as np
import tbplas as tb


def init_wfc_pw(sample, kpt):
    sample.init_orb_pos()
    orb_pos = sample.orb_pos
    wfc = np.zeros(orb_pos.shape[0], dtype=np.complex128)
    for i, pos in enumerate(orb_pos):
        phi = np.dot(pos, kpt)
        wfc[i] = cos(phi) + 1j * sin(phi)
    wfc /= np.linalg.norm(wfc)
    return wfc


def init_wfc_gaussian(sample, mu, sigma, scale=None):
    if scale is None:
        scale = [1.0, 1.0]
    sample.init_orb_pos()
    orb_pos = sample.orb_pos
    wfc = np.zeros(orb_pos.shape[0], dtype=np.complex128)
    for i, pos in enumerate(orb_pos):
        dx = (pos.item(0) - mu[0]) * scale[0]
        dy = (pos.item(1) - mu[1]) * scale[1]
        wfc[i] = exp(-(dx**2 + dy**2) / (2 * sigma**2))
    wfc /= np.linalg.norm(wfc)
    return wfc


def add_scatter_gaussian(sample, mu, sigma, scale=None):
    if scale is None:
        scale = [1.0, 1.0]
    sample.init_orb_pos()
    sample.init_orb_eng()
    orb_pos = sample.orb_pos
    orb_eng = sample.orb_eng
    for i, pos in enumerate(orb_pos):
        dx = (pos.item(0) - mu[0]) * scale[0]
        dy = (pos.item(1) - mu[1]) * scale[1]
        orb_eng[i] += exp(-(dx**2 + dy**2) / (2 * sigma**2))


def main():
    # Build the sample
    prim_cell = tb.make_graphene_rect()
    sample = tb.Sample(tb.SuperCell(prim_cell, dim=(50, 20, 1),
                                    pbc=(True, True, False)))

    # Common quantities
    r_vectors = sample.sc_list[0].sc_lat_vec
    g_vectors = tb.gen_reciprocal_vectors(r_vectors)
    x_max, y_max = np.max(r_vectors[:, 0]), np.max(r_vectors[:, 1])

    # Initialize the wave function
    wfc = init_wfc_gaussian(sample, mu=[x_max/2, y_max/2], sigma=0.5,
                            scale=[1.0, 0.0])
    # kpt = np.matmul((2, 2, 0), g_vectors)
    # wfc = init_wfc_pw(sample, kpt)

    # Add scatting center
    add_scatter_gaussian(sample, mu=[x_max/4, y_max/2], sigma=0.5)

    # Propagate the wave function
    config = tb.Config()
    config.generic['nr_time_steps'] = 256
    time_log = np.array([0, 16, 32, 48, 64, 80, 96, 128, 255])
    sample.rescale_ham()
    solver = tb.Solver(sample, config)
    psi_t = solver.calc_psi_t(wfc, time_log)

    # Plot the wave function
    vis = tb.Visualizer()
    for i in range(len(time_log)):
        vis.plot_wf2(sample, np.abs(psi_t[i]), cmap="hot", scatter=False)


if __name__ == "__main__":
    main()
