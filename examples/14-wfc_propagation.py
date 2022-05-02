#! /usr/bin/env python

from math import cos, sin, exp
from typing import Tuple
import numpy as np
import tbplas as tb


def init_wfc_pw(sample: tb.Sample, kpt: np.ndarray):
    """
    Initialize wave function with plane wave of given wave vector.

    Parameters
    ----------
    sample: tb.Sample
        sample for which wave function will be initialized
    kpt: np.ndarray
        Cartesian coordinate of wave vector in 1/nm

    Returns
    -------
    wfc: np.ndarray
        plane wave projected on each orbital of the sample
    """
    sample.init_orb_pos()
    orb_pos = sample.orb_pos
    wfc = np.zeros(orb_pos.shape[0], dtype=np.complex128)
    for i, pos in enumerate(orb_pos):
        phi = np.dot(pos, kpt).item()
        wfc[i] = cos(phi) + 1j * sin(phi)
    wfc /= np.linalg.norm(wfc)
    return wfc


def init_wfc_gaussian(sample: tb.Sample, mu: Tuple[float, float], sigma: float,
                      scale: Tuple[float, float] = None):
    """
    Initialize wave function with Gaussian function.

    Parameters
    ----------
    sample: tb.Sample
        sample for which wave function will be initialized
    mu: (c_x, c_y)
        Cartesian coordinate of the center of the Gaussian function in nm
    sigma: float
        broadening of the Gaussian function in nm
    scale: (s_x, s_y)
        scaling factor controlling the extension of Gaussian function along
        x and y directions, works in together with sigma

    Returns
    -------
    wfc: np.ndarray
        Gaussian wave projected on each orbital of the sample
    """
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


def add_scatter_gaussian(sample: tb.Sample, mu: Tuple[float, float],  sigma: float,
                         scale: Tuple[float, float] = None, v_pot: float = 1.0):
    """
    Add Gaussian-type scattering potential to the on-site energies of the sample.

    Parameters
    ----------
    sample: tb.Sample
        sample for which scattering potential will be added
    mu: (c_x, c_y)
        Cartesian coordinate of the center of the Gaussian function in nm
    sigma: float
        broadening of the Gaussian function in nm
    scale: (s_x, s_y)
        scaling factor controlling the extension of Gaussian function along
        x and y directions, works in together with sigma
    v_pot: float
        maximum of the scattering potential in eV

    Returns
    -------
    None. sample.orb_eng is modified.
    """
    if scale is None:
        scale = [1.0, 1.0]
    sample.init_orb_pos()
    sample.init_orb_eng()
    orb_pos = sample.orb_pos
    orb_eng = sample.orb_eng
    for i, pos in enumerate(orb_pos):
        dx = (pos.item(0) - mu[0]) * scale[0]
        dy = (pos.item(1) - mu[1]) * scale[1]
        orb_eng[i] += v_pot * exp(-(dx**2 + dy**2) / (2 * sigma**2))


def main():
    # Build the sample
    prim_cell = tb.make_graphene_rect()
    sample = tb.Sample(tb.SuperCell(prim_cell, dim=(50, 20, 1),
                                    pbc=(True, True, False)))

    # Common quantities
    r_vectors = sample.sc_list[0].sc_lat_vec
    x_max, y_max = np.max(r_vectors[:, 0]), np.max(r_vectors[:, 1])

    # Initialize the wave function
    psi0 = init_wfc_gaussian(sample, mu=(x_max/2, y_max/2), sigma=0.5,
                             scale=(1.0, 0.0))
    # g_vectors = tb.gen_reciprocal_vectors(r_vectors)
    # kpt = np.matmul((3, 2, 0), g_vectors)
    # psi0 = init_wfc_pw(sample, kpt)

    # Visualize the initial wave function
    vis = tb.Visualizer()
    vis.plot_wfc(sample, np.abs(psi0)**2, cmap="hot", scatter=False)

    # Add scatting center
    add_scatter_gaussian(sample, mu=(x_max/4, y_max/2), sigma=0.5, v_pot=1.0)

    # Propagate the wave function
    config = tb.Config()
    config.generic['nr_time_steps'] = 128
    time_log = np.array([16, 32, 48, 64, 80, 96, 112, 127])
    sample.rescale_ham()
    solver = tb.Solver(sample, config)
    psi_t = solver.calc_psi_t(psi0, time_log)

    # Plot the wave function
    for i in range(len(time_log)):
        vis.plot_wfc(sample, np.abs(psi_t[i])**2, cmap="hot", scatter=False)


if __name__ == "__main__":
    main()
