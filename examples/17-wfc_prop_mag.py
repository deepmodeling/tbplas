#! /usr/bin/env python

import math

import numpy as np
from numpy.linalg import norm

import tbplas as tb


def make_deform(center, sigma=0.5, extent=(1.0, 1.0), scale=(0.5, 0.5)):
    def _deform(orb_pos):
        x, y, z = orb_pos[:, 0], orb_pos[:, 1], orb_pos[:, 2]
        dx = (x - center[0]) * extent[0]
        dy = (y - center[1]) * extent[1]
        amp = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
        x += dx * amp * scale[0]
        y += dy * amp * scale[0]
        z += amp * scale[1]
    return _deform


def calc_hop(rij):
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


def update_hop(sample):
    sample.init_hop()
    sample.init_dr()
    for i, rij in enumerate(sample.dr):
        sample.hop_v[i] = calc_hop(rij)


def add_efield(sample, center, sigma=0.5, extent=(1.0, 1.0), v_pot=1.0):
    sample.init_orb_pos()
    sample.init_orb_eng()
    orb_pos = sample.orb_pos
    orb_eng = sample.orb_eng
    for i, pos in enumerate(orb_pos):
        dx = (pos.item(0) - center[0]) * extent[0]
        dy = (pos.item(1) - center[1]) * extent[1]
        orb_eng[i] += v_pot * math.exp(-(dx**2 + dy**2) / (2 * sigma**2))


def init_wfc_gaussian(sample, center, sigma=0.5, extent=(1.0, 1.0)):
    sample.init_orb_pos()
    orb_pos = sample.orb_pos
    wfc = np.zeros(orb_pos.shape[0], dtype=np.complex128)
    for i, pos in enumerate(orb_pos):
        dx = (pos.item(0) - center[0]) * extent[0]
        dy = (pos.item(1) - center[1]) * extent[1]
        wfc[i] = math.exp(-(dx**2 + dy**2) / (2 * sigma**2))
    wfc /= np.linalg.norm(wfc)
    return wfc


def init_wfc_pw(sample, kpt):
    sample.init_orb_pos()
    orb_pos = sample.orb_pos
    wfc = np.zeros(orb_pos.shape[0], dtype=np.complex128)
    for i, pos in enumerate(orb_pos):
        phi = np.dot(pos, kpt).item()
        wfc[i] = math.cos(phi) + 1j * math.sin(phi)
    wfc /= np.linalg.norm(wfc)
    return wfc


def init_wfc_random(sample):
    phase = 2 * np.pi * np.random.rand(sample.num_orb_tot)
    wfc = np.exp(1j * phase)
    wfc /= np.linalg.norm(wfc)
    return wfc


def main():
    # Fundamental parameters
    prim_cell = tb.make_graphene_rect()
    dim = (50, 20, 1)
    pbc = (True, True, False)

    # Geometries
    x_max = prim_cell.lat_vec[0, 0] * dim[0]
    y_max = prim_cell.lat_vec[1, 1] * dim[1]
    g_x = 2 * math.pi / x_max
    g_y = 2 * math.pi / y_max

    # Perturbations
    with_deform = False
    deform_center = (x_max * 0.75, y_max * 0.5)
    with_efield = False
    with_mfield = False
    mfield_gauge = 0
    mfiend_intensity = 50

    # Initial wave function
    init_wfc = "gaussian"
    wfc_center = (x_max * 0.5, y_max * 0.5)
    wfc_extent = (1.0, 0.0)
    wfc_kpt = np.array([g_x, 0, 0])

    # Output
    plot_sample = False
    plot_v_pot = False
    plot_wfc = True
    plot_kind = "abs"
    plot_mean_y = False

    # Make sample
    if with_deform:
        deform = make_deform(center=deform_center)
        sample = tb.Sample(tb.SuperCell(prim_cell, dim, pbc,
                                        orb_pos_modifier=deform))
        update_hop(sample)
    else:
        sample = tb.Sample(tb.SuperCell(prim_cell, dim, pbc))

    # Add electric and magnetic field
    if with_efield:
        add_efield(sample, center=deform_center)
    if with_mfield:
        sample.set_magnetic_field(mfiend_intensity, gauge=mfield_gauge)

    # Initialize wave function
    if init_wfc == "pw":
        psi0 = init_wfc_pw(sample, wfc_kpt)
    elif init_wfc == "gaussian":
        psi0 = init_wfc_gaussian(sample, center=wfc_center, extent=wfc_extent)
    else:
        psi0 = init_wfc_random(sample)

    # Propagate wave function
    config = tb.Config()
    if with_mfield:
        config.generic['nr_time_steps'] = 2200
        time_log = np.array([i*100 for i in range(21)])
    else:
        config.generic['nr_time_steps'] = 128
        time_log = np.array([0, 16, 32, 64, 128])
    sample.rescale_ham()
    solver = tb.Solver(sample, config)
    psi_t = solver.calc_psi_t(psi0, time_log)

    # Plot the model
    if plot_sample:
        sample.plot(with_cells=False, hop_as_arrows=False, with_orbitals=False,
                    fig_name="sample.png", fig_dpi=100)

    # Plot potential
    vis = tb.Visualizer()
    if plot_v_pot:
        vis.plot_wfc(sample, sample.orb_eng, cmap="hot", scatter=True,
                     fig_name="v_pot.png", fig_dpi=100)

    # Plot time-dependent wave function
    if plot_wfc:
        for i in range(len(time_log)):
            if plot_kind == "abs":
                wfc = np.abs(psi_t[i])**2
            elif plot_kind == "real":
                wfc = psi_t[i].real
            elif plot_kind == "imag":
                wfc = psi_t[i].imag
            elif plot_kind == "phase":
                wfc = np.angle(psi_t[i])
            else:
                raise ValueError(f"Illegal {plot_kind}")
            vis.plot_wfc(sample, wfc, cmap="hot", scatter=False,
                         fig_name=f"{time_log[i]}.png", fig_dpi=100)

    # Plot mean y
    if plot_mean_y:
        mean_y = np.zeros(len(time_log), dtype=float)
        orb_pos_y = sample.orb_pos[:, 1]
        for i_t, psi in enumerate(psi_t):
            mean_y[i_t] = np.dot(orb_pos_y, np.abs(psi)**2)
        vis.plot_xy(time_log, mean_y)
        np.save("time_log", time_log)
        np.save(f"mean_y_{dim[1]}", mean_y)


if __name__ == "__main__":
    main()
