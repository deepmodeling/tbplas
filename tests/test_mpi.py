#! /usr/bin/env python

import numpy as np

import tbplas as tb


def make_cell():
    # Make graphene primitive cell
    t = 3.0
    vectors = tb.gen_lattice_vectors(a=0.246, b=0.246, c=1.0, gamma=60)
    cell = tb.PrimitiveCell(vectors, unit=tb.NM)
    cell.add_orbital((0.0, 0.0), label="C_pz")
    cell.add_orbital((1 / 3., 1 / 3.), label="C_pz")
    cell.add_hopping((0, 0), 0, 1, t)
    cell.add_hopping((1, 0), 1, 0, t)
    cell.add_hopping((0, 1), 1, 0, t)
    return cell


def test_band(plot=True):
    cell = make_cell()
    cell = tb.extend_prim_cell(cell, dim=(12, 12, 1))
    solver = tb.DiagSolver(cell, enable_mpi=True)
    vis = tb.Visualizer(enable_mpi=True)
    timer = tb.Timer()

    k_points = np.array([
        [0.0, 0.0, 0.0],
        [2./3, 1./3, 0.0],
        [1./2, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        ])
    k_path, k_idx = tb.gen_kpath(k_points, [40, 40, 40])
    timer.tic("band")
    k_len, bands = solver.calc_bands(k_path)[:2]
    timer.toc("band")
    timer.report_total_time()
    if plot:
        vis.plot_bands(k_len, bands, k_idx, ["G", "K", "M", "G"])


def test_dos(plot=True):
    cell = make_cell()
    cell = tb.extend_prim_cell(cell, dim=(12, 12, 1))
    solver = tb.DiagSolver(cell, enable_mpi=True)
    vis = tb.Visualizer(enable_mpi=True)
    timer = tb.Timer()

    k_points = tb.gen_kmesh((20, 20, 1))
    timer.tic("dos")
    energies, dos = solver.calc_dos(k_points)
    timer.toc("dos")
    timer.report_total_time()
    if plot:
        vis.plot_dos(energies, dos)


def test_lind(plot=True):
    cell = make_cell()
    vis = tb.Visualizer(enable_mpi=True)
    timer = tb.Timer()

    # Test polarization
    lind = tb.Lindhard(cell=cell, energy_max=10, energy_step=1000,
                       kmesh_size=(600, 600, 1), mu=0.0, temperature=300, g_s=2,
                       back_epsilon=1.0, dimension=2, enable_mpi=True)
    q_grid = np.array([[20, 20, 0]])
    q_cart = lind.grid2cart(q_grid, unit=tb.NM)
    timer.tic("dp_reg")
    omegas, dp_reg = lind.calc_dyn_pol_regular(q_grid, use_fortran=True)
    timer.toc("dp_reg")
    timer.tic("dp_arb")
    omegas, dp_arb = lind.calc_dyn_pol_arbitrary(q_cart, use_fortran=True)
    timer.toc("dp_arb")
    if plot:
        vis.plot_xy(omegas, dp_reg[0].imag, color="red")
        vis.plot_xy(omegas, dp_arb[0].imag, color="blue")

    # Test AC cond
    t = cell.get_hopping((0, 0), 0, 1).real
    lind = tb.Lindhard(cell=cell, energy_max=t*3.5, energy_step=2048,
                       kmesh_size=(600, 600, 1), mu=0.0, temperature=300.0,
                       g_s=2, back_epsilon=1.0, dimension=2, enable_mpi=True)
    timer.tic("ac_cond")
    omegas, ac_cond = lind.calc_ac_cond(component="xx", use_fortran=True)
    timer.toc("ac_cond")
    timer.report_total_time()
    omegas /= 4
    ac_cond *= 4
    if plot:
        vis.plot_xy(omegas, ac_cond.real)


def test_z2(plot=True):
    prim_cell = tb.make_graphene_soc(is_qsh=True)
    z2 = tb.Z2(prim_cell, num_occ=2, enable_mpi=True)
    vis = tb.Visualizer(enable_mpi=True)
    timer = tb.Timer()

    ka_array = np.linspace(-0.5, 0.5, 200)
    kb_array = np.linspace(0.0, 0.5, 200)
    kc = 0.0
    timer.tic("z2")
    kb_array, phases = z2.calc_phases(ka_array, kb_array, kc)
    timer.toc("z2")
    timer.report_total_time()
    if plot:
        vis.plot_phases(kb_array, phases)
        vis.plot_phases(kb_array, phases, polar=True)


def test_spin_texture(plot=True):
    cell = tb.make_graphene_soc(is_qsh=True)
    vis = tb.Visualizer(enable_mpi=True)
    timer = tb.Timer()

    # Evaluate expectation of sigma_z.
    k_grid = 2 * tb.gen_kmesh((240, 240, 1)) - 1
    spin_texture = tb.SpinTexture(cell, k_grid, spin_major=False,
                                  enable_mpi=True)
    k_cart = spin_texture.k_cart
    timer.tic("s_z")
    sz = spin_texture.eval("z")
    timer.toc("s_z")
    timer.report_total_time()
    if plot:
        vis.plot_scalar(x=k_cart[:, 0], y=k_cart[:, 1], z=sz[:, 2],
                        num_grid=(480, 480), cmap="jet")

    # Evaluate spin_texture
    k_grid = 2 * tb.gen_kmesh((48, 48, 1)) - 1
    spin_texture.k_grid = k_grid
    k_cart = spin_texture.k_cart
    sx = spin_texture.eval("x")
    sy = spin_texture.eval("y")
    if plot:
        vis.plot_vector(x=k_cart[:, 0], y=k_cart[:, 1], u=sx[:, 2], v=sy[:, 2])


def main():
    plot = True
    test_band(plot)
    test_dos(plot)
    test_z2(plot)
    test_lind(plot)
    test_spin_texture(plot)


if __name__ == "__main__":
    main()
