#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import tbplas as tb
from tbplas.parallel import MPIEnv


def test_band(model, mpi_env):
    k_points = np.array([
        [0.0, 0.0, 0.0],
        [2./3, 1./3, 0.0],
        [1./2, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        ])
    if isinstance(model, tb.PrimitiveCell):
        resolution = [8000, 8000, 8000]
    else:
        resolution = [40, 40, 40]
    k_path, k_idx = tb.gen_kpath(k_points, resolution)

    timer = tb.Timer()
    timer.tic("band")
    k_len, bands = model.calc_bands(k_path, enable_mpi=True)
    timer.toc("band")
    timer.report_total_time()

    if mpi_env.rank == 0:
        num_bands = bands.shape[1]
        for i in range(num_bands):
            plt.plot(k_len, bands[:, i], color="red", linewidth=1.0)
        plt.show()


def test_dos(model, mpi_env):
    if isinstance(model, tb.PrimitiveCell):
        k_points = tb.gen_kmesh((240, 240, 1))
    else:
        k_points = tb.gen_kmesh((20, 20, 1))

    timer = tb.Timer()
    timer.tic("dos")
    energies, dos = model.calc_dos(k_points, enable_mpi=True)
    timer.toc("dos")
    timer.report_total_time()

    if mpi_env.rank == 0:
        plt.plot(energies, dos)
        plt.show()


def test_lind(itest, use_fortran):
    # Make graphene primitive cell
    t = 3.0
    vectors = tb.gen_lattice_vectors(a=0.246, b=0.246, c=1.0, gamma=60)
    cell = tb.PrimitiveCell(vectors, unit=tb.NM)
    cell.add_orbital([0.0, 0.0], label="C_pz")
    cell.add_orbital([1 / 3., 1 / 3.], label="C_pz")
    cell.add_hopping([0, 0], 0, 1, t)
    cell.add_hopping([1, 0], 1, 0, t)
    cell.add_hopping([0, 1], 1, 0, t)

    if itest in (0, 1):
        # Create a Lindhard object
        lind = tb.Lindhard(cell=cell, energy_max=10, energy_step=1000,
                           kmesh_size=(600, 600, 1), mu=0.0, temperature=300, g_s=2,
                           back_epsilon=1.0, dimension=2, enable_mpi=True)

        q_grid = np.array([[20, 20, 0]])
        q_cart = lind.grid2cart(q_grid, unit=tb.NM)

    if itest == 0:
        # Calculate dynamic polarization with calc_dyn_pol_regular
        timer = tb.Timer()
        timer.tic("regular")
        omegas, dp_reg = lind.calc_dyn_pol_regular(q_grid, use_fortran=use_fortran)
        timer.toc("regular")
        timer.report_total_time()
        if lind.rank == 0:
            plt.plot(omegas, dp_reg[0].imag, color="red", label="Regular")
            plt.legend()
            plt.show()
            plt.close()
    elif itest == 1:
        # Calculate dynamic polarization with calc_dyn_pol_arbitrary
        timer = tb.Timer()
        timer.tic("arbitrary")
        omegas, dp_arb = lind.calc_dyn_pol_arbitrary(q_cart, use_fortran=use_fortran)
        timer.toc("arbitrary")
        timer.report_total_time()
        if lind.rank == 0:
            plt.plot(omegas, dp_arb[0].imag, color="blue", label="Arbitrary")
            plt.legend()
            plt.show()
            plt.close()
    else:
        # Calculate ac_cond
        lind = tb.Lindhard(cell=cell, energy_max=t*3.5, energy_step=2048,
                        kmesh_size=(600, 600, 1), mu=0.0, temperature=300.0,
                        g_s=2, back_epsilon=1.0, dimension=2, enable_mpi=True)
        timer = tb.Timer()
        timer.tic("ac_cond")
        omegas, ac_cond = lind.calc_ac_cond(component="xx", use_fortran=use_fortran)
        timer.toc("ac_cond")
        timer.report_total_time()
        omegas /= t
        ac_cond *= 4
        if lind.rank == 0:
            plt.plot(omegas, ac_cond.real, color="red")
            plt.minorticks_on()
            plt.show()
            plt.close()


def main():
    cell = tb.make_graphene_diamond()
    sample = tb.Sample(tb.SuperCell(cell, dim=(12, 12, 1), pbc=(True, True, False)))
    mpi_env = MPIEnv()

    test_band(sample, mpi_env)
    # test_dos(sample, mpi_env)
    # test_lind(0, True)


if __name__ == "__main__":
    main()


