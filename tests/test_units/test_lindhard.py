#! /usr/bin/env python

import numpy as np
import tbplas as tb


def make_cell(a, t):
    lat = a * np.sqrt(3)
    vectors = tb.gen_lattice_vectors(a=lat, b=lat, c=1.0, gamma=60)
    cell = tb.PrimitiveCell(vectors, unit=tb.NM)
    cell.add_orbital((0.0, 0.0), label="C_pz")
    cell.add_orbital((1 / 3., 1 / 3.), label="C_pz")
    cell.add_hopping((0, 0), 0, 1, t)
    cell.add_hopping((1, 0), 1, 0, t)
    cell.add_hopping((0, 1), 1, 0, t)
    return cell


def test_dyn_pol_prb():
    """
    Reproducing Phys. Rev. B 84, 035439 (2011) with |q| = 1/a and theta = 30
    degrees.

    :return: None
    """
    # Construct primitive cell
    a = 0.142
    t = 3.0
    cell = make_cell(a, t)

    # Set parameter for Lindhard function
    energy_max = 10
    energy_step = 1000
    mesh_size = (4096, 4096, 1)
    mu = 0.0
    temp = 300
    g_s = 1
    back_epsilon = 1
    dimension = 2
    enable_mpi = True
    q_points = 1 / a * np.array([[0.86602540, 0.5, 0.0]])

    # Instantiate Lindhard calculator
    lindhard = tb.Lindhard(cell=cell, energy_max=energy_max,
                           energy_step=energy_step, kmesh_size=mesh_size,
                           mu=mu, temperature=temp, g_s=g_s,
                           back_epsilon=back_epsilon, dimension=dimension,
                           enable_mpi=enable_mpi)

    # Calculate dyn_pol
    omegas, dyn_pol = lindhard.calc_dyn_pol_arbitrary(q_points)
    if lindhard.is_master:
        np.save("omegas_dp", omegas)
        np.save("dp", dyn_pol)


def test_epsilon_prb():
    """
    Reproducing Phys. Rev. B 84, 035439 (2011) with |q| = 0.476 / Angstrom
    and theta = 30 degrees.

    :return: None
    """
    # Construct primitive cell
    a = 0.142
    t = 3.0
    cell = make_cell(a, t)

    # Set parameter for Lindhard function
    energy_max = 10
    energy_step = 1000
    mesh_size = (4096, 4096, 1)
    mu = 0.0
    temp = 300
    g_s = 1
    back_epsilon = 1
    dimension = 2
    enable_mpi = True
    q_points = np.array([[4.122280922013927, 2.38, 0.0]])

    # Instantiate Lindhard calculator
    lindhard = tb.Lindhard(cell=cell, energy_max=energy_max,
                           energy_step=energy_step, kmesh_size=mesh_size,
                           mu=mu, temperature=temp, g_s=g_s,
                           back_epsilon=back_epsilon, dimension=dimension,
                           enable_mpi=enable_mpi)

    # Evaluate dielectric function
    omegas, dyn_pol = lindhard.calc_dyn_pol_arbitrary(q_points)
    epsilon = lindhard.calc_epsilon(q_points, dyn_pol)
    if lindhard.is_master:
        np.save("omegas_eps", omegas)
        np.save("eps", epsilon)


def test_ac_cond():
    """
    Calculate the AC conductivity of monolayer graphene.

    :return: None
    """
    # Construct primitive cell
    a = 0.142
    t = 3.0
    cell = make_cell(a, t)

    # Set parameter for Lindhard function
    energy_max = 10
    energy_step = 1000
    mesh_size = (4096, 4096, 1)
    mu = 0.0
    temp = 300
    back_epsilon = 1.0
    delta = 0.005
    g_s = 2
    dimension = 2
    enable_mpi = True

    # Instantiate Lindhard calculator
    lindhard = tb.Lindhard(cell=cell, energy_max=energy_max,
                           energy_step=energy_step, kmesh_size=mesh_size,
                           mu=mu, temperature=temp,
                           back_epsilon=back_epsilon, delta=delta, g_s=g_s,
                           dimension=dimension, enable_mpi=enable_mpi)

    # Evaluate AC Cond.
    omegas, ac_cond = lindhard.calc_ac_cond()
    if lindhard.is_master:
        np.save("omegas_ac", omegas)
        np.save("ac", ac_cond)


if __name__ == "__main__":
    print("Calculating dp")
    test_dyn_pol_prb()
    print("Calculating eps")
    test_epsilon_prb()
    print("Calculating ac")
    test_ac_cond()
