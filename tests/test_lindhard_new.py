#! /usr/bin/env python
import matplotlib.pyplot as plt

import tbplas as tb
from tbplas.lindhard import Lindhard


def test_antimonene():
    # Construct primitive cell
    cell = tb.make_antimonene(with_soc=False)

    # Set parameters for Lindhard function
    energy_max = 10
    energy_step = 2048
    mu = 0.0
    temp = 300
    back_epsilon = 3.9
    mesh_size = (120, 120, 1)

    # Instantiate Lindhard calculator
    lindhard = Lindhard(cell=cell, energy_max=energy_max,
                        energy_step=energy_step, kmesh_size=mesh_size,
                        mu=mu, temperature=temp, back_epsilon=back_epsilon,
                        delta=0.05, dimension=2)

    omegas, ac_cond = lindhard.calc_ac_cond_kg()
    # omegas, ac_cond = lindhard.calc_ac_cond_prb()
    d = 0.36
    epsilon_re = 1 - ac_cond.imag / omegas / d / back_epsilon

    # Plot
    plt.plot(omegas, epsilon_re, color="r")
    plt.minorticks_on()
    plt.savefig("epsilon.png")
    plt.close()


def test_graphene():
    cell = tb.make_graphene_diamond()

    energy_max = 10
    energy_step = 2048
    mu = 0.0
    temp = 300
    back_epsilon = 1.0
    mesh_size = (1024, 1024, 1)

    lindhard = Lindhard(cell=cell, energy_max=energy_max,
                        energy_step=energy_step, kmesh_size=mesh_size,
                        mu=mu, temperature=temp, back_epsilon=back_epsilon,
                        delta=0.005)

    omegas, ac_cond = lindhard.calc_ac_cond_kg()
    # omegas, ac_cond = lindhard.calc_ac_cond_prb()
    omegas /= 2.7

    plt.plot(omegas, ac_cond.real, color="r")
    plt.minorticks_on()
    plt.savefig("sigma_xx.png")
    plt.close()


if __name__ == '__main__':
    test_antimonene()
