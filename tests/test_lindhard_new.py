#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

import tbplas as tb
from tbplas.lindhard import Lindhard


def main():
    # Construct primitive cell
    cell = tb.make_antimonene(with_soc=False)

    # Set parameter for Lindhard function
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
                        delta=0.05)

    # Evaluate ac conductivity
    omegas_au, ac_cond = lindhard.calc_ac_cond()

    # Evaluate real part of dielectric function at long-wave limit
    d = 0.36 / tb.BOHR2NM
    epsilon_re = 1 - ac_cond.imag / omegas_au / d / back_epsilon

    # Plot
    plt.plot(omegas_au*tb.HAR2EV, epsilon_re, color="r")
    plt.minorticks_on()
    plt.savefig("epsilon.png")
    plt.close()


if __name__ == '__main__':
    main()
