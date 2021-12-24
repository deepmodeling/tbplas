#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

import tbplas as tb
from tbplas.lindhard import Lindhard


def main():
    # Construct primitive cell
    cell = tb.make_graphene_diamond()

    # Set parameter for Lindhard function
    energy_max = 10
    energy_step = 2048
    mu = 0.0
    temp = 300
    back_epsilon = 1
    mesh_size = (1200, 1200, 1)
    use_fortran = True
    q_points = np.array([[4.122280922013927, 2.38, 0.0]])

    # Instantiate Lindhard calculator
    lindhard = Lindhard(cell=cell, energy_max=energy_max,
                        energy_step=energy_step, kmesh_size=mesh_size,
                        mu=mu, temperature=temp, back_epsilon=back_epsilon)

    # Evaluate dielectric function
    omegas, epsilon = lindhard.calc_epsilon_arbitrary(q_points, use_fortran)

    # Plot
    for i in range(len(q_points)):
        plt.plot(omegas, epsilon[i].real, color="r")
    plt.minorticks_on()
    plt.savefig("epsilon.png")
    plt.close()


if __name__ == '__main__':
    main()
