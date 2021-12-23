#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

import tbplas as tb
from tbplas.lindhard import Lindhard


def main():
    # Construct primitive cell
    cell = tb.make_graphene_diamond()
    t = 2.7  # Absolute hopping energy
    a = 0.142  # C-C distance in NM

    # Set parameter for Lindhard function
    energy_max = 10
    energy_step = 2048
    mesh_size = (1200, 1200, 1)
    mu = 0.0
    temp = 300
    back_epsilon = 1
    itest = 0

    # Test 0: reproducing Phys. Rev. B 84, 035439 (2011)
    # |q| = 1/a with theta = 30 degrees
    if itest == 0:
        regular = False
        use_fortran = True
        q_points = 1 / 0.142 * np.array([[0.86602540, 0.5, 0.0]])
        print(q_points)

    # Test 1: cross-reference of Fortran and cython extension
    else:
        regular = False
        use_fortran = True
        if regular:
            q_points = [(1000, 1000, 0)]
        else:
            q_points = np.array([[8.51380123, 4.91544543, 0.]])

    # Calculate dyn_pol using Lindhard function
    lindhard = Lindhard(cell=cell, energy_max=energy_max,
                        energy_step=energy_step, kmesh_size=mesh_size,
                        mu=mu, temperature=temp, back_epsilon=back_epsilon)
    if regular:
        omegas, dyn_pol = lindhard.calc_dyn_pol_regular(q_points, use_fortran)
    else:
        omegas, dyn_pol = lindhard.calc_dyn_pol_arbitrary(q_points, use_fortran)

    # Plot
    for i in range(len(q_points)):
        plt.plot(omegas/t, -dyn_pol.imag[i])
    plt.savefig("lindhard_im_dyn_pol.png")
    plt.close()


if __name__ == '__main__':
    main()
