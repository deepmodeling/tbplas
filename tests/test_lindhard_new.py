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
    mesh_size = (120, 120, 1)
    mu = 0.0
    temp = 300
    back_epsilon = 1

    regular = False
    use_fortran = True
    wrap = True
    if regular:
        q_points = [(1, 1, 0)]
    else:
        q_points = np.array([(1./mesh_size[0], 1./mesh_size[1], 0)])

    # Calculate dyn_pol using Lindhard function
    lindhard = Lindhard(cell=cell, energy_max=energy_max,
                        energy_step=energy_step, kmesh_size=mesh_size,
                        mu=mu, temperature=temp, back_epsilon=back_epsilon)
    if regular:
        print(lindhard.grid2cart(q_points))
        omegas, dyn_pol = lindhard.calc_dyn_pol_regular(q_points, use_fortran)
    else:
        print(lindhard.frac2cart(q_points))
        omegas, dyn_pol = lindhard.calc_dyn_pol_arbitrary(q_points, use_fortran,
                                                          wrap)

    # Plot
    for i in range(len(q_points)):
        plt.plot(omegas, -dyn_pol.imag[i])
    plt.savefig("lindhard_im_dyn_pol.png")
    plt.close()


if __name__ == '__main__':
    main()
