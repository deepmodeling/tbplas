#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import tbplas as tb


# Make graphene primitive cell
t = 3.0
vectors = tb.gen_lattice_vectors(a=0.246, b=0.246, c=1.0, gamma=60)
cell = tb.PrimitiveCell(vectors, unit=tb.NM)
cell.add_orbital([0.0, 0.0], label="C_pz")
cell.add_orbital([1 / 3., 1 / 3.], label="C_pz")
cell.add_hopping([0, 0], 0, 1, t)
cell.add_hopping([1, 0], 1, 0, t)
cell.add_hopping([0, 1], 1, 0, t)

# Create a Lindhard object
lind = tb.Lindhard(cell=cell, energy_max=10, energy_step=1000,
                   kmesh_size=(600, 600, 1), mu=0.0, temperature=300, g_s=2,
                   back_epsilon=1.0, dimension=2)

# Create a timer
timer = tb.Timer()

# Calculate dynamic polarization with calc_dyn_pol_regular
q_grid = np.array([[20, 20, 0]])
timer.tic("regular")
omegas, dp_reg = lind.calc_dyn_pol_regular(q_grid)
timer.toc("regular")
plt.plot(omegas, dp_reg[0].imag, color="red", label="Regular")
plt.legend()
plt.show()
plt.close()

# Calculate dynamic polarization with calc_dyn_pol_arbitrary
q_cart = lind.grid2cart(q_grid, unit=tb.NM)
timer.tic("arbitrary")
omegas, dp_arb = lind.calc_dyn_pol_arbitrary(q_cart)
timer.toc("arbitrary")
plt.plot(omegas, dp_arb[0].imag, color="blue", label="Arbitrary")
plt.legend()
plt.show()
plt.close()

timer.report_total_time()

# Reproduce the result of Phys. Rev. B 84, 035439 (2011) with
# |q| = 4.76 / nm and theta = 30 degrees.
lind = tb.Lindhard(cell=cell, energy_max=18, energy_step=1800,
                   kmesh_size=(1200, 1200, 1), mu=0.0, temperature=300, g_s=1,
                   back_epsilon=1.0, dimension=2)
q_points = 4.76 * np.array([[0.86602540, 0.5, 0.0]])
omegas, dyn_pol = lind.calc_dyn_pol_arbitrary(q_points)
epsilon = lind.calc_epsilon(q_points, dyn_pol)
plt.plot(omegas, epsilon[0].real, color="red")
plt.xticks(np.linspace(0.0, 18.0, 10))
plt.show()
plt.close()

# Reproduce the result of Phys. Rev. B 82, 115448 (2010).
lind = tb.Lindhard(cell=cell, energy_max=t*3.5, energy_step=2048,
                   kmesh_size=(2048, 2048, 1), mu=0.0, temperature=300.0,
                   g_s=2, back_epsilon=1.0, dimension=2)
omegas, ac_cond = lind.calc_ac_cond(component="xx")
omegas /= t
ac_cond *= 4
plt.plot(omegas, ac_cond.real, color="red")
plt.minorticks_on()
plt.show()
plt.close()
