#! /usr/bin/env python

import matplotlib.pyplot as plt

import tbplas as tb


# In this tutorial we show how to calculate different properties using the
# TBPM algorithms implemented in TBPlaS. First, we create a large graphene
# sample with periodic boundary conditions.
prim_cell = tb.make_graphene_diamond()
super_cell_pbc = tb.SuperCell(prim_cell, dim=(120, 120, 1),
                              pbc=(True, True, True))
sample_pbc = tb.Sample(super_cell_pbc)

# The Hamiltonian of sample must be rescaled for any TBPM calculation.
# Otherwise, the wave function will diverge during propagation. A scaling
# factor is need to scale the Hamiltonian. If the factor is not specified,
# it will be estimated by inspecting the Hamiltonian automatically.
sample_pbc.rescale_ham(9.0)

# Then we set up the parameters governing the TBPM calculation.
# We will use 4 random samples. For each sample 256 the propagation will take
# 256 steps.
config = tb.Config()
config.generic['nr_random_samples'] = 4
config.generic['nr_time_steps'] = 256

# Then we create a solver and an analyzer from sample and configuration.
solver = tb.Solver(sample_pbc, config)
analyzer = tb.Analyzer(sample_pbc, config)

# Get DOS correlation function with solver.
corr_dos = solver.calc_corr_dos()

# Analyze DOS correlation function with analyzer.
energies_dos, dos = analyzer.calc_dos(corr_dos)

# Plot DOS
plt.plot(energies_dos, dos)
plt.xlabel("Energy (eV)")
plt.ylabel("DOS")
plt.savefig("DOS.png")
plt.close()

# Now we demonstrate more features of TBPM algorithms. First, we need to put
# more parameters into configuration.
config.generic['correct_spin'] = True
config.dyn_pol['q_points'] = [[1., 0., 0.]]
config.DC_conductivity['energy_limits'] = (-5, 5)

# Then we create new solver and analyzer
solver = tb.Solver(sample_pbc, config)
analyzer = tb.Analyzer(sample_pbc, config)

# Get AC conductivity
corr_ac = solver.calc_corr_ac_cond()
omegas_ac, ac = analyzer.calc_ac_cond(corr_ac)
plt.plot(omegas_ac, ac[0].real)
plt.xlabel("Energy (eV)")
plt.ylabel("sigma_xx")
plt.savefig("ACxx.png")
plt.close()

# Get dyn pol
corr_dyn_pol = solver.calc_corr_dyn_pol()
q_val, omegas, dyn_pol = analyzer.calc_dyn_pol(corr_dyn_pol)
plt.plot(omegas, -dyn_pol[0].imag)
plt.xlabel("Energy (eV)")
plt.ylabel("-Im(dp)")
plt.savefig("dp_imag.png")
plt.close()

# Get DC conductivity
corr_dos, corr_dc = solver.calc_corr_dc_cond()
energies_dc, dc = analyzer.calc_dc_cond(corr_dos, corr_dc)
plt.plot(energies_dc, dc[0])
plt.xlabel("Energy (eV)")
plt.ylabel("DC conductivity")
plt.savefig("DC.png")
plt.close()
