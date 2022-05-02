#! /usr/bin/env python

import matplotlib.pyplot as plt

import tbplas as tb


# NOTE: TBPlaS uses OpenMP for parallel TBPM calculations by default. So we
# need to set up environment variable OMP_NUM_THREADS to the number of physical
# cores of your computer. If you have compiled TBPlaS with MKL, set up
# MKL_NUM_THREADS instead.

# NOTE: if your computer has HyperThreading enabled in BIOS or UEFI, then the
# number of available cores will be double of the physical cores. DO NOT use
# the virtual cores from HyperThreading since there will be significant
# performance loss.

# NOTE: we do not employ the term of 'CPU' since it is ambiguous. Instead, we
# use 'socket' for the 'CPU' you bought from the vendor and 'core' for the
# 'CPUs' encapsulated within each socket. For example, an Intel Core i7-10700H
# CPU is actually a socket with 8 physical cores. An AMD Ryzen R7-4800H CPU is
# also a socket with 8 physical cores. Most computers for home or office uses
# have only one socket. Workstations or computational nodes at High Performance
# Computer Center have 2 or 4 sockets. Keep these concepts clearly in mind as
# they are very important to efficiency parallelism.

# In this tutorial we show how to calculate different properties using the
# TBPM algorithms implemented in TBPlaS. First, we create a large graphene
# sample with periodic boundary conditions.
prim_cell = tb.make_graphene_diamond()
super_cell_pbc = tb.SuperCell(prim_cell, dim=(480, 480, 1),
                              pbc=(True, True, True))
sample_pbc = tb.Sample(super_cell_pbc)

# NOTE: the Hamiltonian of sample must be rescaled for any TBPM calculation.
# Otherwise, the wave function will diverge during propagation. A scaling
# factor is need to scale the Hamiltonian. If the factor is not specified,
# it will be estimated by inspecting the Hamiltonian automatically.
sample_pbc.rescale_ham(9.0)

# Then we set up the parameters governing the TBPM calculation.
# We will use 4 random samples. For each sample 256 the propagation will take
# 256 steps. The DOS will be evaluated on the energy range of [-10, 10] eV.
config = tb.Config()
config.generic['nr_random_samples'] = 4
config.generic['nr_time_steps'] = 256
config.generic['energy_range'] = 20.

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
config.LDOS['site_indices'] = [0]
config.LDOS['delta'] = 0.1
config.LDOS['recursion_depth'] = 2000

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
plt.plot(omegas, -1 * dyn_pol[0, :].imag)
plt.xlabel("Energy (eV)")
plt.ylabel("-Im(dp)")
plt.savefig("dp_imag.png")
plt.close()

# Get DC conductivity
corr_dos, corr_dc = solver.calc_corr_dc_cond()
energies_dc, dc = analyzer.calc_dc_cond(corr_dos, corr_dc)
plt.plot(energies_dc, dc[0, :])
plt.xlabel("Energy (eV)")
plt.ylabel("DC conductivity")
plt.savefig("DC.png")
plt.close()
