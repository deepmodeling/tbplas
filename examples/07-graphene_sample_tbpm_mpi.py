#! /usr/bin/env python

import matplotlib.pyplot as plt

import tbplas as tb


# In this tutorial we show how to use hybrid parallelism using both MPI+OpenMP.
# You should read the notes in example06 carefully to correctly set up the
# parallel environment.

# The hybrid parallelization firstly distributes random samples among MPI
# processes. For each random sample, matrix operations are then parallelized
# over OpenMP threads. So, the product of MPI processes and OpenMP threads
# should equal to the PHYSICAL cores of your computer. For example, if your
# computer has one socket with 8 cores, then you can use the following settings
#
#     number of MPI processes    number of OpenMP threads
#                           1                           8
#                           2                           4
#                           4                           2
#                           8                           1
#
# Also for each setting, the number of random samples should be a multiple of
# the number MPI processes. Determining the optimal setting is not a trivial
# task, which requires a lot of tests.

# Firstly, we build the sample as in example06.
prim_cell = tb.make_graphene_diamond()
super_cell_pbc = tb.SuperCell(prim_cell, dim=(120, 120, 1),
                              pbc=(True, True, True))
sample_pbc = tb.Sample(super_cell_pbc)
sample_pbc.rescale_ham(9.0)

# Then we set the calculation parameters
config = tb.Config()
config.generic['nr_random_samples'] = 4
config.generic['nr_time_steps'] = 256
config.dyn_pol['q_points'] = [[1., 0., 0.]]
config.DC_conductivity['energy_limits'] = (-0.3, 0.3)
config.LDOS['site_indices'] = [0]
config.LDOS['delta'] = 0.1
config.LDOS['recursion_depth'] = 2000

# Then we create a solver and an analyzer. Note that we need to add an option
# of 'enable_mpi=True' to enable mpi support.
solver = tb.Solver(sample_pbc, config, enable_mpi=True)
analyzer = tb.Analyzer(sample_pbc, config, enable_mpi=True)

# Then we calculate and plot the properties.
# We are going to use 2 MPI processes, with each process spawning 4 OpenMP
# threads. So the command to run this script should be
#   export OMP_NUM_THREADS=4
#   mpirun -np 2 python ./xxxx.py
# Also, note that the plots should be done on master node.

# Get DOS
corr_dos = solver.calc_corr_dos()
energies_dos, dos = analyzer.calc_dos(corr_dos)
if analyzer.is_master:
    plt.plot(energies_dos, dos)
    plt.xlabel("E (eV)")
    plt.ylabel("DOS")
    plt.savefig("DOS.png")
    plt.close()

config.generic['correct_spin'] = True

# Get AC conductivity
corr_ac = solver.calc_corr_ac_cond()
omegas_ac, ac = analyzer.calc_ac_cond(corr_ac)
if analyzer.is_master:
    plt.plot(omegas_ac, ac[0])
    plt.xlabel("h_bar * omega (eV)")
    plt.ylabel("sigma_xx (sigma_0)")
    plt.savefig("ACxx.png")
    plt.close()

# Get dyn pol
corr_dyn_pol = solver.calc_corr_dyn_pol()
q_val, omegas, dyn_pol = analyzer.calc_dyn_pol(corr_dyn_pol)
if analyzer.is_master:
    plt.plot(omegas, -1 * dyn_pol[0, :].imag)
    plt.xlabel("h_bar * omega (eV)")
    plt.ylabel("-Im(dp)")
    plt.savefig("dp_imag.png")
    plt.close()

# Get DC conductivity
corr_dos, corr_dc = solver.calc_corr_dc_cond()
energies_dc, dc = analyzer.calc_dc_cond(corr_dos, corr_dc)
if analyzer.is_master:
    plt.plot(energies_dc, dc[0, :])
    plt.xlabel("E (eV)")
    plt.ylabel("DC conductivity")
    plt.savefig("DC.png")
    plt.close()
