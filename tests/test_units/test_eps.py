#! /usr/bin/env python

import numpy as np
import tbplas as tb
from test_lindhard import make_cell


# Construct primitive cell
t = 3.0  # Absolute hopping energy in eV
a = 0.142  # C-C distance in NM
cell = make_cell(a, t)

# Create sample
super_cell_pbc = tb.SuperCell(cell, dim=(4096, 4096, 1),
                              pbc=(True, True, False))
sample_pbc = tb.Sample(super_cell_pbc)
sample_pbc.rescale_ham(9.0)

# Config, solver and analyzer
config = tb.Config()
config.generic['nr_random_samples'] = 1
config.generic['nr_time_steps'] = 1024
config.generic['correct_spin'] = False
config.dyn_pol['q_points'] = np.array([[4.122280922013927, 2.38, 0.0]])
solver = tb.Solver(sample_pbc, config, prefix="test")
analyzer = tb.Analyzer(sample_pbc, config, dimension=2)
vis = tb.Visualizer()

from_scratch = True
if from_scratch:
    corr_dp = solver.calc_corr_dyn_pol()
else:
    corr_dp = np.load("sim_data/test.corr_dyn_pol.npy")
q, omegas, dp = analyzer.calc_dyn_pol(corr_dp)
epsilon = analyzer.calc_epsilon(dp)

np.save("omegas_eps", omegas)
np.save("eps", epsilon)
