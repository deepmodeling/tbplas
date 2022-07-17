#! /usr/bin/env python

import numpy as np
import tbplas as tb


# Construct primitive cell
t = 3.0  # Absolute hopping energy in eV
a = 0.142  # C-C distance in NM
vectors = tb.gen_lattice_vectors(a=0.246, b=0.246, c=1.0, gamma=60)
cell = tb.PrimitiveCell(vectors, unit=tb.NM)
cell.add_orbital([0.0, 0.0], label="C_pz")
cell.add_orbital([1 / 3., 1 / 3.], label="C_pz")
cell.add_hopping([0, 0], 0, 1, t)
cell.add_hopping([1, 0], 1, 0, t)
cell.add_hopping([0, 1], 1, 0, t)

# Create sample
super_cell_pbc = tb.SuperCell(cell, dim=(4096, 4096, 1),
                              pbc=(True, True, False))
sample_pbc = tb.Sample(super_cell_pbc)
sample_pbc.rescale_ham(9.0)

# Config, solver and analyzer
config = tb.Config()
config.generic['nr_random_samples'] = 1
config.generic['nr_time_steps'] = 1024
config.generic['correct_spin'] = True
solver = tb.Solver(sample_pbc, config, prefix="test")
analyzer = tb.Analyzer(sample_pbc, config, dimension=2)
vis = tb.Visualizer()

from_scratch = True
if from_scratch:
    corr_ac = solver.calc_corr_ac_cond()
else:
    corr_ac = np.load("sim_data/test.corr_AC.npy")
omegas, ac_cond = analyzer.calc_ac_cond(corr_ac)

np.save("omegas_ac", omegas)
np.save("ac", ac_cond)
