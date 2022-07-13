#! /usr/bin/env python

import numpy as np
import tbplas as tb


# Construct primitive cell
cell = tb.make_graphene_rect()

# Create sample
super_cell_pbc = tb.SuperCell(cell, dim=(500, 200, 1),
                              pbc=(True, True, False))
sample_pbc = tb.Sample(super_cell_pbc)
sample_pbc.set_magnetic_field(100.0)
sample_pbc.rescale_ham(9.0)

# Config, solver and analyzer
config = tb.Config()
config.generic['nr_random_samples'] = 1
config.generic['nr_time_steps'] = 1024
config.generic['correct_spin'] = True
config.set_temperature(10.0)
config.dckb['direction'] = 1
config.dckb['energies'] = np.linspace(-1.0, 1.0, 1000)
solver = tb.Solver(sample_pbc, config, prefix="test")
analyzer = tb.Analyzer(sample_pbc, config, dimension=2)
vis = tb.Visualizer()

#mu_hall = solver.calc_hall_mu()
mu_hall = np.load("xy/mu_hall.npy")
omegas, hall = analyzer.calc_hall_cond(mu_hall, unit="h")
np.save("omegas", omegas)
np.save("hall", hall)
