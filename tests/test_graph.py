#! /usr/bin/env python

import unittest
from math import sqrt

import numpy as np
import matplotlib.pyplot as plt

from tbplas import PrimitiveCell, SuperCell, Sample, Config, Solver, Analyzer


def make_sample_new(shape, enable_pbc=True):
    """Make sample for given shape and pbc with new version of builder."""
    a = 0.246
    vectors = [[0.5*a, 0.5*sqrt(3)*a, 0.],
               [-0.5*a, 0.5*sqrt(3)*a, 0.],
               [0.0, 0.0, 1.]]
    cell = PrimitiveCell(np.array(vectors), unit=1.0)
    cell.add_orbital((0.0,  0.0), 0.0)
    cell.add_orbital((1./3, 1./3), 0.0)
    cell.add_hopping((0, 0), 0, 1, 2.7)
    cell.add_hopping((1, 0), 1, 0, 2.7)
    cell.add_hopping((0, 1), 1, 0, 2.7)
    if enable_pbc:
        pbc = (True, True)
    else:
        pbc = (False, False)
    sc = SuperCell(cell, dim=shape, pbc=pbc)
    sample = Sample(sc)
    sample.rescale_ham(9.0)
    return sample


class MyTest(unittest.TestCase):

    def test_tbpm(self):
        """
        Test the new builder with real-case calculations.

        :return: None.
        """
        # sample = make_sample_old((100, 100), enable_pbc=True)
        sample = make_sample_new((100, 100), enable_pbc=True)

        # set config parameters
        config = Config()
        config.generic['nr_time_steps'] = 256
        config.generic['nr_random_samples'] = 4
        config.generic['correct_spin'] = True
        config.dyn_pol['q_points'] = [[1., 0., 0.]]
        config.DC_conductivity['energy_limits'] = (-0.3, 0.3)
        config.LDOS['site_indices'] = [0]
        config.LDOS['delta'] = 0.1
        config.LDOS['recursion_depth'] = 2000

        # create Solver and Analyzer
        solver = Solver(sample, config)
        solver.save_config()
        analyzer = Analyzer(sample, config)

        # get DOS
        corr_dos = solver.calc_corr_dos()
        energies_dos, dos = analyzer.calc_dos(corr_dos)
        if analyzer.is_master:
            plt.plot(energies_dos, dos)
            plt.xlabel("E (eV)")
            plt.ylabel("DOS")
            plt.savefig("DOS.png")
            plt.close()

        # get AC conductivity
        corr_ac = solver.calc_corr_ac_cond()
        omegas_ac, ac = analyzer.calc_ac_cond(corr_ac)
        if analyzer.is_master:
            plt.plot(omegas_ac, ac[0].real)
            plt.xlabel("h_bar * omega (eV)")
            plt.ylabel("sigma_xx (sigma_0)")
            plt.savefig("ACxx.png")
            plt.close()

        # get dyn pol
        corr_dyn_pol = solver.calc_corr_dyn_pol()
        q_val, omegas, dyn_pol = analyzer.calc_dyn_pol(corr_dyn_pol)
        if analyzer.is_master:
            plt.plot(omegas, -1 * dyn_pol[0, :].imag)
            plt.xlabel("h_bar * omega (eV)")
            plt.ylabel("-Im(dp)")
            plt.savefig("dp_imag.png")
            plt.close()

        # get DC conductivity
        corr_dos, corr_dc = solver.calc_corr_dc_cond()
        energies_dc, dc = analyzer.calc_dc_cond(corr_dos, corr_dc)
        if analyzer.is_master:
            plt.plot(energies_dc, dc[0, :])
            plt.xlabel("E (eV)")
            plt.ylabel("DC conductivity")
            plt.savefig("DC.png")
            plt.close()


if __name__ == "__main__":
    unittest.main()
