"""graphene_tbpm.py

Graphene TBPM example.
"""

import matplotlib.pyplot as plt
import numpy as np

import tbplas
from tbplas.materials import graphene


def main():

    # make 1000*1000 unit cell sample
    sample = graphene.sample_rectangle(1024, 1024, nr_processes=8)

    # set config parameters
    config = tbplas.Config(sample)
    config.generic['nr_time_steps'] = 256
    config.generic['nr_random_samples'] = 1
    config.generic['energy_range'] = 20.
    config.generic['correct_spin'] = True
    config.dyn_pol['q_points'] = [[1., 0., 0.]]
    config.DC_conductivity['energy_limits'] = (-0.3, 0.3)
    config.LDOS['site_indices'] = [0]
    config.LDOS['delta'] = 0.1
    config.LDOS['recursion_depth'] = 2000
    config.save()

    # get DOS
    corr_DOS = tbplas.corr_DOS(sample, config)
    energies_DOS, DOS = tbplas.analyze_corr_DOS(config, corr_DOS)
    plt.plot(energies_DOS, DOS)
    plt.xlabel("E (eV)")
    plt.ylabel("DOS")
    plt.savefig("graphene_DOS.png")
    plt.close()

    # get LDOS using Haydock recursion method
    energies_LDOS, LDOS = tbplas.get_ldos_haydock(sample, config)
    plt.plot(energies_LDOS, LDOS)
    plt.xlabel("E (eV)")
    plt.ylabel("LDOS")
    plt.savefig("graphene_LDOS.png")
    plt.close()

    # get AC conductivity
    corr_AC = tbplas.corr_AC(sample, config)
    omegas_AC, AC = tbplas.analyze_corr_AC(config, corr_AC)
    plt.plot(omegas_AC, AC[0])
    plt.xlabel("hbar * omega (eV)")
    plt.ylabel("sigma_xx (sigma_0)")
    plt.savefig("graphene_ACxx.png")
    plt.close()

    # get dyn pol
    corr_dyn_pol = tbplas.corr_dyn_pol(sample, config)
    qval, omegas, dyn_pol = tbplas.analyze_corr_dyn_pol(config, corr_dyn_pol)
    qval, omegas, epsilon = tbplas.analyze_corr_dyn_pol(config, dyn_pol)
    plt.plot(omegas, -1 * dyn_pol[0, :].imag)
    plt.xlabel("hbar * omega (eV)")
    plt.ylabel("-Im(dp)")
    plt.savefig("graphene_dp_imag.png")
    plt.close()

    # get DC conductivity
    corr_DOS, corr_DC = tbplas.corr_DC(sample, config)
    energies_DC, DC = tbplas.analyze_corr_DC(config, corr_DOS, corr_DC)
    plt.plot(energies_DC, DC[0, :])
    plt.xlabel("E (eV)")
    plt.ylabel("DC conductivity")
    plt.savefig("graphene_DC.png")
    plt.close()


if __name__ == '__main__':
    main()
