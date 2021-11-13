"""read_corr.py

Read graphene TBPM correlation functions from sim_data folder.
"""

import matplotlib.pyplot as plt
import numpy as np

import tbplas


def main():

    # set timestamp
    timestamp = "1522172330"

    # read config file
    config = tbplas.read_config("sim_data/" + timestamp + "config.pkl")

    # get DOS
    corr_DOS = tbplas.read_corr_DOS("sim_data/" + timestamp + "corr_DOS.dat")
    energies_DOS, DOS = tbplas.analyze_corr_DOS(config, corr_DOS)
    plt.plot(energies_DOS, DOS)
    plt.xlabel("E (eV)")
    plt.ylabel("DOS")
    plt.savefig("graphene_DOS.png")
    plt.close()

    # get AC conductivity
    corr_AC = tbplas.read_corr_AC("sim_data/" + timestamp + "corr_AC.dat")
    omegas_AC, AC = tbplas.analyze_corr_AC(config, corr_AC)
    plt.plot(omegas_AC, AC[0])
    plt.xlabel("hbar * omega (eV)")
    plt.ylabel("sigma_xx (sigma_0)")
    plt.savefig("graphene_ACxx.png")
    plt.close()

    # get dyn pol
    corr_dyn_pol = tbplas.read_corr_dyn_pol("sim_data/" + timestamp +
                                           "corr_dyn_pol.dat")
    qval, omegas, dyn_pol = tbplas.analyze_corr_dyn_pol(config, corr_dyn_pol)
    qval, omegas, epsilon = tbplas.analyze_corr_dyn_pol(config, dyn_pol)
    plt.plot(omegas, dyn_pol[0, :].imag)
    plt.xlabel("hbar * omega (eV)")
    plt.ylabel("Im(dp)")
    plt.savefig("graphene_dp_imag.png")
    plt.close()

    # get DC conductivity
    corr_DOS = tbplas.read_corr_DOS("sim_data/" + timestamp + "corr_DOS.dat")
    corr_DC = tbplas.read_corr_DC("sim_data/" + timestamp + "corr_DC.dat")
    energies_DC, DC = tbplas.analyze_corr_DC(config, corr_DOS, corr_DC)
    plt.plot(energies_DC, DC[0, :])
    plt.xlabel("E (eV)")
    plt.ylabel("DC conductivity")
    plt.savefig("graphene_DC.png")
    plt.close()


if __name__ == '__main__':
    main()
