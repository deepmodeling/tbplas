"""graphene.py

Graphene TBPM example.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
import tipsi
import params_simple_graphene
import numpy as np

def main():

    #########################
    # SIMULATION PARAMETERS #
    #########################
    
    # available cores for parallel sample construction
    nr_processes = 24
    
    # band plot resolution
    res_bands = 100
    
    # sample size in unit cells
    W = 1024 # must be even
    H = 1024
    
    a = 0.24 # lattice constant in nm
    t = 2.8 # hopping value
    e = 0. # onsite potential
    
    # create lattice, hop_dict and pbc_wrap
    lat = params_simple_graphene.graphene_lattice(a)
    hop_dict = params_simple_graphene.graphene_hop_dict(t, e)
    def pbc_wrap(unit_cell_coords, orbital):
        return params_simple_graphene.graphene_pbc(W, H, unit_cell_coords, orbital)
    
    #######################
    # SAMPLE CONSTRUCTION #
    #######################
    
    # create SiteSet objects
    site_set = params_simple_graphene.graphene_sheet_sites(W, H)
    
    # make sample
    sample = tipsi.Sample(lat, site_set, pbc_wrap, nr_processes)

    # apply HopDict
    sample.add_hop_dict(hop_dict)
    
    # rescale Hamiltonian
    sample.rescale_H(9.)
    
    #########################
    # SIMULATION PARAMETERS #
    #########################
    
    config = tipsi.Config()
    config.generic['nr_time_steps'] = 1024
    config.generic['nr_random_samples'] = 1
    config.generic['energy_range'] = 20.
    config.generic['correct_spin'] = True
    config.dyn_pol['q_points'] = [[0.5,0.0,0.0] , [1.0,0.0,0.0]]
    
    ############
    # RUN TBPM #
    ############
    
    # get DOS
    corr_DOS = tipsi.corr_DOS(sample, config)
    energies_DOS, DOS = tipsi.analyze_corr_DOS(sample, config, corr_DOS)
    plt.plot(energies_DOS, DOS)
    plt.xlabel("E (eV)")
    plt.ylabel("DOS")
    plt.savefig("graphene_DOS.png")
    plt.close()

    # get AC conductivity
    corr_AC = tipsi.corr_AC(sample, config)
    omegas_AC, AC = tipsi.analyze_corr_AC(sample, config, corr_AC)
    plt.plot(omegas_AC, AC[0])
    plt.xlim((0,10))
    plt.ylim((0,8))
    plt.xlabel("hbar * omega (eV)")
    plt.ylabel("sigma_xx (sigma_0)")
    plt.savefig("graphene_ACxx.png")
    plt.close()

    # get dyn pol
    corr_dyn_pol = tipsi.corr_dyn_pol(sample, config)
    qval, omegas, dyn_pol = tipsi.analyze_corr_dyn_pol(sample, config, corr_dyn_pol)
    qval, omegas, epsilon = tipsi.analyze_corr_dyn_pol(sample, config, dyn_pol)
    plt.plot(omegas, dyn_pol[0,:].imag)
    plt.plot(omegas, dyn_pol[1,:].imag)
    plt.xlabel("hbar * omega (eV)")
    plt.ylabel("Re(dp)")
    plt.savefig("graphene_dp_imag.png")
    plt.close()
    
if __name__ == '__main__':
    main()
            
