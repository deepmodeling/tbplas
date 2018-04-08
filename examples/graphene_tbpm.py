"""graphene_tbpm.py

Graphene TBPM example.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("..")
import tipsi
from tipsi.materials import graphene

def main():

    #########################
    # SIMULATION PARAMETERS #
    #########################
    
    # available cores for parallel sample construction
    nr_processes = 24
    
    # sample size in unit cells
    W = 512 # must be even
    H = 512
    
    a = 0.24 # lattice constant in nm
    t = 2.8 # hopping value
    e = 0. # onsite potential
    
    # create lattice, hop_dict and pbc_wrap
    lat = graphene.lattice(a)
    hop_dict = graphene.hop_dict_nn(t, e)
    def pbc_wrap(unit_cell_coords, orbital):
        return graphene.pbc(W, H, unit_cell_coords, orbital)
    
    #######################
    # SAMPLE CONSTRUCTION #
    #######################
    
    # create SiteSet object
    site_set = graphene.sheet(W, H)
    
    # make sample
    sample = tipsi.Sample(lat, site_set, pbc_wrap, nr_processes)

    # apply HopDict
    sample.add_hop_dict(hop_dict)
    
    # rescale Hamiltonian
    sample.rescale_H(9.)
    
    #########################
    # SIMULATION PARAMETERS #
    #########################
    
    config = tipsi.Config(sample)
    config.generic['nr_time_steps'] = 1024
    config.generic['nr_random_samples'] = 1
    config.generic['energy_range'] = 20.
    config.generic['correct_spin'] = True
    config.dyn_pol['q_points'] = [[1., 0., 0.]]
    config.DC_conductivity['energy_limits'] = (-0.3, 0.3)
    config.save(directory = 'sim_data', \
                prefix = config.output['timestamp'])
                  
    ############
    # RUN TBPM #
    ############
    
    # get DOS
    corr_DOS = tipsi.corr_DOS(sample, config)
    energies_DOS, DOS = tipsi.analyze_corr_DOS(config, corr_DOS)
    plt.plot(energies_DOS, DOS)
    plt.xlabel("E (eV)")
    plt.ylabel("DOS")
    plt.savefig("graphene_DOS.png")
    plt.close()

    # get AC conductivity
    corr_AC = tipsi.corr_AC(sample, config)
    omegas_AC, AC = tipsi.analyze_corr_AC(config, corr_AC)
    plt.plot(omegas_AC, AC[0])
    plt.xlabel("hbar * omega (eV)")
    plt.ylabel("sigma_xx (sigma_0)")
    plt.savefig("graphene_ACxx.png")
    plt.close()
    
    # get dyn pol
    corr_dyn_pol = tipsi.corr_dyn_pol(sample, config)
    qval, omegas, dyn_pol = tipsi.analyze_corr_dyn_pol(config, corr_dyn_pol)
    qval, omegas, epsilon = tipsi.analyze_corr_dyn_pol(config, dyn_pol)
    plt.plot(omegas, dyn_pol[0,:].imag)
    plt.xlabel("hbar * omega (eV)")
    plt.ylabel("Im(dp)")
    plt.savefig("graphene_dp_imag.png")
    plt.close()
    
    # get DC conductivity
    corr_DOS, corr_DC = tipsi.corr_DC(sample, config)
    energies_DC, DC = tipsi.analyze_corr_DC(config, corr_DOS, corr_DC)
    plt.plot(energies_DC, DC[0, :])
    plt.xlabel("E (eV)")
    plt.ylabel("DC conductivity")
    plt.savefig("graphene_DC.png")
    plt.close()
    
if __name__ == '__main__':
    main()
            
