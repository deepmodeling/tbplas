"""graphene_tbpm.py

Graphene TBPM example.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random

import sys
sys.path.append("..")
sys.path.append("../materials")
import tipsi
import graphene

def main():

    #########################
    # SIMULATION PARAMETERS #
    #########################
    
    # available cores for parallel sample construction
    nr_processes = 24
    
    # sample size in unit cells
    W = 1024 # must be even
    H = 1024
    nr_vacancies = int(0.01 * H * W * 2)
    
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
    
    # random vacancies
    for i in range(nr_vacancies):
        x0 = random.randrange(W / 2)
        y0 = random.randrange(H)
        y1 = random.randrange(2)
        orb = random.randrange(2)
        x, y = x0 - y0, x0 + y0
        y = y + y1
        unit_cell_coords = (x, y, 0)
        site_set.delete_site(unit_cell_coords, orb)
    
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
    config.generic['nr_time_steps'] = 2048
    config.generic['nr_random_samples'] = 1
    config.generic['energy_range'] = 20.
    config.generic['correct_spin'] = True
    config.dyn_pol['q_points'] = [[1., 0., 0.]]
    config.DC_conductivity['energy_limits'] = (-0.01, 0.01)
    config.save(directory = 'sim_data', \
                prefix = config.output['timestamp'])
                  
    ############
    # RUN TBPM #
    ############
    
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
            
