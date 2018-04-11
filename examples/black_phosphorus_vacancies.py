"""black_phosphorus_vacancies.py

BP with vacancies example.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random

import sys
sys.path.append("..")
import tipsi
from tipsi.materials import black_phosphorus


def main():
    
    # parameters
    W = 100
    H = 100
    n_vacancies = 500
    n_layers = 1
    nr_processes = 24

    # create lattice, hop_dict and pbc_wrap
    lat = black_phosphorus.lattice()
    hops = black_phosphorus.hop_dict()
    def pbc_wrap(unit_cell_coords, orbital):
        return black_phosphorus.pbc(W, H, \
            unit_cell_coords, orbital)

    # create vacancy coordinates
    vacancies = set()
    while len(vacancies) < n_vacancies:
        x = random.randrange(W)
        y = random.randrange(H)
        z = random.randrange(H)
        orb = random.randrange(n_layers)
        vacancies.add(((x, y, z), orb))

    # create SiteSet object
    site_set = tipsi.SiteSet()
    for z in range(n_layers):
        for x in range(W):
            for y in range(H):
                for i in range(4):
                    uc = (x, y, z)
                    if (uc, i) not in vacancies:
                        site_set.add_site(uc, i)

    # make sample
    sample = tipsi.Sample(lat, site_set, \
        pbc_wrap, nr_processes)

    # apply HopDict
    sample.add_hop_dict(hops)

    # rescale Hamiltonian
    sample.rescale_H(8.5)

    # config object
    config = tipsi.Config(sample)
    config.save(directory = 'sim_data', \
        prefix = config.output['timestamp'])

    # get AC conductivity
    corr_AC = tipsi.corr_AC(sample, config)
    omegas_AC, AC = \
        tipsi.analyze_corr_AC(config, corr_AC)
    plt.plot(omegas_AC, AC[0])
    plt.plot(omegas_AC, AC[3])
    plt.xlabel("hbar * omega (eV)")
    plt.ylabel("sigma (sigma_0)")
    plt.xlim((0., 10.))
    plt.ylim((0., 3.))
    plt.savefig("ac_bp_vacancies.png")
    plt.close()

if __name__ == '__main__':
    main()
