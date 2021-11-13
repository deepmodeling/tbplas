"""graphene_vacancies.py

Graphene with vacancies example.
"""

import matplotlib.pyplot as plt
import numpy as np
import random

import tbplas
from tbplas.materials import graphene


def main():

    # parameters
    W = 200
    H = 200
    n_vacancies = 1000
    nr_processes = 8

    # create lattice, hop_dict and pbc_wrap
    lat = graphene.lattice()
    hops = graphene.hop_dict_nn()

    def pbc_wrap(unit_cell_coords, orbital):
        return graphene.pbc(W, H, unit_cell_coords, orbital)

    # create vacancy coordinates
    vacancies = set()
    while len(vacancies) < n_vacancies:
        x = random.randrange(W)
        y = random.randrange(H)
        orb = random.randrange(2)
        vacancies.add(((x, y, 0), orb))

    # create SiteSet object
    site_set = tbplas.SiteSet()
    for x in range(W):
        for y in range(H):
            for i in range(2):
                uc = (x, y, 0)
                if (uc, i) not in vacancies:
                    site_set.add_site(uc, i)

    # make sample
    sample = tbplas.Sample(lat, site_set, pbc_wrap, nr_processes)

    # apply HopDict
    sample.add_hop_dict(hops)

    # rescale Hamiltonian
    sample.rescale_H(9.)

    # config object
    config = tbplas.Config(sample)
    config.generic['correct_spin'] = True
    config.save()

    # get DOS
    corr_DOS = tbplas.corr_DOS(sample, config)
    energies_DOS, DOS = tbplas.analyze_corr_DOS(config, corr_DOS)
    plt.plot(energies_DOS, DOS)
    plt.xlabel("E (eV)")
    plt.ylabel("DOS")
    plt.savefig("graphene_DOS_vacancies.png")
    plt.close()


if __name__ == '__main__':
    main()
