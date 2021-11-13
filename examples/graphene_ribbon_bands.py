"""graphene_ribbon_bands.py

Graphene ribbon band structure example for tbplas.
"""

import matplotlib.pyplot as plt
import numpy as np

import tbplas
from tbplas.materials import graphene


def main():

    # parameters
    res_bands = 100  # band plot resolution
    W = 64  # sample width in unit cells, must be even
    H = 1  # sample width in unit cells
    a = 0.24  # lattice constant in nm
    t = 2.8  # hopping value
    e = 0.  # onsite potential

    # get lattice, hop_dict and pbc_wrap from materials file
    lat = graphene.lattice(a)
    hop_dict = graphene.hop_dict_nn(t, e)

    def pbc_wrap(uc_coords, orbital):
        return graphene.pbc_zigzag(W, H, uc_coords, orbital)

    # create SiteSet object
    site_set = graphene.sheet_rectangle(W, H)

    # remove dangling bonds to make zigzag edge
    site_set.delete_site((0, 0, 0), 0)
    site_set.delete_site((int(W / 2) - H, int(W / 2) - 1 + H, 0), 1)

    # make sample
    sample = tbplas.Sample(lat, site_set, pbc_wrap)

    # apply HopDict
    sample.add_hop_dict(hop_dict)

    # rescale Hamiltonian
    sample.rescale_H(9.)

    # plot sample
    sample.plot()

    # get ribbon band structure
    N = res_bands
    kpoints = [[0., (i / N) * 2 * np.pi / a - np.pi / a, 0.]
               for i in range(N + 1)]
    kvals = [(i / N) * 2 * np.pi / a for i in range(N + 1)]
    bands = sample.band_structure(kpoints)

    # plot bands
    for i in range(len(bands[0, :])):
        plt.plot(kvals, bands[:, i], color='k')
    plt.xlim((0., np.amax(kvals)))
    plt.xlabel("k (1/nm)")
    plt.ylabel("E (eV)")
    plt.savefig("bands.png")
    plt.close()


if __name__ == '__main__':
    main()
