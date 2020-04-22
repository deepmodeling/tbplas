"""black_phosphorus_bands.py

Black phosphorus band structure example for tipsi.
"""

from tipsi.materials import black_phosphorus
import tipsi
import matplotlib.pyplot as plt
import numpy as np


def main():

    # band plot resolution
    res_bands = 25

    # create lattice, hop_dict and pbc_wrap
    lat = black_phosphorus.lattice()
    hop_dict = black_phosphorus.hop_dict()

    # define symmetry points
    G = np.array([0., 0., 0.])
    X = lat.reciprocal_latt()[1] / 2
    Y = lat.reciprocal_latt()[0] / 2
    S = X + Y
    kpoints = [G, Y, S, X, G]
    ticktitles = ["G", "Y", "S", "X", "G"]
    kpoints, kvals, ticks = tipsi.interpolate_k_points(kpoints, res_bands)

    # get band structure
    bands = tipsi.band_structure(hop_dict, lat, kpoints)

    # plot bulk band structure
    for i in range(len(bands[0, :])):
        plt.plot(kvals, bands[:, i], color='k')
    for tick in ticks:
        plt.axvline(tick, color='k', linewidth=0.5)
    plt.xticks(ticks, ticktitles)
    plt.xlim((0., np.amax(kvals)))
    plt.xlabel("k (1/nm)")
    plt.ylabel("E (eV)")
    plt.savefig("bands_bp_bulk.png")
    plt.close()

    # remove z hoppings from hopdict
    hop_dict.remove_z_hoppings()

    # get bands again
    bands = tipsi.band_structure(hop_dict, lat, kpoints)

    # plot single layer band structure
    for i in range(len(bands[0, :])):
        plt.plot(kvals, bands[:, i], color='k')
    for tick in ticks:
        plt.axvline(tick, color='k', linewidth=0.5)
    plt.xticks(ticks, ticktitles)
    plt.xlim((0., np.amax(kvals)))
    plt.xlabel("k (1/nm)")
    plt.ylabel("E (eV)")
    plt.savefig("bands_bp_monolayer.png")
    plt.close()


if __name__ == '__main__':
    main()
