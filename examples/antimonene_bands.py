"""antimonene_bands.py

Antimonene band structure example for tipsi.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("..")
import tipsi
from tipsi.materials import antimonene

def main():
    
    # parameters
    res_bands = 50           # band plot resolution
    lat_const = 0.411975806  # lattice constant

    # create lattice, hop_dict
    lat = antimonene.lattice(a = lat_const)
    hops = antimonene.hop_dict()

    # define momenta
    k1 = np.pi / (np.sqrt(3.) * lat_const)
    k2 = np.pi / lat_const
    G = [0. ,0. ,0. ]
    M = [k1, k2, 0.]
    K = [2 * k1, (2. / 3.) * k2, 0.]
    kpoints = [G, M, K, G]
    ticktitles = ["G","M","K","G"]
    kpoints, kvals, ticks = tipsi.interpolate_k_points(kpoints, res_bands)

    # get band structure
    bands = tipsi.band_structure(hops, lat, kpoints)

    # plot
    for band in bands.swapaxes(0, 1):
        plt.plot(kvals, band, color = 'k')
    for tick in ticks:
        plt.axvline(tick, color = 'k', linewidth = 0.5)
    plt.xticks(ticks, ticktitles)
    plt.xlim((0., np.amax(kvals)))
    plt.xlabel("k (1/nm)")
    plt.ylabel("E (eV)")
    plt.show()
    
    
if __name__ == '__main__':
    main()
            
