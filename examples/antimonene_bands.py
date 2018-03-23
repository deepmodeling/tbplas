"""antimonene_bands.py

Antimonene band structure example for tipsi.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("..")
sys.path.append("../materials")
import tipsi
import antimonene

def main():
    
    # band plot resolution
    res_bands = 50
    
    # create lattice, hop_dict and pbc_wrap
    lat_const = 0.411975806
    vert_disp = 0.16455347
    SOC = True
    SOC_lambda = 0.34
    lat = antimonene.lattice(SOC, lat_const, vert_disp)
    hop_dict = antimonene.hop_dict(SOC, SOC_lambda)
    
    # define symmetry points
    G = [0. ,0. ,0. ]
    M = [np.pi / (np.sqrt(3.) * lat_const), np.pi / lat_const, 0.]
    K = [2 * np.pi / (np.sqrt(3.) * lat_const), 2. * np.pi / (3. * lat_const), 0.]
    kpoints = [G, M, K, G]
    ticktitles = ["G","M","K","G"]

    # get band structure
    kpoints, kvals, ticks = tipsi.interpolate_k_points(kpoints, res_bands)
    bands = tipsi.band_structure(hop_dict, lat, kpoints)
    for i in range(len(bands[0,:])):
        plt.plot(kvals, bands[:,i], color='k')
    for tick in ticks:
        plt.axvline(tick, color='k', linewidth=0.5)
    plt.xticks(ticks, ticktitles)
    plt.xlim((0., np.amax(kvals)))
    plt.xlabel("k (1/nm)")
    plt.ylabel("E (eV)")
    plt.savefig("bands.png")
    plt.close()
    
    
if __name__ == '__main__':
    main()
            
