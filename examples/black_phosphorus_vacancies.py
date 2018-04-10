"""black_phosphorus_vacancies.py

BP with vacancies example.
"""

import sys
sys.path.append("..")

# AC conductivity of black phosphorus
# with vacancies
import tipsi
from tipsi.materials import black_phosphorus
import random
import matplotlib.pyplot as plt

# parameters
W = 1000
H = 1000
n_vacancies = 10
n_layers = 1
nr_processes = 8

# create lattice, hop_dict and pbc_wrap
lat = black_phosphorus.lattice()
hops = black_phosphorus.hop_dict()
def pbc_wrap(unit_cell_coords, orbital):
    return black_phosphorus.pbc(W, H, \
        unit_cell_coords, orbital)

# create SiteSet object
site_set = black_phosphorus.sheet(W, H)

# create vacancies
for i in range(n_vacancies):
    random_site = \
        random.choice(list(site_set.sites.keys()))
    coord = random_site[0:3]
    orbital = random_site[3]
    site_set.delete_site(coord, orbital)

# make sample
sample = tipsi.Sample(lat, site_set, \
    pbc_wrap, nr_processes)

# apply HopDict
sample.add_hop_dict(hops)

# rescale Hamiltonian
sample.rescale_H(8.5)

# config object
config = tipsi.Config(sample)

# get AC conductivity
corr_AC = tipsi.corr_AC(sample, config)
omegas_AC, AC = \
    tipsi.analyze_corr_AC(config, corr_AC)
plt.plot(omegas_AC, AC[0])
plt.plot(omegas_AC, AC[3])
plt.xlabel("hbar * omega (eV)")
plt.ylabel("sigma (sigma_0)")
plt.show()