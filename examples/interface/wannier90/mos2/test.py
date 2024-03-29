#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

from tbplas import wan2pc, gen_kpath


cell = wan2pc("mos2", hop_eng_cutoff=0.0)
k_points = np.array([
    [0.0, 0.0, 0.0],
    [1./3, 1./3, 0.0],
    [1./2, 0.0, 0.0],
    [0.0, 0.0, 0.0],
])
k_path, k_idx = gen_kpath(k_points, [40, 40, 40])
k_len, bands = cell.calc_bands(k_path)


num_bands = bands.shape[1]
for i in range(num_bands):
    plt.plot(k_len, bands[:, i], color="red", linewidth=1.0)
plt.show()
