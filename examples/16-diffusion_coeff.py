#! /usr/bin/env python

import random
import numpy as np
import tbplas as tb


def main():
    # Calculate diffusion coefficient for graphene with 5% vacancies

    # Parameters
    width = 120
    height = 120
    num_vac = int(width * height * 0.05)
    
    # Set random vacancies
    vac = []
    for i in range(num_vac):
        x = random.randrange(width)
        y = random.randrange(height)
        orb = random.randrange(2)
        vac.append((x, y, 0, orb))
    vac = np.array(vac)    
       
    # Build sample
    prim_cell = tb.make_graphene_diamond()
    super_cell = tb.SuperCell(prim_cell, dim=(width, height, 1), pbc=(True, True, False))
    super_cell.unlock()      
    super_cell.set_vacancies(vacancies=vac)    
    sample = tb.Sample(super_cell)
    sample.rescale_ham(9.0)
    
    # Configuration
    config = tb.Config()
    config.generic['nr_random_samples'] = 2
    config.generic['nr_time_steps'] = 1024
    config.DC_conductivity['energy_limits'] = (-0.1, 0.1)
    solver = tb.Solver(sample, config)
    analyzer = tb.Analyzer(sample, config)
    
    # Get correlation function
    corr_dos, corr_dc = solver.calc_corr_dc_cond()

    # DC conductivity (xx and yy)
    energies_dc, dc = analyzer.calc_dc_cond(corr_dos, corr_dc)

    # Diffusion coefficient (xx and yy)
    time, diff = analyzer.calc_diff_coeff(corr_dc)

    # output
    with open('DCxx.dat', 'w+') as f:
        f.write('energies:    DCxx:   \n')
        f.write('\n')
        for i in range(len(energies_dc)):
            f.write("%.10f %.10f\n" % (energies_dc[i], dc[0][i]))
        f.write('\n')

    with open('DCyy.dat', 'w+') as f:
        f.write('energies:    DCyy:   \n')
        f.write('\n')
        for i in range(len(energies_dc)):
            f.write("%.10f %.10f\n" % (energies_dc[i], dc[1][i]))
        f.write('\n')

    for j in range(diff.shape[1]):
        with open('diffxx' + str(j) + '.dat', 'w+') as f:
            f.write("Energy: %.10f \n" % (energies_dc[j],))
            f.write('time:    diffxx:   \n')
            f.write('\n')
            for k in range(len(time)):
                f.write("%.10f %.10f \n" % (time[k], diff[0][j][k]))
            f.write('\n')

        with open('diffyy' + str(j) + '.dat', 'w+') as f:
            f.write("Energy: %.10f \n" % (energies_dc[j],))
            f.write('time:    diffyy:   \n')
            f.write('\n')
            for k in range(len(time)):
                f.write("%.10f %.10f \n" % (time[k], diff[1][j][k]))
            f.write('\n')


if __name__ == '__main__':
    main()
    
    
