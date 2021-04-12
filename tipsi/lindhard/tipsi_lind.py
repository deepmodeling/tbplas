# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 17:32:11 2021

@author: Tipsi
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy.linalg as npla
import json
import copy
import sys
#from numba import autojit
from numba import jit
from numba import njit,prange
sys.path.append("/project/xhkuang/software/Tipsi/lib/python3.6/site-packages")
import Tipsi
import Tipsi.plas_func
import numpy as np


def main():
    a = 0.246 # lattice constant in nm
    b = 0.142 # carbon-carbon distance in nm
    vectors        = [[0.5*a, 1.5 * b, 0.],
                  [0.5 * a, -1.5 * b, 0.]]
    orbital_coords = [[0., 0., 0.],
                  [a / 2., b/2., 0.]]
    lat = Tipsi.Lattice(vectors, orbital_coords)
    
    #get HopDict
    t =3.0 # hopping constant in eV
    e = 0.0 # on-site potential in eV
    A_0   = [[e, t],
         [t, e]]
    A_nn0 = [[0., 0.],
         [t, 0.]]
    A_nn1 = [[0., t],
         [0., 0.]]
    
    hop_dict = Tipsi.HopDict()
    hop_dict.set((0, 0, 0),  A_0)
    hop_dict.set((1, 0, 0),  A_nn0)
    hop_dict.set((-1, 0, 0), A_nn1)
    hop_dict.set((1, 1, 0),  A_nn0)
    hop_dict.set((-1, -1, 0), A_nn1)
    
    
    #lattice, hop_dict =  Tipsi.input.read_wannier90('lattice_vectors.win','inse_centres.xyz','inse_hr.dat')
    enerange,step=26,2048
                                # 0def qa=0.5
    qx=3.52*np.cos(np.pi/6)
    qy=3.52*np.sin(np.pi/6)
    T=300
    mu=0
    back_dielec=1
    kmesh=128*8
    omegas,dyn,epsilon,loss=Tipsi.plas_func.Lindhard(lat,hop_dict,enerange,step,qx,qy,kmesh,mu,T,back_dielec)
    np.savetxt("dynreal.dat",np.column_stack((np.array(omegas),dyn.real)))
    np.savetxt("dynimag.dat",np.column_stack((np.array(omegas)/t,-dyn.imag)))
    
    w_dyn = np.array(omegas) / t
  
    plt.plot(w_dyn, -dyn.imag[:],label="qa=%s" %(1*0.15))
    plt.xlabel(r'$\hbar * \omega$/t0')
    plt.ylabel(r'Im($\Pi) (eV^{-1}nm^{-2})$')
    plt.xlim((0,5))
    plt.legend()
    plt.savefig("lindhard_im_dyn_pol.png",dpi=300)
    plt.savefig("imag.eps",dpi=300)
    plt.close()


if __name__ == '__main__':
    main()
