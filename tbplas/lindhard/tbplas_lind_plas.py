"""
Created on Sun Apr 11 17:32:11 2021

@author: Tipsi

"""
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy.linalg as npla
import scipy.linalg.lapack as spla
#from ..builder import gen_lattice_vectors, PrimitiveCell
#import tbplas_func
import tbplas as tb
import numpy as np
#import tw_tmdc_lib
import h5py
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
import sys

def main():
    #######construct unit cell
    t = 2.7
    vectors = tb.gen_lattice_vectors(a = 2.46, b = 2.46,gamma = 60)
    cell = tb.PrimitiveCell(vectors)
    cell.add_orbital([1/3., 1/3.], 0.0)
    cell.add_orbital([2/3., 2/3.], 0.0)
    cell.add_hopping([0,0],0,1,-t)
    cell.add_hopping([1,0],1,0,-t)
    cell.add_hopping([0,1],1,0,-t)
    #########setting parameters
    enerange,step=26,2048
    q = 1/0.142                           
    qx=q*np.cos(np.pi/6)
    qy=q*np.sin(np.pi/6)
    mesh = 1240
    mu = 0.0
    T=300
    back_dielect=1
    #########calulating with lindhard function
    omegas,dyn,epsilon,loss = tb.Lindhard(cell,enerange,step,qx,qy,mesh,mu,T,back_dielect)
    plt.plot(np.array(omegas)/t, -dyn.imag[:],label="qa=%s" % (1*0.15))
    plt.xlabel(r'$\hbar * \omega$/t0')
    plt.ylabel(r'-Im($\Pi) (eV^{-1}nm^{-2})$')
    plt.xlim((0,5))
    plt.legend()
    plt.savefig("lindhard_im_dyn_pol.png",dpi=300)
    plt.close()
    



if __name__ == '__main__':
    main()
