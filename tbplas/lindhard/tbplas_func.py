# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 14:34:41 2019

@author: Kuang Xueheng
"""
'''
'A'   is the area of a super unit cell 
'momenta_2' is the coordinates of q+k
'momenta_1' is the set of coordinates of k
'bands_2'  is equivalent to E(q+k),the eigenvalues cooresponding to momenta_2
'states_2' are eigenstates of E(q+k)
Likewise,'bands_1' for E(k),'states_1' related to E(k)
'omegas'    is the frequecies from 0 to energy_range
'prefactor' is the factor multiplied to the sum
'g' is the degeneracy of spin,valley,or spin and vally.


'''
import multiprocessing
import matplotlib
import matplotlib.pyplot as plt
import numpy.linalg as npla
import json
import copy
import sys
#from numba import autojit
from numba import jit
from numba import njit,prange
#sys.path.append("./")
#from .builder import *
from ..builder import core,lattice
import time
import numpy as np
import scipy.linalg.lapack as spla
from multiprocessing import Pool


def Lindhard(cell,energy_range,energy_step,q_x,q_y,kmesh,mu,T,back_dielect):
    
    
    ################################################
    n_cpu = multiprocessing.cpu_count()
    "Parameters for bands,states calculation"
    n_orbitals= cell.num_orb
    o_coords=np.array(cell.orb_pos_nm) #have to be an array!
    print("o_coords",o_coords)
    potential=0
    omegas=[i*energy_range / energy_step for i in range(energy_step+1)]
    momenta,momenta_q,s,len_q=coordinates(cell,q_x,q_y,kmesh)
    "pre_factor for units of espilon and polarization"
    g_s=2
    prefactor=(g_s*s)/((2*np.pi)**2)
    #back_dielect=1.0                                                                              #integration parameters and prefactores
    eps_pre=1.4399644 *2 * np.pi   #ev*nm *2 * pi
    eps_factor=eps_pre/back_dielect    
    d=0.29459
    beta=11604.505/T
    ###############################################
    "Out put"
    #print("k,k_q",momenta[q_num],momenta_q[q_num])
    print(prefactor)
    #print("q:",len_q)
    print("q_x_q_y:\n")
    print(q_x,q_y)
    print("number of kpoints: \n")
    print(len(momenta))
    

    
    bands, states= ener_states(momenta,cell,n_cpu)
    bands_q,states_q =ener_states(momenta_q,cell,n_cpu)
    print("states_check",momenta[2],states[2][0],momenta[3],states[3][0])
    ########################################################################################################################################
    dyn_cut_phas=dyn_pol_phase(omegas,bands_q,bands,states_q,states,q_x,q_y,o_coords,n_orbitals,beta,mu)
    dyn_cut_phas=prefactor*dyn_cut_phas
    epsilon_ph,loss_ph=epsilon_q(len_q,eps_factor,dyn_cut_phas,omegas) 
    
    return omegas,dyn_cut_phas,epsilon_ph,loss_ph
   
def band_pick(bands,states,band_idex):    
    
    
                 #remain2*2 bands
    bands_pick=np.zeros((len(bands),band_idex*2))
    n_a=len(bands[0])
    states_pick={}
    states_cut=[]
    for k in range(len(bands)):
            bands_pick[k,:]=bands[k][int(n_a/2)-band_idex:int(n_a/2)+band_idex]
            states_pick[k]=states[k][int(n_a/2)-band_idex:int(n_a/2)+band_idex]
            states_cut.append(states_pick[k])
    states_cut=np.array(states_cut)
    
    return bands_pick, states_cut               #states_pick:{0,array[energy,orbitals],1,array[energ,orbitals],....}


def coordinates(cell,q_x,q_y,mesh=128):
    
    
    kvectors=cell.get_reciprocal_vectors()
    q=np.sqrt(q_x**2+q_y**2)
    coord1=[]
    b1=np.array(kvectors[0])
    b2=np.array(kvectors[1])

    kpoints=[]
    for i in range(mesh):
        for j in range(mesh):
            kpoints.append(i/mesh*b1+j/mesh*b2)
    

    
    coord1=kpoints
    coord1 = np.array(kpoints)
    s=npla.norm(np.cross(b1/mesh,b2/mesh))

    coord=copy.copy(coord1)   
    
    coord_q=np.array(coord)
    for i in range(len(coord_q)):
        coord_q[i,0]=coord_q[i,0]+q_x
        coord_q[i,1]=coord_q[i,1]+q_y
    
   
    
    coord_cart = lattice.cart2frac(kvectors,coord)
    coord_q_cart = lattice.cart2frac(kvectors,coord_q)


    return coord_cart,coord_q_cart,s,q




def band_pick(bands,states,band_min,band_max):    
    
    
                 #remain2*2 bands
    bands_pick=np.zeros((len(bands),band_max-band_min+1))
    states_pick={}
    states_cut=[]
    for k in range(len(bands)):
            bands_pick[k,:]=bands[k][band_min-1:band_max]
            states_pick[k]=states[k][band_min-1:band_max]
            states_cut.append(states_pick[k])
    states_cut=np.array(states_cut)
    
    return bands_pick, states_cut 

  

def bands_cut_off(bands,states,ener_min,ener_max):
    
    bands_pick=copy.deepcopy(bands)
    states_pick=copy.deepcopy(states)
    
    delta_energy=0.0
    for i in range(len(bands_pick)):
        for j in range(len(bands_pick[i])): 
            ener= bands_pick[i][j]
            if (ener > ener_max+delta_energy or ener<ener_min-delta_energy):
                 #bands_pick[i][j]=0
                 for k in prange(len(states_pick[i][j])):
                     states_pick[i][j][k]=0+0.j
                     
                     
    return bands_pick ,states_pick  

def eigen(momenta,cell):
    cell.sync_array()
    momenta=np.array(momenta)
    n_momenta = momenta.shape[0]
    n_orbitals = cell.num_orb
    bands = np.zeros((n_momenta, n_orbitals))
    states= np.zeros((n_momenta,n_orbitals,n_orbitals),dtype=complex)
    ham_k = np.zeros((cell.num_orb, cell.num_orb), dtype=np.complex128)
    
    # iterate over momenta
    for i in range(n_momenta):
        # fill k-space Hamiltonian
        momentum = momenta[i, :]
        #H_K= hop_dict_ft(hop_dict,lat,momentum)
        ham_k *= 0.0
        core.set_ham(cell.orb_pos, cell.orb_eng,
                        cell.hop_ind, cell.hop_eng,
                        momentum, ham_k)
        
        # get eigenvalues, store
        eigenvalues, eigenstates, info = spla.zheev(ham_k)
        bands[i,:]=eigenvalues[:]
        states[i]=eigenstates.T

    return bands,states

@jit(nopython=True,parallel=True)  #up to now best performance
def dyn_pol_phase(omegas,band_q,band,state_q,state,q_x,q_y,o_coords,n_orbitals,beta,mu):

    dyn_polar_q=np.zeros(len(omegas),np.complex_)
    n_k=len(band)  
    for k in prange(len(omegas)): 
        omega=omegas[k]
        for i in prange(n_k):
            for j in range(n_orbitals):
                for l in range(n_orbitals):
                    f_q=1.0/(1.0+np.exp(beta*(band_q[i][j]-mu)))
                    f=1.0/(1.0+np.exp(beta*(band[i][l]-mu)))
                
                    inner_prod=np.vdot(state_q[i][j],state[i][l])
                    F=np.abs(inner_prod)**2
                    
                    
                    dyn_polar_q[k] +=F*(f_q-f)/(band_q[i][j]-band[i][l]-omega-0.005j)
    return dyn_polar_q                    

                
def epsilon_q(q,prefac,dyn,omegas):
    epsilon=np.zeros(len(omegas),dtype=complex)
    for i in range(len(omegas)):
        epsilon[i]=1.0-prefac/q*dyn[i]
    
    s=-np.imag((np.divide(1,epsilon)))   #s:loss function 
    
    return epsilon,s 

def epsilon_2l(q,prefac,dyn,omegas,d):
    F=np.exp(-q*d)
    #epsilon=np.zeros(len(omegas),dtype=complex)
    eps_1l=np.zeros(len(omegas),dtype=complex)
    #eps_im=np.zeros(len(omegas))
    eps_2l=np.zeros(len(omegas),dtype=complex)
    #eps_im_2l=np.zeros(len(omegas))
   
    #eps_matri_im=np.zeros((2,2))
    for i in range(len(omegas)):
        eps_matri=np.zeros((2,2),dtype=complex)
        eps_1l[i]=1.0-prefac/q*dyn[i]
        eps_matri[0][0]=eps_1l[i]
        eps_matri[1][1]=eps_1l[i]
        eps_matri[0][1]=-prefac/q*dyn[i]*F
        eps_matri[1][0]=-prefac/q*dyn[i]*F

        eps_2l[i]=np.linalg.det(eps_matri)

    s1=-np.imag((np.divide(1,eps_1l)))   #s:loss function
    s2=-np.imag((np.divide(1,eps_2l)))
    
    return eps_1l,eps_2l,s1,s2
################################################################################################
#code for parallel
def ener_states(kpoints,cell,nr_process):
    
    k_start=time.time()
    kpoints_split= list_split(kpoints,nr_process)
    k_arg=[(i,cell) for i in kpoints_split]
    k_end=time.time()
    k_time=(k_end-k_start)/3600
    print("k points split time:",k_time)
    
    eigen_values,eigen_states= parallel_eigen(k_arg,nr_process)
    
    
    return eigen_values,eigen_states

def polar_parallel(omegas,bands_q_cut,bands_cut,states_q_cut,states_cut,q_x,q_y,o_coords,n_orbitals,beta,mu,nr_process):
    
    
    w_split=list_split(omegas,nr_process)
    w_arg=[(i,bands_q_cut,bands_cut,states_q_cut,states_cut,q_x,q_y,o_coords,n_orbitals,beta,mu)\
           for i in w_split]
    polar_phas=parallel_dyn_pol(w_arg,nr_process)
    
    return polar_phas

def list_split(split_list,n_cpus):
    
    n_cpu=n_cpus
    a=split_list
    if (len(a) % n_cpu <5):
        n=int(len(a)/n_cpu)
        
    else:
        n=int(len(a)/n_cpu)+1
    
                                     #cpu 
    omegas_n=[a[i:i+n] for i in range(0,len(a) ,n)]
    
    return omegas_n
    
        

def parallel_eigen(list_div,n_cpus):
    p = Pool(n_cpus) # 创建有5个进程的进程池
    
    results=p.starmap(eigen, list_div) # 将f函数的操作给进程池   
    energy=[]
    states=[]
    
    comb_start=time.time()
    for i in range(len(results)):
            list_energy=results[i][0].tolist()
            
            energy.append(list_energy)
            states.append(results[i][1])
    
    
   
    states_com2=[y for x in states for y in x]
    states_com2=np.array(states_com2)
    
    
    ener_com=sum(energy,[])
    ener_com=np.array(ener_com)
    #tates_com=dict_combination(states)                       #here is for combining states of different kpoints sets. 
    comb_end=time.time()
    
    comb_time=(comb_end-comb_start)/3600
    print("combination time:",comb_time)
   
    return ener_com,states_com2

def parallel_dyn_pol(list_div,n_cpus):
    p = Pool(n_cpus) # 创建有5个进程的进程池
    
    results=p.starmap(dyn_pol_phase, list_div) # 将f函数的操作给进程池
    dyn_pol=[y for x in results for y in x]
    dyn_pol=np.array(dyn_pol)
    return dyn_pol

def dict_combination(list_dict):

    
    
    list_value=[]
    dict_combi=[]
    for i in list_dict:
        for j in range(len(i)):
            list_value.append(list_dict[i][j])
    for j in range(len(list_value)):
        dict_combi.append(np.array(list_value[j]))
        
    dict_combi=np.array(dict_combi)
    print("dict_combi:",dict_combi)
    print("len_combi_1",len(dict_combi))
    return dict_combi


        
if __name__ == '__main__':
    Lindhard()
