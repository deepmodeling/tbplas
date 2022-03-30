# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 14:34:41 2019

@author: Kuang Xueheng
'A'   is the area of a super unit cell
'momenta_2' is the coordinates of q+k
'momenta_1' is the set of coordinates of k
'bands_2'  is equivalent to E(q+k),the eigenvalues cooresponding to momenta_2
'states_2' are eigenstates of E(q+k)
Likewise,'bands_1' for E(k),'states_1' related to E(k)
'omegas'    is the frequecies from 0 to energy_range
'prefactor' is the factor multiplied to the sum
'g' is the degeneracy of spin,valley,or spin and vally.
"""
import multiprocessing
from multiprocessing import Pool
import copy

import numpy as np
import numpy.linalg as npla
import scipy.linalg.lapack as spla
from numba import jit, prange

import tbplas.builder.core as core
import tbplas.builder.lattice as lattice


class Lindhard():
    def __init__(self,cell,energy_range,energy_step,kmesh,mu,T,back_dielect) -> None:
        self.cell = cell
        self.omegas = [i*energy_range / energy_step for i in range(energy_step+1)]
        self.mesh = kmesh
        self.mu = mu
        self.beta = 11604.505/T
        self.eps_factor = 1.4399644 *2 * np.pi / back_dielect
        self.g_s = 2

    def DP_qpoints(self,qlist):
        n_cpu = multiprocessing.cpu_count()
        dyn_pol = np.zeros((len(qlist),len(self.omegas)),dtype=complex)
        momenta,momenta_cart,s = self.coord_k()
        band,state = self.ener_states(momenta,self.cell,n_cpu)
        for i, qpoint in enumerate(qlist):
            print("calulating q: \n")
            print(qpoint)
            momenta_q ,lenq = self.coord_kplusq(qpoint,momenta_cart)
            band_q,state_q =self.ener_states(momenta_q,self.cell,n_cpu)
            dyn_pol[i,:] = self.DP(band,state,band_q,state_q,s)
        return self.omegas, dyn_pol

    def DP(self,bands,states,bands_q,states_q,s):
        '''
        q : (n,3) array or list
        '''
        ################################################
        n_orbitals= self.cell.num_orb
        "pre_factor for units of espilon and polarization"
        prefactor=(self.g_s*s)/((2*np.pi)**2)
        omegas = self.omegas
        mu = self.mu
        @jit(nopython=True,parallel=True)  #up to now best performance
        def dyn_pol_phase(omegas,band_q,band,state_q,state,n_orbitals,beta,mu):
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
        dyn_cut_phas = dyn_pol_phase(omegas,bands_q,bands,states_q,states,n_orbitals,self.beta,mu)
        dyn_cut_phas = dyn_cut_phas * prefactor
        return dyn_cut_phas

    def coord_kplusq(self,q,coord_k):
        len_q = np.sqrt(q[0]**2 + q[1]**2)
        kvectors=self.cell.get_reciprocal_vectors()
        coord_q=copy.copy(coord_k)
        for i in range(len(coord_q)):
            coord_q[i,0]=coord_q[i,0]+ q[0]
            coord_q[i,1]=coord_q[i,1]+ q[1]
        coord_q_cart = lattice.cart2frac(kvectors,coord_q)
        return coord_q_cart,len_q

    def coord_k(self):
        kvectors=self.cell.get_reciprocal_vectors()
        coord1=[]
        b1=np.array(kvectors[0])
        b2=np.array(kvectors[1])
        kpoints=[]
        for i in range(self.mesh):
            for j in range(self.mesh):
                kpoints.append(i/self.mesh*b1+j/self.mesh*b2)
        coord1=kpoints
        coord_cart = np.array(kpoints)
        coord_frac = lattice.cart2frac(kvectors,coord_cart)
        s=npla.norm(np.cross(b1/self.mesh,b2/self.mesh))
        return coord_frac,coord_cart,s

    def band_pick(self,bands,states,band_idex):    
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


    def band_pick(self,bands,states,band_min,band_max):    
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

    def bands_cut_off(self,bands,states,ener_min,ener_max):
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

    def eigen(self,momenta,cell):
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

    def dyn_pol_phase(self,omegas,band_q,band,state_q,state,n_orbitals,beta,mu):
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

    def epsilon_q(self,q,prefac,dyn,omegas):
        epsilon=np.zeros(len(omegas),dtype=complex)
        for i in range(len(omegas)):
            epsilon[i]=1.0-prefac/q*dyn[i]
        s=-np.imag((np.divide(1,epsilon)))   #s:loss function
        return epsilon,s

    def epsilon_2l(self,q,prefac,dyn,omegas,d):
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
    def ener_states(self,kpoints,cell,nr_process):
        kpoints_split= self.list_split(kpoints,nr_process)
        k_arg=[(i,cell) for i in kpoints_split]
        eigen_values,eigen_states= self.parallel_eigen(k_arg,nr_process)
        return eigen_values,eigen_states

    def polar_parallel(self,omegas,bands_q_cut,bands_cut,states_q_cut,states_cut,q_x,q_y,o_coords,n_orbitals,beta,mu,nr_process):
        w_split=self.list_split(omegas,nr_process)
        w_arg=[(i,bands_q_cut,bands_cut,states_q_cut,states_cut,q_x,q_y,o_coords,n_orbitals,beta,mu)\
            for i in w_split]
        polar_phas=self.parallel_dyn_pol(w_arg,nr_process)
        return polar_phas

    def list_split(self,split_list,n_cpus):
        n_cpu=n_cpus
        a=split_list
        if (len(a) % n_cpu <5):
            n=int(len(a)/n_cpu)
            
        else:
            n=int(len(a)/n_cpu)+1
                                        #cpu
        omegas_n=[a[i:i+n] for i in range(0,len(a) ,n)]
        return omegas_n

    def parallel_eigen(self,list_div,n_cpus):
        p = Pool(n_cpus) # 创建有5个进程的进程池
        results=p.starmap(self.eigen, list_div) # 将f函数的操作给进程池
        energy=[]
        states=[]
        for i in range(len(results)):
                list_energy=results[i][0].tolist()
                
                energy.append(list_energy)
                states.append(results[i][1])
        states_com2=[y for x in states for y in x]
        states_com2=np.array(states_com2)
        ener_com=sum(energy,[])
        ener_com=np.array(ener_com)
        return ener_com,states_com2

    def parallel_dyn_pol(self,list_div,n_cpus):
        p = Pool(n_cpus) # 创建有5个进程的进程池
        results=p.starmap(self.dyn_pol_phase, list_div) # 将f函数的操作给进程池
        dyn_pol=[y for x in results for y in x]
        dyn_pol=np.array(dyn_pol)
        return dyn_pol

    def dict_combination(self,list_dict):
        list_value=[]
        dict_combi=[]
        for i in list_dict:
            for j in range(len(i)):
                list_value.append(list_dict[i][j])
        for j in range(len(list_value)):
            dict_combi.append(np.array(list_value[j]))
            
        dict_combi=np.array(dict_combi)
        return dict_combi
