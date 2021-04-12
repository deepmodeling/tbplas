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
#sys.path.append("./")
from .builder import *
import time
import numpy as np
import scipy.linalg.lapack as spla
from multiprocessing import Pool


def Lindhard(lattice,hop_dict,energy_range,energy_step,q_x,q_y,kmesh,mu,T,back_dielect):

    ########################################### 
    start1 =time.time()
#read the information of lattice and hopping from wannier90 output
    #lattice, hop_dict =  Tipsi.input.read_wannier90('lattice_vectors.win','inse_centres.xyz','inse_hr.dat')
    #attice, hop_dict =  input.read_wannier90('wannier90.win','wannier90_centres.xyz','wannier90_hr.dat')
    print(lattice.vectors)

  ###########################################################################

    "Parameters for w range and bands cutoff or pick "
    #energy_range=10
    #energy_step=2000
    energy_min=-5
    energy_max=5
    band_num_min=1
    band_num_max=6
    energy_res=2048          
    ####################
    
    
    ################################################
    "Parameters for bands,states calculation"
    n_orbitals=len(lattice.orbital_coords)
    lat=lattice
    o_coords=np.array(lattice.orbital_coords) #have to be an array!
    #print("o_coords",o_coords)
    potential=0

    ##########################################################

    
    ######################################
    "Parameters for polarization"
    coord_qx=[]
    coord_qy=[]
    omegas=[i*energy_range / energy_step for i in range(energy_step+1)]

    momenta,momenta_q,s,len_q=coordinates(lattice,q_x,q_y,kmesh)
    '''
    q_num=int(sys.argv[1])
	#print("q_num",q_num)
    file="kpoints.dat"
    with open(file,'r') as f:
        coord_content = f.readlines()
    #print("coord",float(coord_content[0]))
        #print("data",data[0])
        #float(data[0])
        
    for line in coord_content:
        data = line.split()
        coord_qx.append(float(data[0]))
        coord_qy.append(float(data[1]))
    q_x=coord_qx[q_num]
    q_y=coord_qy[q_num]
    len_q=np.sqrt(q_x**2+q_y**2)
    
    coord_in=copy.copy(momenta)
    coord_q=np.array(coord_in)
    for i in range(len(coord_q)):
        coord_q[i,0]=coord_q[i,0]+q_x
        coord_q[i,1]=coord_q[i,1]+q_y
    
    momenta_q=coord_q.tolist()
	 '''
    #np.savetxt("momenta.dat",np.array(momenta))
    #np.savetxt("momenta_q.dat",np.array(momenta_q))

    
    #mu=-3
    #dos_omegas=[i*energy_range / energy_res for i in range(-energy_res,energy_res+1)]
    
    ######################################
    
    ###########
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
    

    
    bands, states= ener_states(momenta,n_orbitals,lat,hop_dict,potential,16)
    bands_q,states_q =ener_states(momenta_q,n_orbitals,lat,hop_dict,potential,16)
    #print(states[0][0][1])    
    #np.save('bands_copy.npy',bands)
    #np.save('states_copy.npy',states)
    #np.save('bands_q_copy.npy',bands_q)
    #np.save('states_q_copy.npy',states_q)
    
    #bands =np.load('bands_copy.npy')
    #states=np.load('states_copy.npy')
    #bands_q=np.load('bands_q_copy.npy')
    #states_q=np.load('states_q_copy.npy')

    '''
    bands, states= eigen(momenta,n_orbitals,lat,hop_dict,potential)
    bands_q,states_q =eigen(momenta_q,n_orbitals,lat,hop_dict,potential)
    ###################################### for bands pick
    '''
    '''
    bands_cut,states_cut=bands_cut_off(bands,states,energy_min,energy_max)
    bands_q_cut,states_q_cut=bands_cut_off(bands_q,states_q,energy_min,energy_max)
    '''
    '''
    bands_cut,states_cut=band_pick(bands,states,band_num_min,band_num_max)
    bands_q_cut,states_q_cut=band_pick(bands_q,states_q,band_num_min,band_num_max)
    print("bands_pick:",bands_cut[0])
    '''
    
    end1=time.time()
    time_band=(end1-start1)/3600
    print("the time for array band:",time_band)
    start2 =time.time()                        
    ########################################################################################################################################
    dyn_cut_phas=dyn_pol_phase(omegas,bands_q,bands,states_q,states,q_x,q_y,o_coords,n_orbitals,beta,mu)
    dyn_cut_phas=prefactor*dyn_cut_phas
    
    #np.savetxt("dyn_mono_-1eV_nocut.dat",np.column_stack((np.array(omegas),dyn_cut_phas.real,dyn_cut_phas.imag)))
    
    epsilon_ph,loss_ph=epsilon_q(len_q,eps_factor,dyn_cut_phas,omegas) 
    epsilon_1l,eps_2l,loss_1l,loss_2l =epsilon_2l(len_q,eps_factor,dyn_cut_phas,omegas,d)
    #np.savetxt("epsilon_2l_real.dat",np.column_stack((np.array(omegas),eps_2l.real)))
    #np.savetxt("epsilon_2l_imag.dat",np.column_stack((np.array(omegas),eps_2l.imag)))
    #np.savetxt("epsilon_1l_real.dat",np.column_stack((np.array(omegas),epsilon_1l.real)))
    #np.savetxt("epsilon_1l_imag.dat",np.column_stack((np.array(omegas),epsilon_1l.imag)))
    
    #np.savetxt("epsilon_para_real_nocut.dat",np.column_stack((np.array(omegas),epsilon_ph.real,epsilon_ph.imag)))
    #np.savetxt("loss_para_nocut.dat",np.column_stack((np.array(omegas),loss_ph)))
    #np.savetxt('omegas.dat',np.array(omegas))
    #np.savetxt('loss_q.dat',loss_ph)
    #np.savetxt('loss_1L.dat',loss_1l)
    #np.savetxt('loss_2L.dat',loss_2l)
    
    ################################################################this part is for the needs of bands cut off

    
    ###########################################################################################################################################
    '''
    # cut or pick
    dyn_cut_phas=dyn_pol_phase(omegas,bands_q_cut,bands_cut,states_q_cut,states_cut,q_x,q_y,o_coords,n_orbitals,beta,mu)
    dyn_cut_phas=prefactor*dyn_cut_phas
    
    np.savetxt("dyn_mono_-1eV.dat",np.column_stack((np.array(omegas),dyn_cut_phas.real,dyn_cut_phas.imag)))
    
    epsilon_ph,loss_ph=epsilon_q(len_q,eps_factor,dyn_cut_phas,omegas)  
    np.savetxt("epsilon_para_real.dat",np.column_stack((np.array(omegas),epsilon_ph.real,epsilon_ph.imag)))
    np.savetxt("loss_para.dat",np.column_stack((np.array(omegas),loss_ph)))
    np.savetxt('omegas.dat',np.array(omegas))
    np.savetxt('loss_q.dat',loss_ph)
    '''
    end2=time.time()
    time_cal=(end2-start2)/3600 
    f=open('running_time_jitparallel_pick.dat',"w+")
    f.write("sample time(/h): calculating time(/h):  \n")
    f.write(" %.5f  %.5f " % (time_band,time_cal))
    f.close()
    
    return omegas,dyn_cut_phas,epsilon_1l,loss_ph

    
    
    '''
    #######for DOS calculation
    s=0.001
    dos=np.zeros(len(dos_omegas))                    #s is the limitation
    for i in range(len(dos_omegas)):
        print("step: \n")
        print(i)
        for j in range(len(momenta)):
            for k in range(n_orbitals):
                dos[i] +=((1/s)*np.sqrt(1/np.pi))*(np.exp(-(dos_omegas[i]-bands[j][k])**2/s)) 
    dos=dos/(np.sum(dos)*energy_range/energy_res)           
    np.array(dos_omegas)
    np.savetxt("dos_gausi.dat",np.column_stack((dos_omegas,dos)))
    f=open("Congratulations.txt","w+")
    f.write("It has been done!")
    f.close()
 
    end2=time.time()
    time_cal=(end2-start2)/3600    #unit-h hours
    f=open('running_time_jitparallel_pick.dat',"w+")
    f.write("sample time(/h): calculating time(/h):  \n")
    f.write(" %.5f  %.5f " % (time_band,time_cal))
    f.close()
    '''
   
    

    
 
'''   
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
'''

def coordinates(lattice,q_x,q_y,mesh=128):
    
    
    kvectors=lattice.reciprocal_latt()
    q=np.sqrt(q_x**2+q_y**2)
    coord1=[]
    b1=np.array(kvectors[0])
    b2=np.array(kvectors[1])
    print("kvectors",b1,b2)

    kpoints=[]
    for i in range(mesh):
        for j in range(mesh):
            kpoints.append(i/mesh*b1+j/mesh*b2)
    
    #np.savetxt("kpoints_128mesh.dat",kpoints)
    
    coord1=kpoints
    
    s=npla.norm(np.cross(b1/mesh,b2/mesh))

    print("resolution",s)
    coord=copy.copy(coord1)   
    
    coord_q=np.array(coord)
    for i in range(len(coord_q)):
        coord_q[i,0]=coord_q[i,0]+q_x
        coord_q[i,1]=coord_q[i,1]+q_y
    
    coord_q=coord_q.tolist()
    
    return coord,coord_q,s,q




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

def eigen(momenta,n_orbitals,lat,hop_dict,potential):
        
    
    
    
    onsite_poten=potential
    momenta=np.array(momenta)
    n_momenta = momenta.shape[0]
    
    bands = np.zeros((n_momenta, n_orbitals))
    states= np.zeros((n_momenta,n_orbitals,n_orbitals),dtype=complex)
    
    
    # iterate over momenta
    for i in range(n_momenta):
        
        
        # fill k-space Hamiltonian
        momentum = momenta[i, :]
        H_K=hop_dict_ft(hop_dict,lat,momentum)
        
        
        # get eigenvalues, store
        eigenvalues, eigenstates, info = spla.zheev(H_K)
        bands[i,:]=eigenvalues[:]+onsite_poten
        states[i]=eigenstates.T
   
    
    

            #States are not been checked
    return bands,states
#############################################################################
###s s'=-1__bands[i][0] or s s'=1__bands_1[i][1]

    


#dyn_pick=dyn_pol(bands_q_pick,bands_pick,states_q_pick,states_pick,half_bands,2,beta,mu)
def dyn_pol(band_q,band,state_q,state,omegas,beta,mu):
    
    dyn_polar=np.zeros(len(omegas),dtype=complex)
    for k in range(len(omegas)): 
        f=open("step_.txt","a")
        f.write("step:")
        f.write("%d" % k)
        f.write( "\n")
        omega=omegas[k]
        for i in range (len(band)):
            for j in range(len(band_q[i])):
                for l in range(len(band[i])):
                    f_q=1.0/(1.0+np.exp(beta*(band_q[i][j]-mu)))
                    f=1.0/(1.0+np.exp(beta*(band[i][l]-mu)))
                    state_q_t=state_q[i][j].T
                    inner_prod=np.vdot(state_q_t,state[i][l])
                    F=np.conjugate(inner_prod)*inner_prod
                    dyn_polar[k] +=F*(f_q-f)/(band_q[i][j]-band[i][l]-omega-0.005j)
    return dyn_polar

            
#dyn_pol_ph=dyn_pol_phase(bands_q,bands,states_q,states,q_x,q_y,o_coords,omegas,n_orbitals,beta,mu)  
@jit(nopython=True,parallel=True)  #up to now best performance
def dyn_pol_phase(omegas,band_q,band,state_q,state,q_x,q_y,o_coords,n_orbitals,beta,mu):
    for i in range(len(band)):
        for j in range(len(band[i])):
            for k in range(n_orbitals):
                '''
                state[i][j][k]=np.exp(1j*(q_x*o_coords[0][0]+q_y*o_coords[0][1]))*state[i][j][0]
                state[i][j][1]=np.exp(1j*(q_x*o_coords[1][0]+q_y*o_coords[1][1]))*state[i][j][1]
                '''
            
                state[i][j][k]=np.exp(1j*(q_x*o_coords[k][0]+q_y*o_coords[k][1]))*state[i][j][k]
    
                
                
                      
    dyn_polar_q=np.zeros(len(omegas),np.complex_)
    n_k=len(band)  
    for k in prange(len(omegas)): 
        #print(k)
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
def ener_states(kpoints,n_orbitals,lat,hop_dict,potential,nr_process):
    
    k_start=time.time()
    kpoints_split= list_split(kpoints,nr_process)
    k_arg=[(i,n_orbitals,lat,hop_dict,potential) for i in kpoints_split]
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
