import os
import numpy as np
from pyscf import lib
from pydmfet import tools
import time
import ctypes

libhess = np.ctypeslib.load_library('libhess', os.path.dirname(__file__))

def symmtrize_hess(hess, sym_tab, size):

    dim = sym_tab.shape[0]

    ind_dict = dict()
    for i in range(dim):
        for j in range(i,dim):
            ind = sym_tab[i,j]
            ind2 = (i,j,)
            if not (ind in ind_dict):
                ind_dict.update({ind:[ind2]})
            else:
                ind_dict[ind].append(ind2)

    nelem = len(ind_dict)
    res = np.zeros([size,nelem], dtype=float)
    for ind, ind2 in ind_dict.items():
        for i,value in enumerate(ind2):
            mu = value[0]
            nu = value[1]
            ioff = (2*dim-mu+1)*mu//2 + nu-mu
            res[:,ind] += hess[:,ioff]

    res_small = np.zeros([nelem,nelem], dtype=float)
    for ind, ind2 in ind_dict.items():
        for i,value in enumerate(ind2):
            mu = value[0]
            nu = value[1]
            ioff = (2*dim-mu+1)*mu//2 + nu-mu
            res_small[ind,:] = res[ioff,:]
            break

    return res_small


def oep_hess(jCa, orb_Ea, size, NOrb, NAlpha=None, mo_occ=None, smear=0.0, sym_tab=None):

    mo_coeff = np.reshape(jCa, (NOrb*NOrb), 'F')
    hess = np.ndarray((size,size),dtype=float, order='F')
    nthread  = lib.num_threads()
    e_tol = 1e-4
    occ_tol = 1e-8 #this should be small enough?

    t0 = tools.time0()

    if smear < 1e-8:
        if NAlpha is None: 
            raise ValueError("NAlpha has to be set")

        libhess.calc_hess_dm_fast(hess.ctypes.data_as(ctypes.c_void_p), \
                              mo_coeff.ctypes.data_as(ctypes.c_void_p), orb_Ea.ctypes.data_as(ctypes.c_void_p), \
                              ctypes.c_int(size), ctypes.c_int(NOrb), ctypes.c_int(NAlpha), ctypes.c_int(nthread))
    else:
        if mo_occ is None:
            raise ValueError("mo_occ has to be set")

        libhess.calc_hess_dm_fast_frac(hess.ctypes.data_as(ctypes.c_void_p), \
                              mo_coeff.ctypes.data_as(ctypes.c_void_p), orb_Ea.ctypes.data_as(ctypes.c_void_p), \
                              mo_occ.ctypes.data_as(ctypes.c_void_p),\
                              ctypes.c_int(size), ctypes.c_int(NOrb), ctypes.c_int(nthread),\
                              ctypes.c_double(smear), ctypes.c_double(e_tol), ctypes.c_double(occ_tol))

    #tools.MatPrint(hess,"hess")
    t1 = tools.timer("hessian construction", t0)

    #if sym_tab is not None:
    #    return symmtrize_hess(hess,sym_tab,size)

    return hess


def oep_hess_old(jCa,orb_Ea,size,NOrb,NOcc):

    mo_coeff = np.reshape(jCa, (NOrb*NOrb), 'F')
    hess = np.ndarray((size,size),dtype=float, order='F')
    nthread  = lib.num_threads()

    t0 = (time.clock(),time.time())
    libhess.calc_hess_dm_fast(hess.ctypes.data_as(ctypes.c_void_p), \
                              mo_coeff.ctypes.data_as(ctypes.c_void_p), orb_Ea.ctypes.data_as(ctypes.c_void_p), \
                              ctypes.c_int(size), ctypes.c_int(NOrb), ctypes.c_int(NOcc), ctypes.c_int(nthread))

    t1 = tools.timer("hessian construction", t0)

    return hess
