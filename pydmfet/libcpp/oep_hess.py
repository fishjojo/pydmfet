import os
import numpy as np
from pyscf import lib
from pydmfet import tools
import time
import ctypes

libhess = np.ctypeslib.load_library('libhess', os.path.dirname(__file__))

def oep_hess(jCa, orb_Ea, size, NOrb, NAlpha=None, mo_occ=None, smear=0.0):

    mo_coeff = np.reshape(jCa, (NOrb*NOrb), 'F')
    hess = np.ndarray((size,size),dtype=float, order='F')
    nthread  = lib.num_threads()
    occ_tol = 1e-8

    t0 = (time.clock(),time.time())

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
                              ctypes.c_double(smear), ctypes.c_double(occ_tol))

    #tools.MatPrint(hess,"hess")
    t1 = tools.timer("hessian construction", t0)

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
