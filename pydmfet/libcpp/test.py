import os
import ctypes
import numpy as np
from pydmfet import tools
from pydmfet import locints
import time

import pyscf
from pyscf import scf
from pyscf import gto
from pyscf import ao2mo
from pyscf import lo
from pyscf.lo import nao, orth
from pyscf.tools import molden, localizer
from pydmfet.locints import iao_helper
import copy




libhess = np.ctypeslib.load_library('libhess', os.path.dirname(__file__))

def calc_hess(jCa,orb_Ea,size,NOcc,NOrb):

    mo_coeff = np.reshape(jCa, (NOrb,NOrb),'F')
    hess = np.ndarray((size*size),dtype=float)

    nthread = 16
    t0 = (time.clock(),time.time())
    libhess.calc_hess_dm_fast(hess.ctypes.data_as(ctypes.c_void_p), \
                              mo_coeff.ctypes.data_as(ctypes.c_void_p), orb_Ea.ctypes.data_as(ctypes.c_void_p), \
                              ctypes.c_int(size), ctypes.c_int(NOrb), ctypes.c_int(NOcc), ctypes.c_int(nthread))


    t1 = tools.timer("hessian", t0)



if __name__ == "__main__":

    nbas = 107
    jCa =  np.random.rand(nbas*nbas)
    orb_Ea = np.random.rand(nbas)
    size = nbas*(nbas+1)//2
    NOcc = 16

    calc_hess(jCa,orb_Ea,size,NOcc,nbas) 
