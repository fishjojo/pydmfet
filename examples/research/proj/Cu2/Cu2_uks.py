from pydmfet import proj_ao
from pydmfet.qcwrap.pyscf_rks_ao import rks_ao
from pyscf import gto,scf
import numpy as np
from pyscf.tools import molden
from pyscf import lo
from pyscf.lo import iao,orth
from functools import reduce
import math

bas ='lanl2dz'
ecp = 'lanl2dz'
temp = 0.005

mol = gto.Mole()
mol.atom = open('Cu2.xyz').read()
mol.basis = bas
mol.ecp = ecp
mol.charge = -2
mol.spin = 2
mol.build(max_memory = 4000, verbose=4)


mf = scf.UKS(mol)
#mf = rks_ao(mol,smear_sigma = temp)
mf.xc = "pbe,pbe"
mf.max_cycle = 50

DMguess = None
mf.scf(dm0=DMguess)
