from pydmfet import proj_ao
from pydmfet.qcwrap.pyscf_rks_ao import rks_ao
from pyscf import gto,scf
import numpy as np
from pyscf.tools import molden
from pyscf import lo
from pyscf.lo import iao,orth
from functools import reduce
import math

bas ='stuttgartdz'
ecp = 'stuttgartdz'
temp = 0.005

mol = gto.Mole()
mol.atom = open('CBVN4.xyz').read()
mol.basis = {'H':'ccpvdz', 'B':bas, 'C':bas, 'N':bas}
mol.ecp = ecp
mol.build(max_memory = 24000, verbose=4)


#mf = scf.RKS(mol)
mf = rks_ao(mol,smear_sigma = temp)
mf.xc = "pbe,pbe"
mf.max_cycle = 100

DMguess = None
mf.scf(dm0=DMguess)

