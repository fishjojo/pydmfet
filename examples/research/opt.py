from pydmfet import locints, sdmfet,oep,tools,dfet_ao
from pydmfet.qcwrap.pyscf_rks_ao import rks_ao
from pyscf import gto, scf, ao2mo, geomopt
import numpy as np
from pyscf.tools import molden
import time
from pydmfet import dfet_ao
from pydmfet.dfet_ao import dfet
from read_umat import read_umat

bas ='ccpvdz'
temp = 0.001


mol = gto.Mole()
mol.atom = open('C3H6_0.xyz').read()
mol.basis = bas
mol.charge = 0
mol.build(max_memory = 4000, verbose=4)


mf = scf.RHF(mol)
#mf = rks_ao(mol,smear_sigma = temp)
#mf.xc = "pbe,pbe"
mf.max_cycle = 50

opt1 = geomopt.optimize(mf)
