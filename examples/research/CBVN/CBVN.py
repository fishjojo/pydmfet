from pydmfet import locints, sdmfet,oep,tools,dfet_ao
from pydmfet.qcwrap.pyscf_rks_ao import rks_ao
from pyscf import gto, scf, ao2mo, geomopt
import numpy as np
from pyscf.tools import molden
import time
from pydmfet import dfet_ao
from pydmfet.dfet_ao import dfet

bas ='stuttgartdz'
ecp = 'stuttgartdz'
temp = 0.005


mol = gto.Mole()
mol.atom = open('CBVN3.xyz').read()
mol.basis = {'H':'ccpvdz', 'B':bas, 'C':bas, 'N':bas}
mol.ecp = ecp
mol.charge = 0
mol.build(max_memory = 24000, verbose=4)


mf = scf.RHF(mol)
#mf = rks_ao(mol,smear_sigma = temp)
#mf.xc = "pbe,pbe"
mf.max_cycle = 50

#DMguess = None
#mf.scf(dm0=DMguess)

opt1 = geomopt.optimize(mf)
