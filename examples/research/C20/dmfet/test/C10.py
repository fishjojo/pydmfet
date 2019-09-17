from pydmfet import locints, sdmfet,oep,tools,dfet_ao,proj_ao
from pydmfet.qcwrap.pyscf_rks_ao import rks_ao
from pyscf import gto, scf, ao2mo
import numpy as np
from pyscf.tools import molden
import time
from pydmfet import dfet_ao
from pydmfet.dfet_ao import dfet

bas ='ccpvdz'
temp = 0.00


mol = gto.Mole()
mol.atom = open('C10.xyz').read()
mol.basis = bas
mol.charge = 0
mol.build(max_memory = 4000, verbose=4)


#mf = scf.RHF(mol)
mf = rks_ao(mol,smear_sigma = temp)
mf.xc = "hf"
mf.max_cycle = 50

DMguess = None
mf.scf(dm0=DMguess)

