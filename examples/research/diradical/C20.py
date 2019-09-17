from pydmfet import locints, sdmfet,oep,tools,dfet_ao,proj_ao
from pydmfet.qcwrap.pyscf_rks_ao import rks_ao
from pyscf import gto, scf, ao2mo
import numpy as np
from pyscf.tools import molden, cubegen
import time
from pydmfet import dfet_ao
from pydmfet.dfet_ao import dfet

bas ='ccpvdz'
temp = 0.005


mol = gto.Mole()
mol.atom = open('C20.xyz').read()
mol.basis = bas
mol.charge = 0
mol.build(max_memory = 4000, verbose=4)


#mf = scf.RHF(mol)
mf = rks_ao(mol,smear_sigma = temp)
mf.xc = "pbe,pbe"
mf.max_cycle = 50

DMguess = None
mf.scf(dm0=DMguess)

natoms = mol.natm
impAtom = np.zeros([natoms], dtype=int)
for i in range(11):
    impAtom[i] = 1


embed = proj_ao.proj_embed(mf,impAtom, Ne_env = 46)
#embed.lo_method = 'boys'
embed.pop_method = 'meta_lowdin'
embed.pm_exponent = 2
embed.make_frozen_orbs(norb = 44)
embed.embedding_potential()

