from pydmfet import proj_ao
from pydmfet.qcwrap.pyscf_rks_ao import rks_ao
from pyscf import gto,scf
import numpy as np
from pyscf.tools import molden
from pyscf import lo
from pyscf.lo import iao,orth
from functools import reduce
import math

bas ='ccpvdz'
temp = 0.01

mol = gto.Mole()
mol.atom = open('C3H6.xyz').read()
mol.basis = bas
mol.charge = 0
mol.build(max_memory = 4000, verbose=4)


#mf = scf.RKS(mol)
mf = rks_ao(mol,smear_sigma = temp)
mf.xc = "pbe,pbe"
mf.max_cycle = 50

DMguess = None
mf.scf(dm0=DMguess)


natoms = mol.natm
impAtom = np.zeros([natoms], dtype=int)
for i in range(5):
    impAtom[i] = 1


embed = proj_ao.proj_embed(mf,impAtom, Ne_env = 8)
embed.pop_method = 'meta_lowdin'
embed.make_frozen_orbs(norb = 11)
#embed.embedding_potential()
