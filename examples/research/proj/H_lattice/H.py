from pydmfet import proj_ao
from pydmfet.qcwrap.pyscf_rks_ao import rks_ao
from pyscf import gto,scf
import numpy as np
from pyscf.tools import molden
from pyscf import lo
from pyscf.lo import iao,orth
from functools import reduce
import math

bas ='sto-6g'
temp = 0.00

mol = gto.Mole()
mol.atom = open('H.xyz').read()
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
#for i in range(6):
#    impAtom[i] = 1
impAtom[5-1] = 1
impAtom[8-1] = 1


embed = proj_ao.proj_embed(mf,impAtom, Ne_env = 10)
#embed.lo_method = 'boys'
embed.pop_method = 'meta_lowdin'
embed.pm_exponent = 2
embed.make_frozen_orbs(norb = 6)
embed.embedding_potential()
