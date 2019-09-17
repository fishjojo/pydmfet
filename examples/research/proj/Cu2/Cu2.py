from pydmfet import proj_ao
from pydmfet.qcwrap.pyscf_rks_ao import rks_ao
from pyscf import gto,scf
import numpy as np
from pyscf.tools import molden

bas ='lanl2dz'
ecp = 'lanl2dz'
temp = 0.005

mol = gto.Mole()
mol.atom = open('Cu2.xyz').read()
mol.basis = bas
mol.ecp = ecp
mol.charge = -2
mol.build(max_memory = 4000, verbose=4)


#mf = scf.RKS(mol)
mf = rks_ao(mol,smear_sigma = temp)
mf.xc = "pbe,pbe"
mf.max_cycle = 50

DMguess = None
mf.scf(dm0=DMguess)

natoms = mol.natm
impAtom = np.zeros([natoms], dtype=int)
for i in range(2):
    impAtom[i] = 1


embed = proj_ao.proj_embed(mf,impAtom, Ne_env = 34)
embed.pop_method = 'meta_lowdin'
embed.pm_exponent = 2
embed.make_frozen_orbs(norb = 33)
embed.embedding_potential()
