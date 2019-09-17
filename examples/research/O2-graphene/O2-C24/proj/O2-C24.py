from pydmfet import proj_ao, tools
from pydmfet.qcwrap.pyscf_rks_ao import rks_ao
from pyscf import gto,scf
import numpy as np
from pyscf.tools import molden

t0 = tools.time0()

bas ='6-31G*'
temp = 0.005


mol = gto.Mole()
mol.atom = open('O2-C24.xyz').read()
mol.basis = bas
mol.charge = 0
mol.build(max_memory = 24000, verbose=4)

dm_guess=None
_, _, mo_coeff, mo_occ, _, _ = molden.load("MO_pbe.molden")
dm_guess = np.dot(mo_coeff*mo_occ, mo_coeff.T)


#mf = scf.UKS(mol)
mf = rks_ao(mol,smear_sigma = temp)
mf.xc = "pbe,pbe"
mf.max_cycle = 50
mf.scf(dm0=dm_guess)

'''
with open( 'MO.molden', 'w' ) as thefile:
    molden.header(mf.mol, thefile)
    molden.orbital_coeff(mf.mol, thefile, mf.mo_coeff,occ = mf.mo_occ, ene = mf.mo_energy)
'''

natoms = mol.natm
impAtom = np.zeros([natoms], dtype=int)
for i in range(8):
    impAtom[i] = 1


embed = proj_ao.proj_embed(mf,impAtom, Ne_env = 110)
embed.pop_method = 'meta_lowdin'
embed.pm_exponent = 2
embed.make_frozen_orbs(norb = 83)
embed.embedding_potential()

