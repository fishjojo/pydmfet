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
mol.atom = open('Cu.xyz').read()
mol.basis = bas
mol.ecp = ecp
mol.charge = 0
mol.build(max_memory = 4000, verbose=4)

_, mo_energy, mo_coeff, mo_occ, irrep_labels, spins = molden.load("Cu_mo.molden")

#mf = scf.RKS(mol)
mf = rks_ao(mol,smear_sigma = temp)
mf.xc = "pbe,pbe"
mf.max_cycle = 100
#mf.init_guess = "hcore"

mf.mo_energy = mo_energy
mf.mo_occ = mo_occ
mf.mo_coeff = mo_coeff

DMguess = mf.make_rdm1()
#DMguess = None
mf.scf(dm0=DMguess)

'''
with open( 'Cu_mo.molden', 'w' ) as thefile:
    molden.header(mol, thefile)
    molden.orbital_coeff(mol, thefile, mf.mo_coeff,ene=mf.mo_energy,occ=mf.mo_occ)
'''
natoms = mol.natm
impAtom = np.zeros([natoms], dtype=int)
impAtom[:2] = 1


embed = proj_ao.proj_embed(mf,impAtom, Ne_env = 266)
embed.pop_method = 'meta_lowdin'
embed.pm_exponent = 2
embed.make_frozen_orbs(norb = 147)

embed.embedding_potential()
