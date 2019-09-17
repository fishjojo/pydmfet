from pydmfet import locints, sdmfet,oep,tools,dfet_ao
from pydmfet.qcwrap.pyscf_rks_ao import rks_ao
from pyscf import gto, scf, ao2mo
import numpy as np
from pyscf.tools import molden
import time
from pydmfet import dfet_ao
from pydmfet.dfet_ao import dfet
from read_umat import read_umat

bas ='6-31G*'
temp = 0.005


mol = gto.Mole()
mol.atom = open('O2-graphene.xyz').read()
mol.basis = bas
mol.charge = 0
mol.build(max_memory = 24000, verbose=4)

dm_guess=None
_, _, mo_coeff, mo_occ, _, _ = molden.load("MO.molden")
dm_guess = np.dot(mo_coeff*mo_occ, mo_coeff.T)


#mf = scf.UKS(mol)
mf = rks_ao(mol,smear_sigma = temp)
mf.xc = "pbe,pbe"
mf.max_cycle = 50
mf.scf(dm0=dm_guess)

with open( 'MO.molden', 'w' ) as thefile:
    molden.header(mf.mol, thefile)
    molden.orbital_coeff(mf.mol, thefile, mf.mo_coeff,occ = mf.mo_occ)
