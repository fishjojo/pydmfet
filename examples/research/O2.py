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
temp = 0.001


mol = gto.Mole()
mol.atom = open('O2.xyz').read()
mol.basis = bas
mol.charge = 0
mol.build(max_memory = 4000, verbose=4)


#mf = scf.UKS(mol)
mf = rks_ao(mol,smear_sigma = temp)
mf.xc = "lda"
mf.verbose = 4
mf.max_cycle = 50
DMguess = None
mf.scf(dm0=DMguess)

with open( 'O2.molden', 'w' ) as thefile:
    molden.header(mf.mol, thefile)
    molden.orbital_coeff(mf.mol, thefile, mf.mo_coeff,occ = mf.mo_occ, ene = mf.mo_energy)

