from pydmfet import proj_ao
from pydmfet.qcwrap.pyscf_rks_ao import rks_ao
from pyscf import gto
import numpy as np
from pyscf.tools import molden
from pyscf import scf,lo
from pyscf.lo import iao,orth,pipek,boys
from functools import reduce
import math

bas ='ccpvdz'
temp = 0.0


mol = gto.Mole()
mol.atom = open('acid.xyz').read()
mol.basis = bas
mol.charge = 0
mol.build(max_memory = 4000, verbose=4)


mf = scf.RKS(mol)
#mf = rks_ao(mol,smear_sigma = temp)
#mf.xc = "pbe,pbe"
mf.max_cycle = 50

DMguess = None
mf.scf(dm0=DMguess)

'''
embed = proj_ao.proj_embed(mf)
embed.pop_method = 'meta_lowdin'
embed.exponent = 2
embed.make_frozen_orbs(norb = 21)
'''

#mo = boys.Boys(mol).kernel(mf.mo_coeff[:,:21], verbose=4)
#mo = pipek.PM(mol).kernel(mf.mo_coeff[:,:21], verbose=4)
pm = pipek.PM(mol)
pm.pop_method = 'meta_lowdin'
pm.exponent = 2
mo = pm.kernel(mf.mo_coeff[:,:19], verbose=4)

with open( 'mo_lo.molden', 'w' ) as thefile:
    molden.header(mol, thefile)
    molden.orbital_coeff(mol, thefile, mo)
