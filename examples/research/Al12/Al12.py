from pydmfet import locints, sdmfet, oep, tools, dfet_ao, proj_ao
from pydmfet.dfet_ao import dfet
from pydmfet.qcwrap.pyscf_rks_ao import rks_ao
from pyscf import gto, scf,dft, ao2mo,cc
import numpy as np
from pyscf.tools import molden, cubegen
import copy,time
from pydmfet.tools.sym import h_lattice_sym_tab

bas = 'stuttgartdz'
ecp = 'stuttgartdz'
temp = 0.005


mol = gto.Mole()
mol.atom = open('Al12.xyz').read()
mol.basis = bas
mol.ecp = ecp
mol.charge = 0
mol.build(max_memory = 4000, verbose=4)

mf = rks_ao(mol,smear_sigma = temp)
mf.xc = 'pbe,pbe'
mf.max_cycle = 100
mf.scf(dm0=None)


natoms = mol.natm
impAtom = np.zeros([natoms], dtype=int)
for i in range(6):
    impAtom[i] = 1


embed = proj_ao.proj_embed(mf,impAtom, Ne_env = 12)
#embed.lo_method = 'boys'
embed.pop_method = 'iao'
embed.pm_exponent = 4
embed.make_frozen_orbs(norb = 12)
embed.embedding_potential()

