from pydmfet import locints, sdmfet, oep, tools, dfet_ao, proj_ao
from pydmfet.dfet_ao import dfet
from pydmfet.qcwrap.pyscf_rks_ao import rks_ao
from pyscf import gto, scf,dft, ao2mo,cc
import numpy as np
from pyscf.tools import molden, cubegen
import copy,time
from pydmfet.tools.sym import h_lattice_sym_tab
import time

t0 = (time.clock(), time.time())

bas1 = '6-31G*'
bas2 = 'lanl2dz'
ecp = 'lanl2dz'
temp = 0.005


mol = gto.Mole()
mol.atom = open('PATFIA.xyz').read()
mol.basis = {'N':bas1,'O':bas1, 'C':bas1, 'H':bas1, 'Cu':bas2}
mol.ecp = {'Cu':ecp}
mol.charge = 2
mol.build(max_memory = 16000, verbose=4)

mf = rks_ao(mol,smear_sigma = temp)
mf.xc = 'pbe0'
mf.max_cycle = 100
mf.scf(dm0=None)

