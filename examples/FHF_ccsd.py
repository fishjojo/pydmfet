from pydmfet import locints, sdmfet,oep,tools
from pyscf import gto, scf, ao2mo,cc
import numpy as np
from pyscf.tools import molden
import time
import FHF_4H2O_struct

basis_env = 'ccpvdz'
basis_frag = 'ccpvdz'
e_tot_list = []

for thestructure in range(0,20):

    t0 = (time.clock(), time.time())
    #mol = FHF_4H2O_struct.structure( thestructure, thebasis)
    mol_frag, mol_env = FHF_4H2O_struct.structure( thestructure, basis_frag, basis_env)
    mol = gto.mole.conc_mol(mol_frag, mol_env)
    mol.build()

    mf = scf.RHF(mol)
    mf.max_cycle = 100

    DMguess = None
    mf.scf(dm0=DMguess)

    print mf.e_tot

    mycc = cc.CCSD(mf).run()
    et = 0.0
    et = mycc.ccsd_t()
    e_hf = mf.e_tot
    print mycc.e_corr + et

    e_ccsd = e_hf + mycc.e_corr + et
    print "e_tot = ", e_ccsd

