#!/usr/bin/env python
from pydmfet import oep, tools
from pydmfet.dfet_ao import dfet
from pydmfet.qcwrap.pyscf_rks_ao import rks_ao
from pydmfet.tools.sym import h_lattice_sym_tab
from pyscf import gto
from pyscf.lib import fingerprint
import numpy as np

def test_dmfet_ao_symm():
    '''H chain with symmetry'''
    temp = 0.
    bas = 'sto-6g'
    nat = 8

    mol = gto.Mole()
    mol.atom = []
    r = 0.74
    for i in range(nat):
        mol.atom.append(('H', (r*i, 0, 0)))

    mol.basis = bas
    mol.build(max_memory = 4000,verbose=4)

    #total system SCF
    mf = rks_ao(mol,smear_sigma = temp)
    mf.xc = 'pbe,pbe'
    mf.max_cycle = 50
    mf.scf(dm0=None)

    #symmetry table (test)
    atm_ind = np.zeros([nat],dtype=int)
    for i in range(nat):
        atm_ind[i] = i
    atm_ind = atm_ind.reshape([1,nat])
    sym_tab = h_lattice_sym_tab(atm_ind)


    impAtom = np.zeros([nat], dtype=int)
    for i in range(nat//2):
        impAtom[i] = 1

    ghost_frag = 1-impAtom
    ghost_env = 1-ghost_frag

    #embedded region mol object
    mol_frag = gto.Mole()
    mol_frag.atom = tools.add_ghost(mol.atom, ghost_frag)
    mol_frag.basis = bas
    mol_frag.build(max_memory = 4000,verbose = 4)

    #environment mol object
    mol_env = gto.Mole()
    mol_env.atom = tools.add_ghost(mol.atom, ghost_env)
    mol_env.basis =  bas
    mol_env.build(max_memory = 4000,verbose = 4)

    #define electron numbers for each subsystem
    Ne_frag = nat//2
    Ne_env = nat//2

    #option to add point charges at boundary
    boundary_atoms = None
    boundary_atoms2 =  None

    #set initial embedding potential
    umat=None

    #parameters control embedding potential optimization
    params = oep.OEPparams(algorithm = 'split', opt_method = 'L-BFGS-B', diffP_tol=1e-4, outer_maxit = 20)
    params.options['ftol'] = 1e-9
    params.options['gtol'] = 1e-4
    params.options['maxiter'] = 50

    #dmfet object
    theDMFET = dfet.DFET(mf, mol_frag, mol_env, Ne_frag, Ne_env,\
                         boundary_atoms=boundary_atoms, boundary_atoms2=boundary_atoms2,umat = umat,\
                         oep_params=params, smear_sigma=temp, ecw_method = 'hf',mf_method = mf.xc, plot_dens=True)
    #use symmetry
    theDMFET.sym_tab = sym_tab

    #embedding potential optimization
    umat = theDMFET.embedding_potential()

    assert abs(fingerprint(umat) - 0.4563881579329332) < 1e-4

if __name__ == "__main__":
    test_dmfet_ao_symm()
