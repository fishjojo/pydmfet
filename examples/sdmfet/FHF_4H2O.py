from pydmfet import locints,sdmfet,oep,tools
from pyscf import gto, dft
import numpy as np
#import FHF_4H2O_struct

#basis_frag = 'ccpvdz'
#basis_env = 'ccpvdz'
bas = 'ccpvdz'
#e_tot_list = []
for thestructure in range(17,18):

    #mol_frag, mol_env = FHF_4H2O_struct.structure( thestructure, basis_frag, basis_env)
    #mol = gto.mole.conc_mol(mol_frag, mol_env)
    #mol.build()

    mol = gto.Mole()
    mol.atom = open('FHF_4H2O_struct.xyz').read()
    mol.basis = bas
    mol.charge = -1
    mol.build(max_memory = 8000, verbose=4)

    #total system SCF
    mf = dft.RKS(mol)
    mf.xc = 'pbe,pbe'
    mf.scf(dm0=None)

    #orbital localization
    myInts = locints.LocalIntegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )

    natoms = mol.natm
    impAtom = np.zeros([natoms], dtype=int)
    for i in range(3):
        impAtom[i] = 1

    ghost_frag = 1-impAtom
    ghost_env = 1-ghost_frag

    mol_frag = gto.Mole()
    mol_frag.atom = tools.add_ghost(mol.atom, ghost_frag)
    mol_frag.charge = -1
    mol_frag.basis = bas
    mol_frag.build(max_memory = 8000,verbose = 4)

    mol_env = gto.Mole()
    mol_env.atom = tools.add_ghost(mol.atom, ghost_env)
    mol_env.basis =  bas
    mol_env.build(max_memory = 8000,verbose = 4)


    aoslice = mol.aoslice_by_atom()
    impurities = np.zeros([mol.nao_nr()], dtype = int)
    for i in range(natoms):
        if(impAtom[i] == 1):
            impurities[aoslice[i,2]:aoslice[i,3]] = 1


    params = oep.OEPparams(algorithm = 'split', opt_method = 'L-BFGS-B', diffP_tol=1e-4, outer_maxit = 20)
    params.options['maxiter'] = 50
    params.options['ftol']  = 1e-10
    params.options['gtol']  = 1e-4


    Ne_frag = 20
    theDMFET = sdmfet.DMFET(mf, mol_frag, mol_env, myInts, impurities, impAtom, Ne_frag,\
                        boundary_atoms=None, boundary_atoms2=None,\
                        dim_imp =None, dim_bath =None,dim_big=None, oep_params=params, ecw_method = 'hf',mf_method = mf.xc)

    umat = theDMFET.embedding_potential()

