from __future__ import print_function
from pydmfet import locints, sdmfet,oep,tools
from pyscf import gto, scf,dft, ao2mo
import numpy as np
from pyscf.tools import molden
import FHF_4H2O_struct



basis_frag = 'ccpvdz'
basis_env = 'ccpvdz'
bas = 'ccpvdz'
e_tot_list = []
for thestructure in range(17,18):

    #mol_frag, mol_env = FHF_4H2O_struct.structure( thestructure, basis_frag, basis_env)
    #mol = gto.mole.conc_mol(mol_frag, mol_env)
    #mol.build()

    mol = gto.Mole()
    mol.atom = open('FHF_4H2O_struct.xyz').read()
    mol.basis = bas
    mol.charge = -1
    mol.build(max_memory = 24000, verbose=4)


    #total system HF
    #mf = scf.RHF(mol)
    mf = dft.RKS(mol)
    mf.xc = 'pbe,pbe'
    mf.max_cycle = 100
    mf.verbose = 3
    DMguess = None
    mf.scf(dm0=DMguess)
    e_mf = mf.e_tot
    print ("e_mf = ", e_mf)

    myInts = locints.LocalIntegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )
    #myInts.loc_molden( 'loc.molden' )
    myInts.TI_OK = False

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
    mol_frag.build(max_memory = 24000,verbose = 4)

    mol_env = gto.Mole()
    mol_env.atom = tools.add_ghost(mol.atom, ghost_env)
    mol_env.basis =  bas
    mol_env.build(max_memory = 24000,verbose = 4)



    aoslice = mol.aoslice_by_atom()
    impurities = np.zeros([mol.nao_nr()], dtype = int)
    for i in range(natoms):
        if(impAtom[i] == 1):
            impurities[aoslice[i,2]:aoslice[i,3]] = 1
#    for i in range(mol.nao_nr()):
#	impurities[i] = 1

#    for i in range(natoms):
#        print aoslice[i,2],aoslice[i,3]


    Ne_frag = 20
    boundary_atoms = np.zeros([natoms], dtype=int)
    #boundary_atoms[5] = 1
    #boundary_atoms[8]=1
    #boundary_atoms[9]=1
    #boundary_atoms[12]=1
    boundary_atoms =  None

    nbas = mol.nao_nr()
    params = oep.OEPparams(algorithm = 'split', opt_method = 'trust-ncg', \
                        ftol = 1e-8, gtol = 1e-4,diffP_tol=1e-4, outer_maxit = 20, maxit = 100,\
			l2_lambda = 0.0, oep_print = 0, svd_thresh=1e-6)
    theDMFET = sdmfet.DMFET(mf, mol_frag, mol_env, myInts, impurities, impAtom, Ne_frag,\
                        boundary_atoms=boundary_atoms, boundary_atoms2=None,\
                        dim_imp =None, dim_bath =None,dim_big=None, oep_params=params, ecw_method = 'ccsd',mf_method = mf.xc)

    umat = theDMFET.embedding_potential()
    exit()
    e_corr = theDMFET.correction_energy()

    e_tot = e_mf + e_corr

    e_tot_list.append(e_tot)
    print ("e_tot = ", e_tot)


print (e_tot_list)
