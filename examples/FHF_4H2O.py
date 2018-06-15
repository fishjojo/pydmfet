from pydmfet import locints, sdmfet,oep
from pyscf import gto, scf,dft, ao2mo
import numpy as np
from pyscf.tools import molden
import FHF_4H2O_struct

basis_frag = 'ccpvdz'
basis_env = 'ccpvdz'
e_tot_list = []
for thestructure in range(17,18):

    mol_frag, mol_env = FHF_4H2O_struct.structure( thestructure, basis_frag, basis_env)
    mol = gto.mole.conc_mol(mol_frag, mol_env)
    mol.build()

    #total system HF
    #mf = scf.RHF(mol)
    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.max_cycle = 100
    mf.verbose = 3
    DMguess = None
    mf.scf(dm0=DMguess)
    e_mf = mf.e_tot
    print "e_mf = ", e_mf

    myInts = locints.LocalIntegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )
    #myInts.loc_molden( 'loc.molden' )
    myInts.TI_OK = False

    natoms = mol.natm
    impAtom = np.zeros([natoms], dtype=int)
    for i in range(3):
        impAtom[i] = 1

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
    params = oep.OEPparams(algorithm = '2011', ftol = 1e-13, gtol = 1e-6,diffP_tol=1e-6, \
                       outer_maxit = 200, maxit = 200,l2_lambda = 0.0, oep_print = 0)
    theDMFET = sdmfet.DMFET(mf, mol_frag, mol_env, myInts, impurities, impAtom, Ne_frag,\
                        boundary_atoms=boundary_atoms, boundary_atoms2=None,\
                        dim_imp =nbas, dim_bath =nbas,dim_big=None, oep_params=params, ecw_method = 'ccsd',mf_method = 'b3lyp')

    umat = theDMFET.embedding_potential()

    e_corr = theDMFET.correction_energy()

    e_tot = e_mf + e_corr

    e_tot_list.append(e_tot)
    print "e_tot = ", e_tot


print e_tot_list
