from pydmfet import locints, sdmfet,oep
from pyscf import gto, scf, ao2mo
import numpy as np
from pyscf.tools import molden
import FHF_4H2O_struct

thebasis = 'ccpvdz'
e_tot_list = []
for thestructure in range(0,1):

    mol = FHF_4H2O_struct.structure( thestructure, thebasis)

    #total system HF
    mf = scf.RHF(mol)
    mf.max_cycle = 1000
    mf.verbose = 3
    DMguess = None
    mf.scf(dm0=DMguess)
    e_mf = mf.e_tot
    print "e_mf = ", e_mf
    print mol.nao_nr()


    myInts = locints.LocalIntegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )
#    myInts.molden( 'iao.molden' )
    myInts.TI_OK = False # Only s functions

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

    Ne_frag = 20
    boundary_atoms = np.zeros([natoms], dtype=int)
#boundary_atoms[5] = 1
#boundary_atoms[8]=1
#boundary_atoms[9]=1
#boundary_atoms[12]=1
    boundary_atoms =  None

    params = oep.OEPparams(algorithm = 'split', ftol = 1e-11, gtol = 1e-6,diffP_tol=1e-6, outer_maxit = 100, maxit = 100,l2_lambda = 0.0, oep_print = 0)
    theDMFET = sdmfet.DMFET( myInts,impurities, impAtom, Ne_frag, boundary_atoms=boundary_atoms, dim_bath = 25, oep_params=params, ecw_method = 'CCSD')

    umat = theDMFET.embedding_potential()
#    exit()
    e_corr = theDMFET.correction_energy()

    e_tot = e_mf + e_corr



    e_tot_list.append(e_tot)
    print "e_tot = ", e_tot


print e_tot_list
