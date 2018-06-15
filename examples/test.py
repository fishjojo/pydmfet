from pydmfet import locints, sdmfet,oep,tools
from pyscf import gto, scf, ao2mo,dft
import numpy as np
from pyscf.tools import molden
import time

t0 = (time.clock(), time.time())

if(True):

    mol = gto.Mole()
    mol.atom = open('test2.xyz').read()
    mol.basis = 'ccpvdz'
    mol.charge = -1
    mol.spin = 0
    mol.build()
    print mol.nao_nr()

    #total system scf
    mf = scf.RHF(mol)
    #mf = scf.RKS(mol)
    #mf.xc = 'b3lyp'
    #mf.init_guess = 'atom'
    '''
    mf.grids.level = -1
    #mf.grids.atom_grid = {'H': (50,194), 'O': (50,194), 'F': (50,194)}
    mf.grids.prune = dft.gen_grid.sg1_qchem
    #mf.grids.prune = dft.gen_grid.treutler_prune
    #mf.grids.radi_method = dft.radi.delley
    #mf.grids.radi_method = dft.radi.gauss_chebyshev
    mf.grids.radi_method = dft.radi.murray
    #mf.grids.atomic_radii = dft.radi.SG1RADII
    #mf.grids.radii_adjust = None
    mf.grids.radii_adjust = dft.radi.becke_atomic_radii_adjust
    #mf.small_rho_cutoff = 1e-9
    '''
    mf.max_cycle = 1000
    mf.verbose = 4
    DMguess = None
    mf.scf(dm0=DMguess)
    e_mf = mf.e_tot
    print "e_mf = ", e_mf
    t1 = tools.timer("full scf", t0)

    myInts = locints.LocalIntegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )
#    myInts.molden( 'iao.molden' )
    myInts.TI_OK = False # Only s functions
    t2 = tools.timer("orbital localization", t1)

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

    params = oep.OEPparams(algorithm = '2011', ftol = 1e-11, gtol = 1e-6,diffP_tol=1e-6, outer_maxit = 100, maxit = 100,l2_lambda = 0.0, oep_print = 0)
    theDMFET = sdmfet.DMFET(mf,mol,mol, myInts,impurities, impAtom, Ne_frag, boundary_atoms=boundary_atoms, dim_bath = 33, oep_params=params, ecw_method = 'CCSD',mf_method = 'hf')

    t3 = tools.timer("dfet init", t2)

    umat = theDMFET.embedding_potential()

    t4 = tools.timer("oep", t3)

    e_corr = theDMFET.correction_energy()

    t5 = tools.timer("ccsd(t)", t4)
    e_tot = e_mf + e_corr

    t6 = tools.timer("total", t0)



