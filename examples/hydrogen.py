from pydmfet import locints, sdmfet, oep, tools, dfet_ao
from pydmfet.dfet_ao import dfet
from pydmfet.qcwrap.pyscf_rks_ao import rks_ao
from pyscf import gto, scf,dft, ao2mo,cc
import numpy as np
from pyscf.tools import molden, cubegen
import copy,time

DMguess  = None

bondlengths = np.arange(0.74, 0.79, 0.1)
energies = []

bas = 'sto-6g'

for bondlength in bondlengths:

    nat = 20
    mol = gto.Mole()
    mol.atom = []
    r = 0.5 * bondlength / np.sin(np.pi/nat)
    for i in range(nat):
        theta = i * (2*np.pi/nat)
        mol.atom.append(('H', (r*np.cos(theta), r*np.sin(theta), 0)))

    mol.basis = bas
    mol.build(max_memory = 4000,verbose=4)

    mf = dft.UKS(mol)
    #mf = rks_ao(mol,smear_sigma = 0.005)
    #mf = dfet_ao.scf.EmbedSCF(mol, 0.0, smear_sigma = 0.005)
    mf.xc = 'pbe,pbe'
    mf.max_cycle = 50
    mf.scf(dm0=DMguess)
    exit()

    P=mf.make_rdm1()
    #tools.MatPrint(P,"P_ref_ao")
    cubegen.density(mol, "h20_dens.cube", P, nx=100, ny=100, nz=100)

    if ( False ):   
#        ENUCL = mf.mol.energy_nuc()
#        OEI   = np.dot(np.dot(mf.mo_coeff.T, mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph')), mf.mo_coeff)
#        TEI   = ao2mo.outcore.full_iofree(mol, mf.mo_coeff, compact=False).reshape(mol.nao_nr(), mol.nao_nr(), mol.nao_nr(), mol.nao_nr())
#        import chemps2
#        Energy, OneDM = chemps2.solve( ENUCL, OEI, OEI, TEI, mol.nao_nr(), mol.nelectron, mol.nao_nr(), 0.0, False )
#        print "bl =", bondlength," and energy =", Energy

        mycc = cc.CCSD(mf).run()
        et = mycc.ccsd_t()
        e_hf = mf.e_tot
        e_ccsd = e_hf + mycc.e_corr + et

        print e_ccsd    #-4.96124910741
        
    else:
#        myInts = locints.LocalIntegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )
        #myInts.loc_molden( 'hydrogen-loc.molden' )
        #myInts.TI_OK = True # Only s functions

	nbas =  mol.nao_nr()
	natoms = mol.natm

	impAtom = np.zeros([natoms], dtype=int)
	for i in range(10):
	    impAtom[i] = 1

	ghost_frag = 1-impAtom
	ghost_env = 1-ghost_frag

	mol_frag = gto.Mole()
	mol_frag.atom = tools.add_ghost(mol.atom, ghost_frag)
	mol_frag.basis = bas
	mol_frag.build(max_memory = 16000,verbose = 4)
	'''
	mf_frag = dft.RKS(mol_frag)
        mf_frag.xc = 'pbe,pbe'
        mf_frag.smear_sigma = 0.00
        mf_frag.scf()
        P_frag=mf_frag.make_rdm1()
        cubegen.density(mol_frag, "h10_dens_1.cube", P_frag, nx=100, ny=100, nz=100)
	'''

	mol_env = gto.Mole()
	mol_env.atom = tools.add_ghost(mol.atom, ghost_env)
	mol_env.basis =  bas
	mol_env.build(max_memory = 16000,verbose = 4)

	'''
	mf_env = dft.RKS(mol_env)
        mf_env.xc = 'pbe,pbe'
        mf_env.smear_sigma = 0.00
        mf_env.scf()
        P_env=mf_env.make_rdm1()
        cubegen.density(mol_env, "h10_dens_2.cube", P_env, nx=100, ny=100, nz=100)
	exit()
	'''



	aoslice = mol.aoslice_by_atom()
	impurities = np.zeros([nbas], dtype = int)
	for i in range(natoms):
	    if(impAtom[i] == 1):
		impurities[aoslice[i,2]:aoslice[i,3]] = 1

	Ne_frag = 10
	
	boundary_atoms = np.zeros((natoms))
	boundary_atoms[10:20] = 1
	boundary_atoms2 = np.zeros((natoms))
	boundary_atoms2[0:10] = 1
	
	#boundary_atoms = None
	#boundary_atoms2 =None

	umat=None
	#P_frag=0.5*mf.make_rdm1()
	#P_env=0.5*mf.make_rdm1()
	P_frag=None
	P_env=None
	params = oep.OEPparams(algorithm = '2011', opt_method = 'L-BFGS-B', \
                       ftol = 1e-12, gtol = 1e-5,diffP_tol=1e-5, outer_maxit = 1, maxit = 200,l2_lambda = 0.0, oep_print = 0)
#	theDMFET = sdmfet.DMFET( mf, mol_frag, mol_env,myInts,impurities, impAtom, Ne_frag, boundary_atoms=boundary_atoms, boundary_atoms2=boundary_atoms2,\
#                         umat = umat, P_frag_ao = P_frag, P_env_ao = P_env, \
#                         dim_imp = nbas, dim_bath=nbas, dim_big =nbas, smear_sigma = 0.005, oep_params=params,ecw_method='hf', mf_method =mf.xc)

	theDMFET = dfet.DFET(mf, mol_frag, mol_env,\
                     boundary_atoms=boundary_atoms, boundary_atoms2=boundary_atoms2,\
                     oep_params=params, smear_sigma=0.005, ecw_method = 'hf',mf_method = mf.xc, plot_dens=True)

	umat = theDMFET.embedding_potential()
	#energy = theDMFET.correction_energy()
	exit()

	#write subspace orbitals
	transfo = np.dot( myInts.ao2loc, theDMFET.loc2sub )
	filename =  'hydrogen-sub.molden'
	with open( filename, 'w' ) as thefile:
            molden.header( myInts.mol, thefile )
            molden.orbital_coeff( myInts.mol, thefile, transfo )


        umat = theDMFET.embedding_potential()
	print umat
	exit()
	energy = theDMFET.correction_energy()

