from pydmfet import locints, sdmfet, oep, tools
from pyscf import gto, scf,dft, ao2mo,cc
import numpy as np
from pyscf.tools import molden, cubegen
import copy,time

DMguess  = None

bondlengths = np.arange(1.8,1.9, 0.1)
energies = []

bas = 'sto-6g'

for bondlength in bondlengths:

    nat = 30
    mol = gto.Mole()
    mol.atom = []
    r = 0.5 * bondlength / np.sin(np.pi/nat)
    for i in range(nat):
        theta = i * (2*np.pi/nat)
        mol.atom.append(('Be', (r*np.cos(theta), r*np.sin(theta), 0)))

    mol.basis = bas
    mol.build(max_memory=16000, verbose=4)

    #mf = scf.RHF(mol)
    mf = dft.RKS(mol)
    mf.xc = 'pbe,pbe'
    mf.smear_sigma = 0.005
    mf.max_cycle = 100
    mf.scf()

    #P=mf.make_rdm1()
    #tools.MatPrint(P,"P_ref_ao")
    #cubegen.density(mol, "h20_dens.cube", P, nx=100, ny=100, nz=100)
    print 'e_mf = ',  mf.e_tot
    #continue
    if ( False ):   
#        ENUCL = mf.mol.energy_nuc()
#        OEI   = np.dot(np.dot(mf.mo_coeff.T, mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph')), mf.mo_coeff)
#        TEI   = ao2mo.outcore.full_iofree(mol, mf.mo_coeff, compact=False).reshape(mol.nao_nr(), mol.nao_nr(), mol.nao_nr(), mol.nao_nr())
#        import chemps2
#        Energy, OneDM = chemps2.solve( ENUCL, OEI, OEI, TEI, mol.nao_nr(), mol.nelectron, mol.nao_nr(), 0.0, False )
#        print "bl =", bondlength," and energy =", Energy

        mycc = cc.CCSD(mf).run()
	et=0.0
        #et = mycc.ccsd_t()
        e_hf = mf.e_tot
        e_ccsd = e_hf + mycc.e_corr + et

        print 'e_tot = ', e_ccsd    #-4.96124910741
        
    else:
        myInts = locints.LocalIntegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )
        #myInts.loc_molden( 'hydrogen-loc.molden' )
        #myInts.TI_OK = True # Only s functions

	nbas =  mol.nao_nr()
	natoms = mol.natm

	impAtom = np.zeros([natoms], dtype=int)
	for i in range(2):
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

	Ne_frag = 8
	boundary_atoms = np.zeros((natoms))
	boundary_atoms[20:40] = 1.0
	#boundary_atoms[19] = -1
	boundary_atoms2 = np.zeros((natoms),dtype =int)
	#boundary_atoms2[9] = -1
	boundary_atoms2[0:20] = 1

	boundary_atoms = None
	boundary_atoms2 =None

	umat=None
	P_frag=None
	P_env=None
	params = oep.OEPparams(algorithm = 'split', opt_method = 'L-BFGS-B', \
                       ftol = 1e-11, gtol = 1e-5,diffP_tol=1e-5, outer_maxit = 50, maxit = 200,l2_lambda = 0.0, oep_print = 0)
	theDMFET = sdmfet.DMFET( mf, mol_frag, mol_env,myInts,impurities, impAtom, Ne_frag, boundary_atoms=boundary_atoms, boundary_atoms2=boundary_atoms2,\
                         umat = umat, P_frag_ao = P_frag, P_env_ao = P_env, \
                         dim_imp = nbas, dim_bath=nbas, dim_big =nbas, smear_sigma = 0.005, oep_params=params,ecw_method='hf', mf_method = mf.xc)

	umat = theDMFET.embedding_potential()

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

