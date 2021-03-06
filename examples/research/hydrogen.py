from pydmfet import locints, sdmfet, oep, tools, dfet_ao
from pydmfet.dfet_ao import dfet
from pydmfet.qcwrap.pyscf_rks_ao import rks_ao
from pyscf import gto, scf,dft, ao2mo,cc
import numpy as np
from pyscf.tools import molden, cubegen
import copy,time
from read_umat import read_umat
from pydmfet.tools.sym import h_lattice_sym_tab

DMguess  = None

bondlengths = np.arange(0.74, 0.79, 0.1)
energies = []

bas = 'sto-6g'
#bas = 'cc-pvdz'

temp = 0.00

for bondlength in bondlengths:

    nat = 20
    mol = gto.Mole()
    mol.atom = []
    r = 0.5 * bondlength / np.sin(np.pi/nat)
    for i in range(nat):
        theta = i * (2*np.pi/nat)
        mol.atom.append(('H', (r*np.cos(theta), r*np.sin(theta), 0)))

    mol.basis = bas
    mol.build(max_memory = 4000,verbose=5)

    #mf = scf.UKS(mol)
    mf = rks_ao(mol,smear_sigma = temp)
    #mf = dfet_ao.scf.EmbedSCF(mol, 0.0, smear_sigma = temp)
    mf.xc = 'hf'
    mf.max_cycle = 50
    mf.scf(dm0=DMguess)

    #P=mf.make_rdm1()
    #P = P[0]+P[1]
    #tools.MatPrint(P,"P_ref_ao")
    #cubegen.density(mol, "dens_tot.cube", P, nx=100, ny=100, nz=100)

    with open( 'h20_mo.molden', 'w' ) as thefile:
        molden.header(mf.mol, thefile)
        molden.orbital_coeff(mf.mol, thefile, mf.mo_coeff, occ=mf.mo_occ, ene=mf.mo_energy)
    exit()


    atm_ind = np.zeros([nat//2,2],dtype=int)
    for i in range(nat//2):
        atm_ind[i,0] = i
        atm_ind[i,1] = nat-i-1

    sym_tab = h_lattice_sym_tab(atm_ind)
    print("sym_tab")
    print (sym_tab)



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

        print (e_ccsd)    #-4.96124910741
        
    else:
        #myInts = locints.LocalIntegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )
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

        mol_env = gto.Mole()
        mol_env.atom = tools.add_ghost(mol.atom, ghost_env)
        mol_env.basis =  bas
        mol_env.build(max_memory = 16000,verbose = 4)

        '''
        aoslice = mol.aoslice_by_atom()
        impurities = np.zeros([nbas], dtype = int)
        for i in range(natoms):
            if(impAtom[i] == 1):
                impurities[aoslice[i,2]:aoslice[i,3]] = 1
        '''

        Ne_frag = 10
        Ne_env = 10
        
        boundary_atoms = None
        boundary_atoms2 =None

        umat = None
        #umat = read_umat(20,"hydrogen.u")
        umat = np.load("umat.npy")

        P_frag=None
        P_env=None
        params = oep.OEPparams(algorithm = 'split', opt_method = 'newton', diffP_tol=1e-4, outer_maxit = 1)
        params.options['ftol'] = 1e-9
        params.options['gtol'] = 1e-4
        params.options['maxiter'] = 10
        params.options['svd_thresh'] = 1e-4


        '''
        theDMFET = sdmfet.DMFET( mf, mol_frag, mol_env,myInts,impurities, impAtom, Ne_frag, boundary_atoms=boundary_atoms, boundary_atoms2=boundary_atoms2,\
                         umat = umat, P_frag_ao = P_frag, P_env_ao = P_env, \
                         dim_imp = nbas, dim_bath=nbas, dim_big =None, smear_sigma = temp, oep_params=params,ecw_method='hf', mf_method =mf.xc)
        '''

        theDMFET = dfet.DFET(mf, mol_frag, mol_env,Ne_frag,Ne_env,\
                     boundary_atoms=boundary_atoms, boundary_atoms2=boundary_atoms2,umat = umat,\
                     oep_params=params, smear_sigma=temp, ecw_method = 'hf',mf_method = mf.xc, plot_dens=True)

        #theDMFET.sym_tab = sym_tab

        umat = theDMFET.embedding_potential()
        #energy = theDMFET.correction_energy()
        exit()

        #write subspace orbitals
        transfo = np.dot( myInts.ao2loc, theDMFET.loc2sub )
        filename =  'hydrogen-sub.molden'
        with open( filename, 'w' ) as thefile:
            molden.header( myInts.mol, thefile )
            molden.orbital_coeff( myInts.mol, thefile, transfo )

