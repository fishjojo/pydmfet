from pydmfet import locints, sdmfet, oep
from pyscf import gto, scf, ao2mo,cc
import numpy as np
from pyscf.tools import molden
import copy,time

DMguess  = None

bondlengths = np.arange(0.7, 0.79, 0.1)
energies = []

for bondlength in bondlengths:

    nat = 10
    mol = gto.Mole()
    mol.atom = []
    r = 0.5 * bondlength / np.sin(np.pi/nat)
    for i in range(nat):
        theta = i * (2*np.pi/nat)
        mol.atom.append(('H', (r*np.cos(theta), r*np.sin(theta), 0)))

    mol.basis = 'sto-6g'
    mol.build(verbose=0)


    mf = scf.RHF(mol)
    mf.verbose = 3
    mf.max_cycle = 1000
    mf.scf(dm0=DMguess)
    #mf.analyze()

    #molden.from_mo(mol,'hydrogen-mo.molden',mf.mo_coeff)

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
        myInts = locints.LocalIntegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )
        #myInts.molden( 'hydrogen-loc.molden' )
        myInts.TI_OK = True # Only s functions

        orbs_per_imp = 8

        impurities = np.zeros( [ myInts.Norbs ], dtype=int )
        for orb in range( orbs_per_imp ):
            impurities[ orb ] = 1

        impAtom = impurities
        Ne_frag = 8

	boundary_atoms = np.zeros( [ myInts.Norbs ], dtype=int )
	boundary_atoms[8] = 1
	boundary_atoms[9] = 1

	params = oep.OEPparams(algorithm = 'split', ftol = 1e-10, gtol = 1e-6, diffP_tol = 1e-6, outer_maxit = 100, oep_print = 3)
        theDMFET = sdmfet.DMFET( myInts,impurities, impAtom, Ne_frag, boundary_atoms = boundary_atoms, oep_params = params,ecw_method = 'CCSD')
        umat = theDMFET.embedding_potential()
	print umat
	
	energy = theDMFET.total_energy()

