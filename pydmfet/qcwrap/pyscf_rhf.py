import numpy as np
from pyscf import ao2mo, gto
from pyscf import scf as pyscf_scf

def scf( OEI, TEI, Norb, Nelec, OneDM0=None ):


    # Get the RHF solution
    mol = gto.Mole()
    mol.build( verbose=0 )
    mol.atom.append(('C', (0, 0, 0)))
    mol.nelectron = Nelec
    mol.incore_anyway = True
    mf = pyscf_scf.RHF( mol )
    mf.get_hcore = lambda *args: OEI
    mf.get_ovlp = lambda *args: np.eye( Norb )
    mf._eri = ao2mo.restore(8, TEI, Norb)
    mf.scf( OneDM0 )
    DMloc = np.dot(np.dot( mf.mo_coeff, np.diag( mf.mo_occ )), mf.mo_coeff.T )

    if ( mf.converged == False ):
        assert(0==1)

    ERHF = mf.e_tot
    RDM1 = mf.make_rdm1()
    JK   = mf.get_veff(None, dm=RDM1)

    RDM1 = 0.5*(RDM1.T + RDM1)

    return ( ERHF, RDM1 )

