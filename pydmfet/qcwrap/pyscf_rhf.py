import numpy as np
from pyscf import ao2mo, gto
from pyscf import scf as pyscf_scf

def scf( OEI, TEI, Norb, Nelec, OneDM0=None ):


    # Get the RHF solution
    OEI = 0.5*(OEI.T + OEI)
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

    if ( mf.converged == False ):
        raise Exception(" SCF not converged!")

    #ERHF = mf.e_tot
    RDM1 = mf.make_rdm1()
    RDM1 = 0.5*(RDM1.T + RDM1)
    JK   = mf.get_veff(None, dm=RDM1)
    JK = 0.5*(JK.T + JK)

    mo_coeff = mf.mo_coeff

    energy = np.trace(np.dot(RDM1,OEI)) + 0.5*np.trace(np.dot(RDM1,JK)) 

    return ( energy, RDM1, mo_coeff)


def scf_oei( OEI, Norb, Nelec):

    OEI = 0.5*(OEI.T + OEI)
    eigenvals, eigenvecs = np.linalg.eigh( OEI )

    Nocc = Nelec/2  #closed shell

    idx = eigenvals.argsort()
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:,idx]
    RDM1 = 2.0 * np.dot( eigenvecs[:,:Nocc] , eigenvecs[:,:Nocc].T )
    RDM1 = 0.5*(RDM1.T + RDM1)

    energy = np.trace(np.dot(RDM1,OEI))

    return ( energy, RDM1 )

