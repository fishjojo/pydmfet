import numpy as np
from pydmfet import tools
from pyscf import ao2mo, gto
from pyscf import scf as pyscf_scf


def rhf( OEI, TEI, Norb, Nelec, OneDM0=None ):

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
    mf.kernel( OneDM0 )

    if ( mf.converged == False ):
        raise Exception(" rhf not converged!")

    return mf


def rks( OEI, TEI, Norb, Nelec, xcfunc, OneDM0=None ):

    # Get the RKS solution
    OEI = 0.5*(OEI.T + OEI)
    mol = gto.Mole()
    mol.build( verbose=0 )
    mol.atom.append(('C', (0, 0, 0)))
    mol.nelectron = Nelec
    mol.incore_anyway = True
    mf = pyscf_scf.RKS( mol )
    mf.xc = xcfunc.lower()
    mf.get_hcore = lambda *args: OEI
    mf.get_ovlp = lambda *args: np.eye( Norb )
    mf._eri = ao2mo.restore(8, TEI, Norb)
    mf.kernel( OneDM0 )

    if ( mf.converged == False ):
        raise Exception(" rks not converged!")

    return mf



def scf( OEI, TEI, Norb, Nelec, OneDM0=None, mf_method = 'HF' ):

    # Get the mean-field solution
    if(mf_method.lower() == 'hf'):
	mf = rhf( OEI, TEI, Norb, Nelec, OneDM0 )
    else:
	mf = rks( OEI, TEI, Norb, Nelec, xcfunc ,OneDM0 )

    RDM1 = mf.make_rdm1()
    RDM1 = 0.5*(RDM1.T + RDM1)

    mo_coeff = mf.mo_coeff

    energy = mf.energy_elec()[0]

    print "mo energy"
    print mf.mo_energy
    #tools.MatPrint(mf.get_fock(),"fock")
    tools.MatPrint(mf.mo_coeff,"mo_coeff")
    return (energy, RDM1, mo_coeff)


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

