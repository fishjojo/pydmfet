import numpy as np
from pydmfet import tools
from pyscf import ao2mo, gto
from pyscf import scf as pyscf_scf


def rhf( OEI, TEI, Norb, Nelec, OneDM0=None ):

    # Get the RHF solution
    OEI = 0.5*(OEI.T + OEI)
    mol = gto.Mole()
    mol.max_memory = 8000
    mol.build( verbose=0 )
    mol.atom.append(('C', (0, 0, 0)))
    mol.nelectron = Nelec
    mol.incore_anyway = True
    mf = pyscf_scf.RHF( mol )
    mf.get_hcore = lambda *args: OEI
    mf.get_ovlp = lambda *args: np.eye( Norb )
    mf._eri = ao2mo.restore(8, TEI, Norb)
    mf.max_cycle = 100
    #mf.conv_tol = 1e-8
    #adiis = pyscf_scf.diis.ADIIS()
    #mf.diis = adiis
    #mf.verbose = 5
    mf.kernel(OneDM0)

    if ( mf.converged == False ):
	#RDM1 = mf.make_rdm1()
	#cdiis = pyscf_scf.diis.SCF_DIIS()
	#mf.diis = cdiis
	#mf.max_cycle = 200
	#mf.kernel(RDM1)
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
    mo_energy = mf.mo_energy
    energy = mf.energy_elec()[0]

    mo = np.zeros([Norb,Norb+1],dtype=float)
    mo[:,:-1] = mo_coeff
    mo[:,-1] = mo_energy

    #print "mo energy"
    #print mf.mo_energy
    #tools.MatPrint(mf.get_fock(),"fock")
    #JK   = mf.get_veff(None, dm=RDM1)
    #tools.MatPrint(JK,"JK")
    #tools.MatPrint(np.dot(mf.get_fock(), mf.mo_coeff),"test")
    #tools.MatPrint(mf.mo_coeff,"mo_coeff")
    return (energy, RDM1, mo)


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

