import numpy as np
from pydmfet import tools
from .fermi import find_efermi, entropy_corr
from pyscf import ao2mo, gto, scf, dft, lib
from pydmfet.qcwrap import fermi
import time
from functools import reduce

def scf_oei( OEI, Norb, Nelec, smear_sigma = 0.0):

    OEI = 0.5*(OEI.T + OEI)
    eigenvals, eigenvecs = np.linalg.eigh( OEI )
    idx = np.argmax(abs(eigenvecs), axis=0)
    eigenvecs[:,eigenvecs[ idx, np.arange(len(eigenvals)) ]<0] *= -1

    Nocc = Nelec//2  #closed shell

    e_homo = eigenvals[Nocc-1]
    e_lumo = eigenvals[Nocc]
    print ('HOMO: ', e_homo, 'LUMO: ', e_lumo)
    print ("mo_energy:")
    print (eigenvals[:Nocc+5])

    e_fermi = e_homo
    mo_occ = np.zeros((Norb))

    if(smear_sigma < 1e-8): #T=0
        mo_occ[:Nocc] = 1.0
    else: #finite T
        e_fermi, mo_occ = find_efermi(eigenvals, smear_sigma, Nocc, Norb)

    mo_occ*=2.0 #closed shell

    Ne_error = np.sum(mo_occ) - Nelec
    if(Ne_error > 1e-8):
        print ('Ne error = ', Ne_error)
    print ("fermi energy: ", e_fermi)
    np.set_printoptions(precision=4)
    flag = mo_occ > 1e-4
    print (mo_occ[flag])
    np.set_printoptions()


    RDM1 = reduce(np.dot, (eigenvecs, np.diag(mo_occ), eigenvecs.T))
    RDM1 = (RDM1.T + RDM1)/2.0

    energy = np.trace(np.dot(RDM1,OEI))

    es = entropy_corr(mo_occ, smear_sigma)
    print ('entropy correction: ', es)

    energy += es
    print ('e_tot = ', energy)

    return ( energy, RDM1, eigenvecs, eigenvals, mo_occ )


# The following is deprecated!
class scf_pyscf(): 

    '''
        subspace scf
        wrapper for scf module of pyscf
    '''

    def __init__(self, Ne, Norb, mol=None, oei=None, tei=None, ovlp=1, dm0=None, coredm=0, ao2sub=None, mf_method='HF'):

        self.mol = mol
        self.Ne = Ne
        self.Norb = Norb
        self.method = mf_method

        self.oei = oei
        self.tei = tei
        self.ovlp = ovlp 
        self.dm0 = dm0
        self.coredm = coredm
        self.ao2sub = ao2sub
        self.method = mf_method.lower()

        self.mf = None

        if(self.mol is None):
            #what molecule does not matter
            self.mol = gto.Mole()
            self.mol.build( verbose=0 )
            self.mol.atom.append(('C', (0, 0, 0)))

        #adjust number of electrons
        self.mol.nelectron = Ne

        if(self.tei is not None):
            self.mol.incore_anyway = True

        if(self.method == 'hf'):
            self.mf = scf.RHF(self.mol)
            self.prep_rhf()
        else:
            self.mf = scf.RKS(self.mol)
            self.mf.xc = self.method
            self.prep_rhf()
            self.prep_rks()

        self.elec_energy = 0.0
        self.rdm1 = None
        self.mo_coeff = None
        self.mo_energy = None
        self.mo_occ = None

    def prep_rhf(self):

        if(self.ovlp == 1):
            self.mf.get_ovlp = lambda *args: np.eye( self.Norb )
        if(self.oei is not None):
            self.mf.get_hcore = lambda *args: self.oei
        if(self.tei is not None):
            self.mf._eri = ao2mo.restore(8, self.tei, self.Norb)


    def prep_rks(self):

        if(self.ao2sub is None):
            return

        #overload dft.rks.get_veff if necessary
        self.mf.get_veff = get_veff_rks_decorator(self.ao2sub, self.coredm)


    def kernel(self):

        self.mf.kernel(self.dm0)
        if ( self.mf.converged == False ):
            raise Exception("scf not converged!")


        rdm1 = self.mf.make_rdm1()
        self.rdm1 = 0.5*(rdm1.T + rdm1)
        self.elec_energy = self.mf.energy_elec(self.rdm1)[0]

        self.mo_coeff = self.mf.mo_coeff
        self.mo_energy = self.mf.mo_energy
        self.mo_occ = self.mf.mo_occ



def get_veff_rks_decorator(ao2sub, coredm):


    def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):

        if mol is None: mol = ks.mol
        if dm is None: dm = ks.make_rdm1()

        dm_sub = np.asarray(dm) + coredm
        dm_ao = tools.dm_sub2ao(dm_sub, ao2sub)

        if hasattr(dm, 'mo_coeff'):
            mo_coeff_sub = dm.mo_coeff
            mo_occ_sub = dm.mo_occ

            mo_coeff_ao = tools.mo_sub2ao(mo_coeff_sub, ao2sub)
            mo_occ_ao = mo_occ_sub
            dm_ao = lib.tag_array(dm_ao, mo_coeff=mo_coeff_ao, mo_occ=mo_occ_ao)

        n, exc, vxc_ao, hyb = get_vxc(ks, mol, dm_ao)
        vxc = tools.op_ao2sub(vxc_ao, ao2sub)

        vj = None
        vk = None
        if abs(hyb) < 1e-10:
            if (ks._eri is None and ks.direct_scf and
                getattr(vhf_last, 'vj', None) is not None):
                ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
                vj = ks.get_jk(mol, ddm, hermi)[0]
                vj += vhf_last.vj
            else:
                vj = ks.get_jk(mol, dm, hermi)[0]
            vxc += vj
        else:
            if (ks._eri is None and ks.direct_scf and
                getattr(vhf_last, 'vk', None) is not None):
                ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
                vj, vk = ks.get_jk(mol, ddm, hermi)
                vj += vhf_last.vj
                vk += vhf_last.vk
            else:
                vj, vk = ks.get_jk(mol, dm, hermi)
            vxc += vj - vk * (hyb * .5)
            exc -= np.einsum('ij,ji', dm, vk) * .5 * hyb*.5

        ecoul = np.einsum('ij,ji', dm, vj) * .5

        vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)

        return vxc

    return get_veff


def get_vxc(ks, mol, dm, hermi=1):

    ground_state = (isinstance(dm, numpy.ndarray) and dm.ndim == 2)
    if(not ground_state):
        raise Exception("fatal error")

    if ks.grids.coords is None:
        ks.grids.build(with_non0tab=True)
        if ks.small_rho_cutoff > 1e-20 and ground_state:
            # Filter grids the first time setup grids
            t0 = (time.clock(), time.time())
            ks.grids = dft.rks.prune_small_rho_grids_(ks, mol, dm, ks.grids)
            t1 = tools.timer("prune grid",t0)

    if hermi == 2:  # because rho = 0
        n, exc, vxc = 0, 0, 0
    else:
        n, exc, vxc = ks._numint.nr_rks(mol, ks.grids, ks.xc, dm)

    hyb = ks._numint.hybrid_coeff(ks.xc, spin=mol.spin)

    return n, exc, vxc, hyb



'''
def rhf(mol, OEI, TEI, Norb, Nelec, OneDM0=None ):

    # Get the RHF solution
    OEI = 0.5*(OEI.T + OEI)
    #mol = gto.Mole()
    #mol.max_memory = 8000
    #mol.build( verbose=0 )
    #mol.atom.append(('C', (0, 0, 0)))
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


def rks(mol, OEI, TEI, Norb, Nelec, xcfunc, OneDM0=None ):

    # Get the RKS solution
    OEI = 0.5*(OEI.T + OEI)
    #mol = gto.Mole()
    #mol.build( verbose=5 )
    #mol.atom.append(('C', (0, 0, 0)))
    mol.nelectron = Nelec
#    mol.incore_anyway = True

    mf = pyscf_scf.RKS( mol )
    mf.xc = xcfunc.lower()
#    mf.get_hcore = lambda *args: OEI
#    mf.get_ovlp = lambda *args: np.eye( Norb )
#    mf._eri = ao2mo.restore(8, TEI, Norb)
    OneDM0 = None
    mf.kernel( OneDM0 )

    if ( mf.converged == False ):
        raise Exception(" rks not converged!")

    return mf


def scf(mol, OEI, TEI, Norb, Nelec, OneDM0=None, mf_method = 'HF' ):

    # Get the mean-field solution
    if(mf_method.lower() == 'hf'):
        mf = rhf(mol, OEI, TEI, Norb, Nelec, OneDM0 )
    else:
        mf = rks(mol, OEI, TEI, Norb, Nelec, mf_method ,OneDM0 )

    RDM1 = mf.make_rdm1()
    RDM1 = 0.5*(RDM1.T + RDM1)

    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    energy = mf.energy_elec(RDM1)[0]

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
'''
