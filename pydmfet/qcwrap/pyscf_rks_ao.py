import numpy
import time
from pyscf.scf import hf
from pyscf.dft import rks
from .fermi import find_efermi, entropy_corr


def calc_ne(ks,dm):

    mol = ks.mol
    s = mol.intor_symmetric('int1e_ovlp')
    ne = int(numpy.trace(numpy.dot(dm,s)))

    return ne

def energy_elec(ks, dm=None, h1e=None, vhf=None):

    tot_e,e_hxc = rks.energy_elec(ks, dm=dm, h1e=h1e, vhf=vhf)
    if(ks.smear_sigma > 1e-8 ):
        #add entropy contribution
        tot_e += entropy_corr(ks.mo_occ, ks.smear_sigma)

    #add frozen density hcore energy
    if(isinstance(ks.coredm, numpy.ndarray) and ks.coredm.ndim == 2): 
        h = hf.get_hcore(ks.mol)
        if(ks.add_coredm_ext_energy == True):
            h = h + ks.vext_1e
        e1 = numpy.einsum('ij,ji', h, ks.coredm).real
        tot_e += e1

    return tot_e, e_hxc


def get_hcore(ks,mol=None):

    if mol is None: mol = ks.mol

    h = hf.get_hcore(mol)
    if ks.vext_1e is not None:
        h = h + ks.vext_1e

    return h


def get_occ(mf, mo_energy=None, mo_coeff=None, smear_sigma = None):

    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if smear_sigma is None: smear_sigma = mf.smear_sigma

    nelectron = mf.mol.nelectron - mf.ne_frozen

    nmo = mo_energy.size
    mo_occ = numpy.zeros(nmo)
    Nocc = nelectron // 2

    e_idx = numpy.argsort(mo_energy)
    e_sort = mo_energy[e_idx]
    mo_occ[e_idx[:Nocc]] = 1.0

    e_homo = e_sort[Nocc-1]
    e_lumo = e_sort[Nocc]
    print ('HOMO: ',e_homo, 'LUMO: ', e_lumo)

    e_fermi = e_homo

    if(smear_sigma > 1e-8):
        e_fermi, mo_occ = find_efermi(mo_energy, smear_sigma, Nocc, nmo)

    mo_occ *= 2.0  #closed shell

    ne = numpy.sum(mo_occ)
    Ne_error = ne - nelectron
    if(abs(Ne_error) > 1e-8):
        print ('Ne error: ', Ne_error)
    print ("Fermi energy: {e:18.10f}".format(e=e_fermi) )
    numpy.set_printoptions(precision=4)
    is_occupied = mo_occ > 1e-4
    print (mo_occ[is_occupied])
    numpy.set_printoptions()

    return mo_occ


def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):

    if mol is None: mol = ks.mol
    if dm is None: dm = ks.make_rdm1()

    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        #add frozen density
        dm = dm + ks.coredm
    else:
        raise RuntimeError("something is wrong!")

    if isinstance(dm_last, numpy.ndarray): dm_last = dm_last + ks.coredm 

    vxc = rks.get_veff(ks, mol, dm, dm_last, vhf_last, hermi)

    #print("Ecoul = {ec:20.14f}  Exc = {exc:20.14f}".format(ec=vxc.ecoul, exc=vxc.exc))
    return vxc


def kernel(mf, dm0=None, **kwargs):

    if dm0 is None:
        dm0 = mf.dm_guess

    mf.converged, mf.e_tot, \
                mf.mo_energy, mf.mo_coeff, mf.mo_occ = \
                hf.kernel(mf, mf.conv_tol, mf.conv_tol_grad,
                          dm0=dm0, callback=mf.callback,
                          conv_check=mf.conv_check, **kwargs)

    if (mf.converged == False ):
        print ("scf did not converge")
        #raise RuntimeError("scf did not converge")

    mf.rdm1 = mf.make_rdm1()
    mf.elec_energy = mf.energy_elec(mf.rdm1)[0]


class rks_ao(rks.RKS):

    '''
        rks with Fermi smearing and external potential
        wrapper for dft.rks module of pyscf 
    '''

    def __init__(self, mol, xc_func = 'lda,vwn', vext_1e = None, extra_oei=None, \
                 coredm = 0.0, dm0=None, smear_sigma = 0.0, max_cycle = 50, add_coredm_ext_energy = False):

        self.smear_sigma = smear_sigma
        rks.RKS.__init__(self, mol)
        self.xc = xc_func
        self.max_cycle = max_cycle
        self.vext_1e = vext_1e
        if extra_oei is not None:
            self.vext_1e += extra_oei

        self.coredm = coredm
        self.add_coredm_ext_energy = add_coredm_ext_energy
        self.ne_frozen = 0
        self.dm_guess = dm0

        if(isinstance(self.coredm, numpy.ndarray) and self.coredm.ndim == 2):
            self.ne_frozen = calc_ne(self, self.coredm)
        print("rks_ao.ne_frozen = ",self.ne_frozen)

    energy_elec = energy_elec
    get_occ = get_occ
    get_hcore = get_hcore
    get_veff = get_veff
    kernel = kernel
