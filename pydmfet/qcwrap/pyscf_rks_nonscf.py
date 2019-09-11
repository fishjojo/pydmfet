import numpy as np
from .pyscf_rks import rks_pyscf, get_vxc
from pydmfet import tools
from .fermi import find_efermi, entropy_corr
from functools import reduce


def kernel(ks):

    fock = ks.oei + ks.vhxc
    if ks.vext_1e is not None:
        fock += ks.vext_1e

    fock = 0.5*(fock.T + fock)

    eigenvals, eigenvecs = np.linalg.eigh( fock )
    idx = np.argmax(abs(eigenvecs), axis=0)
    eigenvecs[:,eigenvecs[ idx, np.arange(len(eigenvals)) ]<0] *= -1

    Nocc = ks.Ne//2  #closed shell

    e_homo = eigenvals[Nocc-1]
    e_lumo = eigenvals[Nocc]
    print ('HOMO: ', e_homo, 'LUMO: ', e_lumo)
    print ("mo_energy:")
    print (eigenvals[:Nocc+5])

    e_fermi = e_homo
    mo_occ = np.zeros((ks.Norb))

    if(ks.smear_sigma < 1e-8): #T=0
        mo_occ[:Nocc] = 1.0
    else: #finite T
        e_fermi, mo_occ = find_efermi(eigenvals, ks.smear_sigma, Nocc, ks.Norb)

    mo_occ*=2.0 #closed shell

    Ne_error = np.sum(mo_occ) - ks.Ne
    if(Ne_error > 1e-8):
        print ('Ne error = ', Ne_error)
    print ("fermi energy: ", e_fermi)
    np.set_printoptions(precision=4)
    flag = mo_occ > 1e-4
    print (mo_occ[flag])
    np.set_printoptions()

    rdm1 = reduce(np.dot, (eigenvecs, np.diag(mo_occ), eigenvecs.T))
    rdm1 = 0.5*(rdm1.T + rdm1)

    energy = np.trace(np.dot(rdm1,fock))

    es = entropy_corr(mo_occ, ks.smear_sigma)
    print ('entropy correction: ', es)

    energy += es
    print ('e_tot = ', energy)

    ks.mo_occ = mo_occ
    ks.mo_energy = eigenvals
    ks.mo_coeff = eigenvecs
    ks.rdm1 = rdm1
    ks.elec_energy = energy
 

class rks_nonscf(rks_pyscf):

    def __init__(self, Ne, Norb, mf_method, mol=None, vext_1e = None, oei=None, vhxc=None, tei=None, ovlp=1, dm0=None,\
                 coredm=0.0, ao2sub=1.0, level_shift=0.0, smear_sigma = 0.0, max_cycle = 50):

        rks_pyscf.__init__(self, Ne, Norb, mf_method, mol, vext_1e, oei, tei, ovlp, dm0, coredm, ao2sub, level_shift, smear_sigma)
        self.max_cycle = max_cycle

        if self.dm_guess is None:
            raise ValueError("dm0 has to be set since it's used as the fixed density")
        if self.tei is None:
            raise ValueError("tei has to be set")

        Kcoeff = self._numint.hybrid_coeff(self.xc)
        self.vhxc = vhxc
        if self.vhxc is None:
            self.vhxc = tools.dm2jk(self.dm_guess, self.tei, Kcoeff)
            if(self.method != 'hf'):
                vxc_ao = get_vxc(self, self.mol, self.coredm + tools.dm_sub2ao(self.dm_guess, ao2sub))[2]
                self.vhxc += tools.op_ao2sub(vxc_ao, ao2sub)

    kernel = kernel 
