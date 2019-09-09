import numpy as np
from pyscf import lib
from .pyscf_rks_ao import rks_ao
from pydmfet import tools

def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):

    vxc = rks_ao.get_veff(ks, mol=mol, dm=ks.dm_guess)

    if dm is None:
        dm = ks.make_rdm1()

    vj = vxc.vj
    vk = vxc.vk
    ecoul = np.einsum('ij,ji', dm, vj)
    exc = np.einsum('ij,ji', dm, np.asarray(vxc) )
    exc -= ecoul

    vxc = lib.tag_array(np.asarray(vxc), ecoul=ecoul, exc=exc, vj=vj, vk=vk)

    return vxc


class rks_nonscf_ao(rks_ao):

    def __init__(self, mol, xc_func = 'lda,vwn', vext_1e = None, extra_oei=None, \
                 coredm = 0.0, dm0=None, smear_sigma = 0.0, max_cycle=50,add_coredm_ext_energy = False):

        rks_ao.__init__(self, mol, xc_func, vext_1e, extra_oei, coredm, dm0, smear_sigma, add_coredm_ext_energy)
        self.max_cycle = max_cycle

        if self.dm_guess is None:
            raise ValueError("dm0 has to be set since it's used as the fixed density")

        self.direct_scf = False

    get_veff = get_veff
