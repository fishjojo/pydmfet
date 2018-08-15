from pyscf import lib
from pyscf.dft import rks
from pydmfet.qcwrap import pyscf_rks
import numpy as np


def get_hcore(mf, mol=None, umat=None):
    '''Core Hamiltonian
    '''

    if mol is None: mol = mf.mol
    if umat is None: umat = mf.umat

    h = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
    if mol.has_ecp():
        h += mol.intor_symmetric('ECPscalar')
    h += umat

    return h


def energy_elec(ks, dm=None, h1e=None, vhf=None):

    tot_e, ecoul_exc = rks.energy_elec(ks,dm,h1e,vhf)
    if(hasattr(ks,'smear_sigma')):
	tot_e += pyscf_rks.entropy_corr(ks.mo_occ, ks.smear_sigma)

    return tot_e, ecoul_exc


#restricted scf
class EmbedSCF(rks.RKS):

    def __init__(self, mol, umat=0.0, smear_sigma=0.0):

	self.umat = umat
	self.smear_sigma = smear_sigma	

	rks.RKS.__init__(self,mol)

    get_hcore = get_hcore
    get_occ = pyscf_rks.get_occ 
    energy_elec = energy_elec


#restricted nonscf
class EmbedSCF_nonscf(rks.RKS):

    def __init__(self, mol, dm_fix, umat=0.0, smear_sigma=0.0):

        self.umat = umat
	self.dm_fix = dm_fix
	self.smear_sigma = smear_sigma

        rks.RKS.__init__(self,mol)

    get_hcore = get_hcore
    get_occ = pyscf_rks.get_occ
    energy_elec = energy_elec


    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):

	vxc = rks.get_veff(self, mol=mol, dm=self.dm_fix, dm_last=0, vhf_last=0, hermi=hermi)
	
	if dm is None:
	    dm = self.make_rdm1()

	vj = vxc.vj
	vk = vxc.vk
	ecoul = np.einsum('ij,ji', dm, vj)
	exc = np.einsum('ij,ji', dm, np.asarray(vxc) )
	exc -= ecoul

	vxc = lib.tag_array(np.asarray(vxc), ecoul=ecoul, exc=exc, vj=vj, vk=vk)

	return vxc
