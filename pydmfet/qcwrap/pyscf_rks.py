import numpy as np
from pydmfet import tools
from pydmfet.qcwrap import pyscf_rhf
from pyscf import ao2mo, gto, scf, dft, lib
from pyscf.scf import hf
from pyscf.dft import rks
import time

def _scf_common_init(mf, Ne, Norb, mol0=None, oei=None, tei=None, ovlp=1, dm0=None, coredm=0.0, ao2sub=1.0, mf_method='LDA'):

        mf.Ne = Ne
        mf.Norb = Norb

        mf.oei = oei
        mf.tei = tei
        mf.ovlp = ovlp
        mf.dm_guess = dm0
        mf.coredm = coredm
        mf.ao2sub = ao2sub
        mf.method = mf_method.lower()

        mf.elec_energy = 0.0
        mf.rdm1 = None

	mol = mol0
        if(mol is None):
            #what molecule does not matter
            mol = gto.Mole()
            mol.build(max_memory=4000, verbose=4 )
            mol.atom.append(('C', (0, 0, 0)))

        #adjust number of electrons
        mol.nelectron = mf.Ne

        if(mf.tei is not None):
            mol.incore_anyway = True

	return mol

def init_guess_by_minao(mf, mol=None, ao2sub = None):
    if mol is None: mol = mf.mol
    if ao2sub is None: ao2sub = mf.ao2sub
    dm_ao = hf.init_guess_by_minao(mol)
    s = hf.get_ovlp(mol)
    dm_sub = tools.dm_ao2loc(dm_ao, s, ao2sub)
    return dm_sub

def get_ovlp(mf, mol = None):

    if mol is None: mol = mf.mol

    s = 0.0
    if(isinstance(mf.ovlp, np.ndarray)):
	s = mf.ovlp 
    elif(mf.ovlp == 1):
	s = np.eye( mf.Norb )
    else:
        s = hf.get_ovlp(mol)

    return s


def get_hcore(mf, mol = None):

    if mol is None: mol = mf.mol

    h = 0.0
    if(mf.oei is not None):
	h = mf.oei
    else:
	h = hf.get_hcore(mol)

    return h


def energy_tot(mf, dm=None, h1e=None, vhf=None, mo_occ = None):
    r'''Total Hartree-Fock energy, electronic part plus nuclear repulstion
    See :func:`scf.hf.energy_elec` for the electron part
    '''
    e_tot = mf.energy_elec(dm, h1e, vhf, mo_occ)[0]# + mf.energy_nuc()
    return e_tot.real

def runscf(mf):

    mf.kernel(mf.dm_guess)
    if ( mf.converged == False ):
	print "scf not converged!"
        #raise Exception("scf not converged!")

    rdm1 = mf.make_rdm1()
    mf.rdm1 = 0.5*(rdm1.T + rdm1)
    mf.elec_energy = mf.energy_elec(mf.rdm1)[0]


def get_occ(mf, mo_energy=None, mo_coeff=None, smear_sigma = None):

    if mo_energy is None: mo_energy = mf.mo_energy
    if smear_sigma is None: smear_sigma = mf.smear_sigma

    nmo = mo_energy.size
    mo_occ = np.zeros(nmo)
    Nocc = mf.mol.nelectron // 2

    e_idx = np.argsort(mo_energy)
    e_sort = mo_energy[e_idx]
    mo_occ[e_idx[:Nocc]] = 1.0

    e_homo = e_sort[Nocc-1]
    e_lumo = e_sort[Nocc]
    print 'HOMO:',e_homo, 'LUMO:', e_lumo

    e_fermi = e_homo

    if(smear_sigma > 1e-9):
        e_fermi, mo_occ = pyscf_rhf.find_efermi(mo_energy, smear_sigma, Nocc, nmo)

    mo_occ *= 2.0  #closed shell

    ne = np.sum(mo_occ)
    Ne_error = ne - mf.mol.nelectron
    if(abs(Ne_error) > 1e-12):
        print 'Ne error = ', Ne_error
    print e_fermi
    np.set_printoptions(precision=3)
    flag = mo_occ > 1e-3
    print mo_occ[flag]
    np.set_printoptions()

    return mo_occ


class rhf_pyscf(hf.RHF):

    '''
        subspace rhf
        wrapper for scf.hf module of pyscf
    '''

    def __init__(self, Ne, Norb, mol=None, oei=None, tei=None, ovlp=1, dm0=None, \
		 coredm=0.0, ao2sub=1.0, level_shift=0.0, smear_sigma = 0.0):

        mol = _scf_common_init(self, Ne, Norb, mol, oei, tei, ovlp, dm0, coredm, ao2sub, mf_method='HF')
	self.smear_sigma = smear_sigma
        hf.RHF.__init__(self, mol)
	self.level_shift = level_shift

        if(self.tei is not None):
            self._eri = ao2mo.restore(8, self.tei, self.Norb)


    def energy_elec(mf, dm=None, h1e=None, vhf=None, mo_occ=None):

        if dm is None: dm = mf.make_rdm1()
        if h1e is None: h1e = mf.get_hcore()
        if vhf is None: vhf = mf.get_veff(mf.mol, dm)
	if mo_occ is None: mo_occ = mf.mo_occ

        e1 = np.einsum('ji,ji', h1e.conj(), dm).real
        e_coul = np.einsum('ji,ji', vhf.conj(), dm).real * .5
        es = entropy_corr(mo_occ, mf.smear_sigma)
        tot_e = e1 + e_coul + es

        return tot_e, e_coul

    
    get_ovlp = get_ovlp
    get_hcore = get_hcore
    energy_tot = energy_tot
    runscf = runscf
    get_occ = get_occ
    init_guess_by_minao = init_guess_by_minao

class rks_pyscf(rks.RKS): 

    '''
	subspace rks
	wrapper for dft.rks module of pyscf
    '''

    def __init__(self, Ne, Norb, mf_method, mol=None, oei=None, tei=None, ovlp=1, dm0=None,\
		 coredm=0.0, ao2sub=1.0, level_shift=0.0, smear_sigma = 0.0):

	mol = _scf_common_init(self, Ne, Norb, mol, oei, tei, ovlp, dm0, coredm, ao2sub, mf_method)
	rks.RKS.__init__(self, mol, smear_sigma = smear_sigma)
	self.xc = self.method

	self.level_shift = level_shift

	'''
	self.grids.atom_grid = {'H': (50,194), 'O': (50,194), 'F': (50,194)}
        self.grids.prune = dft.gen_grid.sg1_prune
        self.grids.radi_method = dft.radi.gauss_chebyshev
        self.grids.atomic_radii = dft.radi.SG1RADII
        self.grids.radii_adjust = None
        #self.small_rho_cutoff = 1e-9
	'''

	if(self.tei is not None):
            self._eri = ao2mo.restore(8, self.tei, self.Norb)


    get_ovlp = get_ovlp
    get_hcore = get_hcore
    runscf = runscf
    energy_tot = energy_tot
    #get_occ = get_occ
    init_guess_by_minao = init_guess_by_minao

    def energy_elec(mf, dm=None, h1e=None, vhf=None, mo_occ=None):

        if dm is None: dm = mf.make_rdm1()
        if h1e is None: h1e = mf.get_hcore()
        if vhf is None or getattr(vhf, 'ecoul', None) is None:
            vhf = mf.get_veff(mf.mol, dm)
	if mo_occ is None: mo_occ = mf.mo_occ
    
        e1 = np.einsum('ij,ji', h1e, dm).real
        es = entropy_corr(mo_occ, mf.smear_sigma)
        tot_e = e1 + vhf.ecoul + vhf.exc + es
    
        return tot_e, vhf.ecoul+vhf.exc


    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):

	#t0 = (time.clock(), time.time())

	if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()

	ao2sub = self.ao2sub
	coredm = self.coredm

	use_mo = False
	n_core_elec = 0.0
	if (isinstance(coredm, np.ndarray) and coredm.ndim == 2):
	    s1e_ao = mol.intor_symmetric('int1e_ovlp')
	    n_core_elec = np.trace(np.dot(coredm, s1e_ao))
	    use_mo = False
	else:
	    use_mo = True

	use_mo = False #debug
	dm_ao = coredm + tools.dm_sub2ao(np.asarray(dm), ao2sub)
	
	if (hasattr(dm, 'mo_coeff') and use_mo):
            mo_coeff_sub = dm.mo_coeff
            mo_occ_sub = dm.mo_occ

	    mo_coeff_ao = tools.mo_sub2ao(mo_coeff_sub, ao2sub)
	    mo_occ_ao = mo_occ_sub
	    dm_ao = lib.tag_array(dm_ao, mo_coeff=mo_coeff_ao, mo_occ=mo_occ_ao)
	
	n, exc, vxc_ao, hyb = get_vxc(self, mol, dm_ao, n_core_elec = n_core_elec)
	vxc = tools.op_ao2sub(vxc_ao, ao2sub)

	vj = None
	vk = None
        if abs(hyb) < 1e-10:
            if (self._eri is None and self.direct_scf and
                getattr(vhf_last, 'vj', None) is not None):
                ddm = np.asarray(dm) - np.asarray(dm_last)
                vj = self.get_jk(mol, ddm, hermi)[0]
                vj += vhf_last.vj
            else:
                vj = self.get_jk(mol, dm, hermi)[0]
            vxc += vj
        else:
            if (self._eri is None and self.direct_scf and
                getattr(vhf_last, 'vk', None) is not None):
                ddm = np.asarray(dm) - np.asarray(dm_last)
                vj, vk = self.get_jk(mol, ddm, hermi)
                vj += vhf_last.vj
                vk += vhf_last.vk
            else:
                vj, vk = self.get_jk(mol, dm, hermi)
            vxc += vj - vk * (hyb * .5)
            exc -= np.einsum('ij,ji', dm, vk) * .5 * hyb*.5

        ecoul = np.einsum('ij,ji', dm, vj) * .5

	vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)

	#t1 = tools.timer("get_veff",t0)

	return vxc



def get_vxc(ks, mol, dm, n_core_elec=0.0, hermi=1):

    ground_state = (isinstance(dm, np.ndarray) and dm.ndim == 2)
    if(not ground_state):
        raise Exception("fatal error")

    if ks.grids.coords is None:
        ks.grids.build(with_non0tab=True)
        if ks.small_rho_cutoff > 1e-20 and ground_state:
            # Filter grids the first time setup grids
	    #t0 = (time.clock(), time.time())
            ks.grids = prune_small_rho_grids_(ks, mol, dm, ks.grids,n_core_elec=n_core_elec)
	    #t1 = tools.timer("prune grid",t0)

    if hermi == 2:  # because rho = 0
        n, exc, vxc = 0, 0, 0
    else:
        n, exc, vxc = ks._numint.nr_rks(mol, ks.grids, ks.xc, dm)

    hyb = ks._numint.hybrid_coeff(ks.xc, spin=mol.spin)

    return n, exc, vxc, hyb


def prune_small_rho_grids_(ks, mol, dm, grids, n_core_elec = 0.0):
    n, idx = ks._numint.large_rho_indices(mol, dm, grids, ks.small_rho_cutoff)
    print 'No. of grids = ',grids.weights.size

    error_tol = 0.001
    nelec = mol.nelectron+n_core_elec
    print 'if prune density:',n, nelec, abs(n-nelec), error_tol
    if abs(n - nelec) < error_tol:
        print 'No. of dropped grids = ',grids.weights.size - np.count_nonzero(idx)
        grids.coords  = np.asarray(grids.coords [idx], order='C')
        grids.weights = np.asarray(grids.weights[idx], order='C')
        grids.non0tab = grids.make_mask(mol, grids.coords)
    return grids



def entropy_corr(mo_occ, smear_sigma):

    if mo_occ is None:
	return 0.0

    S = 0.0
    if(smear_sigma >= 1e-9):
	nmo = mo_occ.size
        for i in range(nmo):
            occ_i = mo_occ[i]/2.0 #closed shell
            if(occ_i > 1e-16 and occ_i < 1.0-1e-16):
                S += occ_i * np.log(occ_i) + (1.0-occ_i) * np.log(1.0-occ_i)
            else:
                S += 0.0

    energy = 2.0*S*smear_sigma
    return energy
