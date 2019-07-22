import numpy as np
from pyscf import lo
from pyscf.tools import molden, cubegen
from functools import reduce
from math import sqrt
from pydmfet import tools
from pydmfet.qcwrap.pyscf_rks_ao import rks_ao

class proj_embed:

    def __init__(self, mf_full, cluster, Ne_env = None, loc_method = 'PM', pop_method = 'meta_lowdin', pm_exponent=2, mu = 1e5):

        self.mf_full = mf_full
        self.mol = mf_full.mol
        self.xc_func = self.mf_full.xc
        self.cluster = cluster
        self.env = 1-cluster
        self.Ne_env = Ne_env

        self.loc_method = loc_method
        self.pop_method = pop_method
        self.pm_exponent = pm_exponent
        self.mu = mu

        self.smear_sigma = 0.0        
        smear_sigma = getattr(self.mf_full, "smear_sigma", None)
        if(smear_sigma is not None):
            self.smear_sigma = self.mf_full.smear_sigma

        self.mo_lo = None
        self.pop_lo = None
        self.P_env = None
        self.P_ref = self.mf_full.make_rdm1()


    def make_frozen_orbs(self, norb = None):

        mf = self.mf_full
        mol = mf.mol
        loc_method = self.loc_method
        pop_method = self.pop_method
        pm_exponent = self.pm_exponent

        if(norb is None):
            norb = mol.nelectron // 2

        mo_lo = None
        if(loc_method.upper() == 'PM'):
            pm = lo.pipek.PM(mol)
            pm.pop_method = pop_method
            pm.exponent = pm_exponent
            mo_lo = pm.kernel(mf.mo_coeff[:,:norb], verbose=4)
        elif(loc_method.upper() == 'BOYS'):
            boys = lo.boys.Boys(mol)
            mo_lo = boys.kernel(mf.mo_coeff[:,:norb], verbose=4)
        else:
            raise NotImplementedError('loc_method %s' % loc_method)

        s = mol.intor_symmetric('int1e_ovlp')
        nbas = mol.nao_nr()
        dm_lo = np.empty((norb,nbas,nbas))
        for i in range(norb):
            dm_lo[i] = np.outer(mo_lo[:,i],mo_lo[:,i])

        pop = np.zeros((norb))
        for i in range(norb):
            for iatom, (b0, b1, p0, p1) in enumerate(mol.offset_nr_by_atom()):
                if(self.env[iatom] == 1):
                    pop[i] += np.trace(np.dot(dm_lo[i,p0:p1,:],s[:,p0:p1]))

        #tools.VecPrint(pop,"Mulliken popupaltion")

        ind = np.argsort(-pop)
        pop = pop[ind]
        mo_lo[:,:norb] = mo_lo[:,ind]

        tools.VecPrint(pop,"sorted Mulliken popupaltion: 1.0 for fully occupied")
        with open( 'mo_lo.molden', 'w' ) as thefile:
            molden.header(mol, thefile)
            molden.orbital_coeff(mol, thefile, mo_lo)


        self.mo_lo = mo_lo
        self.pop_lo = pop
        
        return (mo_lo,pop)


    def embedding_potential(self):

        mo_lo = self.mo_lo
        pop = self.pop_lo
        mol = self.mf_full.mol
        s = mol.intor_symmetric('int1e_ovlp')

        norb_frozen = 0
        Ne_env = self.Ne_env
        if Ne_env is None:
            is_env = pop > 0.4
            norb_frozen = np.sum(is_env)
        elif Ne_env > 1:
            norb_frozen = Ne_env//2

        print("{n:2d} enviroment orbitals kept frozen.".format(n=norb_frozen) )
        if(norb_frozen>0):
            self.P_env = np.dot(mo_lo[:,:norb_frozen], mo_lo[:,:norb_frozen].T)
            self.P_env = self.P_env + self.P_env.T
        else:
            raise RuntimeError("There is no frozen orbital!")

        proj_op = 0.5*self.mu * reduce(np.dot, (s,self.P_env,s))

        proj_ks = rks_ao(mol, xc_func=self.xc_func, coredm=self.P_env, vext_1e=proj_op, smear_sigma=self.smear_sigma)
        proj_ks.kernel(dm0 = self.P_ref-self.P_env)

        P_frag = proj_ks.make_rdm1()
        print ("level shift energy:" , np.trace(np.dot(P_frag, proj_op)) )

        P_diff = P_frag + self.P_env - self.P_ref
        print ('|P_frag + P_bath - P_ref| / N = ', np.linalg.norm(P_diff)/P_diff.shape[0] )
        print ('max(P_frag + P_bath - P_ref) = ', np.amax(np.absolute(P_diff)))

        cubegen.density(mol, "dens_error.cube", P_diff, nx=100, ny=100, nz=100)
        cubegen.density(mol, "dens_frag.cube", P_frag, nx=100, ny=100, nz=100)
        cubegen.density(mol, "dens_env.cube", self.P_env, nx=100, ny=100, nz=100)

        return None


    def correction_energy(self):


        return None
