import numpy as np
from pydmfet import subspac,oep

class DMFET:

    def __init__(self, ints, cluster, impAtom, Ne_frag, charge=[0,0], spin=[0,0], sub_threshold = 1e-13, oep_params = None, ecw_method = 'HF', do_dfet = False):

        self.ints = ints
        self.cluster = cluster
        self.impAtom = impAtom

        self.Ne_frag = Ne_frag

        self.dim_frag = np.sum(self.cluster)
        self.dim_env = self.cluster.size - self.dim_frag

	self.charge = charge
	self.spin   = spin

        self.ecw_method = ecw_method
        self.do_dfet = do_dfet
	self.sub_threshold = sub_threshold

        #construct subspace
        self.OneDM_loc = self.ints.build_1pdm_loc()
        self.dim_imp, self.dim_bath, self.Occupations, self.loc2sub, eignimp, eignbath = subspac.construct_subspace(self.OneDM_loc, self.cluster, self.sub_threshold)
        self.dim_sub = self.dim_imp + self.dim_bath

	print 'dimension of subspace'
	print self.dim_imp,  self.dim_bath

        #construct core determinant
	idx = self.dim_frag + self.dim_bath
        self.core1PDM_loc, self.Nelec_core, Norb_imp_throw = subspac.build_core(self.Occupations, self.loc2sub, idx)
	self.Ne_frag = self.Ne_frag - Norb_imp_throw*2
        self.Ne_env = self.ints.Nelec - self.Ne_frag - self.Nelec_core
	
	print 'Ne_frag, Ne_env, Ne_core'
	print self.Ne_frag, self.Ne_env, self.Nelec_core

        self.umat = None
        dim = self.dim_sub
        loc2sub = self.loc2sub
        self.P_ref_sub = np.dot(np.dot(loc2sub[:,:dim].T ,self.OneDM_loc - self.core1PDM_loc), loc2sub[:,:dim]) 

        self.oep_params = oep_params
        if(self.oep_params is None):
            self.oep_params = oep.OEPparams()

    def calc_umat(self):
        
        myoep = oep.OEP(self,self.oep_params)
        myoep.kernel()
        self.umat = myoep.umat


    def embedding_potential(self):

        if(self.umat is None):
            self.calc_umat()
            
        return self.umat



    def total_energy(self):
        energy = 0.0
        return energy

