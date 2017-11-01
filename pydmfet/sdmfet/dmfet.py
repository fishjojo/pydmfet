import numpy as np
from pydmfet import subspac,oep

class DMFET:

    def __init__(self, ints, cluster, impAtom, Ne_frag, oep_params = None, ecw_method = 'HF', is_dfet = False):

        self.ints = ints
        self.cluster = cluster
        self.impAtom = impAtom

        self.Ne_frag = Ne_frag

        self.dim_frag = np.sum(self.cluster)
        self.dim_env = self.cluster.size - self.dim_frag



        self.ecw_method = ecw_method
        self.is_dfet = is_dfet

        #construct subspace
        self.OneDM_loc = self.ints.build_1pdm_loc()
        self.dim_imp, self.dim_bath, self.Occupations, self.loc2sub = subspac.construct_subspace(self.OneDM_loc, self.cluster)
        self.dim_sub = self.dim_imp + self.dim_bath

        #construct core determinant
        self.core1PDM_loc, self.Nelec_core = subspac.build_core(self.Occupations, self.loc2sub)
        self.Ne_env = self.ints.Nelec - self.Ne_frag - self.Nelec_core

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

