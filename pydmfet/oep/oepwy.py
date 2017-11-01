import numpy as np
from scipy import optimize
from pydmfet import qcwrap,tools

class OEPWY:

    def __init__(self, embedobj, params):

        self.P_ref = embedobj.P_ref_sub
        self.umat = embedobj.umat
        self.dim = embedobj.dim_sub
        self.Ne_frag = embedobj.Ne_frag
        self.Ne_env = embedobj.Ne_env 
        self.loc2sub = embedobj.loc2sub
        self.impAtom = embedobj.impAtom
        self.core1PDM_loc = embedobj.core1PDM_loc
        self.ints = embedobj.ints

        self.params = params

    def kernel(self):

        dim = self.dim
        if(self.umat is None):
            self.umat = np.zeros([dim,dim],dtype=float)

        self.umat = self.oep_bfgs()


    def oep_bfgs(self):

        umat = self.umat.copy()
        dim = self.dim
        x = tools.mat2vec(umat, dim)

        Ne_frag = self.Ne_frag
        Ne_env = self.Ne_env

        loc2sub = self.loc2sub
        impAtom = self.impAtom
        core1PDM_loc = self.core1PDM_loc

        ints = self.ints

        subKin = ints.frag_kin_sub( impAtom, loc2sub, dim )
        subVnuc1 = ints.frag_vnuc_sub( impAtom, loc2sub, dim )
        subVnuc2 = ints.frag_vnuc_sub( 1-impAtom, loc2sub, dim )
        subCoreJK = ints.dmet_fock( loc2sub, dim, core1PDM_loc )
        subTEI = ints.dmet_tei( loc2sub, dim )

        P_ref = self.P_ref

        _args = (dim,Ne_frag,Ne_env,subKin,subTEI,subVnuc1,subVnuc2,subCoreJK,P_ref)

        opt_method = self.params.opt_method
        maxit = self.params.maxit
        gtol = self.params.gtol

        result = optimize.minimize( self.cost_wuyang, x, args=_args, method=opt_method, jac=True, options={'disp': True, 'maxiter': maxit, 'gtol':gtol} )

        umat_flat = result.x
        umat = tools.vec2mat(umat_flat, dim)

        return umat

    def cost_wuyang(self, x, dim, Ne_frag, Ne_env, subKin, subTEI, subVnuc1, subVnuc2, subCoreJK, P_ref):

        umat = tools.vec2mat(x, dim)
        print "umat"
        print umat

        #guess density
        subOEI1 = subKin+subVnuc1+umat
        subOEI2 = subKin+subVnuc2+umat
        energy_tmp1, dm_tmp1 = qcwrap.pyscf_rhf.scf( subOEI1, subTEI, dim, Ne_frag)
        energy_tmp2, dm_tmp2 = qcwrap.pyscf_rhf.scf( subOEI2, subTEI, dim, Ne_env)

        FRAG_energy, FRAG_1RDM = qcwrap.pyscf_rhf.scf( subOEI1+subVnuc2, subTEI, dim, Ne_frag, OneDM0=dm_tmp1)
        ENV_energy, ENV_1RDM   = qcwrap.pyscf_rhf.scf( subOEI2+subVnuc1, subTEI, dim, Ne_env,  OneDM0=dm_tmp2)

        print "frag/env densities"
        print FRAG_1RDM 
        print ENV_1RDM
        print "******************"


        diffP = FRAG_1RDM + ENV_1RDM - P_ref
        energy = FRAG_energy + ENV_energy - np.trace(np.dot(P_ref,umat))
        #print energy
        grad = tools.mat2vec(diffP, dim)
        grad = -1.0 * grad
        print "2-norm (grad),       max(grad):"
        print np.linalg.norm(grad), ", ", np.amax(np.absolute(grad))
        #print diffP
        #print umat
        return (-energy, grad)

