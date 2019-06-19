import sys
import numpy as np
from pydmfet import tools
from pydmfet.dfet_ao import scf
from scipy import optimize
from pydmfet.libcpp import oep_hess
from pydmfet.opt import newton
from functools import reduce

def init_density_partition(oep, umat=None, mol1=None, mol2=None, mf_method=None):

    if(umat is None): umat = oep.umat
    if(mol1 is None): mol1 = oep.mol_frag
    if(mol2 is None): mol2 = oep.mol_env
    if(mf_method is None): mf_method = oep.mf_method

    if(oep.use_sub_umat):
        s=oep.s
        ao2sub = oep.ao2sub
        umat = reduce(np.dot,(s,ao2sub,umat,ao2sub.T,s))

    mf_frag = scf.EmbedSCF(mol1, umat+oep.vnuc_bound_frag, oep.smear_sigma)
    mf_frag.xc = mf_method
    mf_frag.scf(dm0 = oep.P_imp)
    FRAG_1RDM = mf_frag.make_rdm1()
    mo_frag = mf_frag.mo_coeff
    occ_frag = mf_frag.mo_occ
    oep.frag_occ = occ_frag

    tools.mo_molden(mol1,mo_frag,'frag_mo.molden')

    mf_env = scf.EmbedSCF(mol2, umat+oep.vnuc_bound_env, oep.smear_sigma)
    mf_env.xc = mf_method
    mf_env.scf(dm0 = oep.P_bath)
    ENV_1RDM = mf_env.make_rdm1()
    mo_env = mf_env.mo_coeff
    occ_env = mf_env.mo_occ
    oep.env_occ = occ_env

    tools.mo_molden(mol2,mo_env,'env_mo.molden')

    diffP = (FRAG_1RDM + ENV_1RDM - oep.P_ref)
    print ('|P_imp+P_bath-P_ref| = ', np.linalg.norm(diffP))
    print ('max(P_imp+P_bath-P_ref) = ', np.amax(np.absolute(diffP)))

    print ('mo orthogonality:')
    print (reduce(np.dot,(mo_frag[:,occ_frag>1e-8].T,oep.s,mo_env[:,occ_env>1e-8])))

    return (FRAG_1RDM, ENV_1RDM)


def grad_umat_sub(diffP,s,ao2sub,dim):

    sc = np.dot(s,ao2sub)
    grad = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            tmp = np.dot(diffP,sc[:,i])
            grad[i,j] = np.dot(sc[:,j].T,tmp)

    return grad

class OEPao:

    def __init__(self, dfet, params):

        self.params = params
        self.umat = dfet.umat
        self.dim = self.umat.shape[0]

        self.s = dfet.mf_full.get_ovlp(dfet.mol)
        self.smear_sigma = dfet.smear_sigma

        self.mol_frag = dfet.mol_frag
        self.mol_env = dfet.mol_env

        self.Ne_frag = dfet.Ne_frag
        self.Ne_env = dfet.Ne_env

        self.vnuc_bound_frag = dfet.vnuc_bound_frag
        self.vnuc_bound_env  = dfet.vnuc_bound_env

        self.mf_method = dfet.mf_method
        self.P_ref = dfet.P_ref
        self.P_imp = dfet.P_imp
        self.P_bath = dfet.P_bath

        self.frag_occ = None
        self.env_occ = None
        self.fixed_occ = False

        self.gtol_dyn = self.params.gtol

        dim = self.dim
        if(self.umat is None): self.umat = np.zeros([dim,dim])

        self.use_sub_umat = False
        self.ao2sub=None
        if( hasattr(dfet, 'use_umat_ao')):
            if(dfet.use_umat_ao):
                self.use_sub_umat = True
                self.ao2sub = dfet.ao2sub[:,:self.dim]
                #self.umat = reduce(np.dot,(s,self.ao2sub,self.umat_sub,self.ao2sub.T,s))





    def kernel(self):

        algorithm = self.params.algorithm
        if(algorithm == '2011'):
            self.umat = self.oep_base(self.umat)
        elif(algorithm == 'split'):
            self.umat = self.oep_loop(self.umat)

        self.P_imp, self.P_bath = self.verify_scf(self.umat)
        #tools.MatPrint(self.P_imp-self.P_bath,"P_imp-P_bath")

        return self.umat


    init_density_partition = init_density_partition
    verify_scf = init_density_partition


    def oep_loop(self, _umat):

        umat = _umat.copy()
        self.P_imp, self.P_bath = self.init_density_partition(umat)

        diffP=(self.P_imp+self.P_bath-self.P_ref)
        gtol = np.amax(np.absolute(diffP))
        self.gtol_dyn = max(gtol/5.0,self.params.gtol)


        self.fixed_occ = False
        
        threshold = self.params.diffP_tol
        maxit = self.params.outer_maxit
        it = 0
        while it < maxit:
            it += 1
            print (" OEP iteration ", it)

            P_imp_old = self.P_imp.copy()
            P_bath_old = self.P_bath.copy()

            umat = self.oep_base(umat, True)
            umat_loc = umat

            if(self.use_sub_umat):
                s=self.s
                ao2sub = self.ao2sub
                umat_loc = reduce(np.dot,(s,ao2sub,umat,ao2sub.T,s))

            #########
            mf_frag = scf.EmbedSCF_nonscf(self.mol_frag, self.P_imp, umat_loc+self.vnuc_bound_frag, self.smear_sigma)
            mf_frag.xc = self.mf_method
            mf_frag.scf(dm0=self.P_imp)
            self.P_imp = mf_frag.make_rdm1()
            self.frag_occ = mf_frag.mo_occ

            mf_env = scf.EmbedSCF_nonscf(self.mol_env, self.P_bath, umat_loc+self.vnuc_bound_env, self.smear_sigma)
            mf_env.xc = self.mf_method
            mf_env.scf(dm0=self.P_bath)
            self.P_bath = mf_env.make_rdm1()
            self.env_occ = mf_env.mo_occ
            #########


            #tools.MatPrint(self.P_imp,"P_imp")
            #tools.MatPrint(self.P_bath,"P_bath")
            tools.MatPrint(umat,"umat")
            #np.savetxt("umat.gz",umat)

            diffP_imp = self.P_imp - P_imp_old
            diffP_bath = self.P_bath - P_bath_old
            gmax_imp = np.amax(np.absolute(diffP_imp))
            gmax_bath = np.amax(np.absolute(diffP_bath))
            print ("diffP_max_imp, diffP_max_bath ")
            print (gmax_imp, gmax_bath)

            sys.stdout.flush()
            if(gmax_imp < threshold and gmax_bath < threshold ):
                break

            P_imp_old = None
            P_bath_old = None

            '''
            mf_frag = scf.EmbedSCF_nonscf(self.mol_frag, self.P_imp, umat_loc+self.vnuc_bound_frag, self.smear_sigma)
            mf_frag.xc = self.mf_method
            mf_frag.scf()
            P_imp = mf_frag.make_rdm1()
            self.frag_occ = mf_frag.mo_occ

            mf_env = scf.EmbedSCF_nonscf(self.mol_env, self.P_bath, umat_loc+self.vnuc_bound_env, self.smear_sigma)
            mf_env.xc = self.mf_method
            mf_env.scf()
            P_bath = mf_env.make_rdm1()
            self.env_occ = mf_env.mo_occ
            '''
            diffP=(self.P_imp+self.P_bath-self.P_ref)
            gtol = np.amax(np.absolute(diffP))
            self.gtol_dyn = max(gtol/5.0,self.params.gtol)
            

        return umat


    def oep_base(self, umat, nonscf = False):

        P_ref = self.P_ref
        dim = umat.shape[0]

        x = tools.mat2vec(umat, dim)
        #x = tools.mat2vec_hchain(umat,dim)

        _args = [P_ref, dim, nonscf]
        _args = tuple(_args)

        opt_method = self.params.opt_method
        result = None
        if( opt_method == 'BFGS' or opt_method == 'L-BFGS-B'):
            result = self.oep_bfgs(x, _args)
            x = result.x
        elif( opt_method == 'trust-krylov' or opt_method == 'trust-ncg' or opt_method == 'trust-exact'):
            result = self.oep_trust(x, _args)
            x = result.x
        elif( opt_method == 'newton'):
            x = self.oep_newton(x,_args)

        umat = tools.vec2mat(x, dim)
        #umat = tools.vec2mat_hchain(x,dim)

        return umat

    def oep_newton(self, x, _args):

        ftol = self.params.ftol
        #gtol = self.params.gtol
        gtol = self.gtol_dyn
        print ('gtol = ', gtol)
        maxit = self.params.maxit
        svd_thresh = self.params.svd_thresh

        res = newton(self.cost_hess_wuyang,x,args=_args,ftol=ftol,gtol=gtol,maxit=maxit,svd_thresh=svd_thresh)

        return res


    def oep_trust(self, x, _args):

        #gtol = self.params.gtol
        gtol = self.gtol_dyn
        maxit = self.params.maxit
        algorithm = self.params.opt_method

        res = optimize.minimize(self.cost_wuyang, x, args=_args, method=algorithm,jac=True, hess=self.hess_wuyang, \
                       options={'maxiter': maxit, 'gtol':gtol, 'disp': True, 'initial_trust_radius':1.0, 'max_trust_radius':10.0})

        return res


    def oep_bfgs(self, x, _args):

        maxit = self.params.maxit
        gtol = self.params.gtol
        #gtol = self.gtol_dyn
        print ('gtol = ', gtol)
        ftol = self.params.ftol
        algorithm = self.params.opt_method

        result = optimize.minimize(self.cost_wuyang,x,args=_args,method=algorithm, jac=True, \
                                   options={'disp': True, 'maxiter': maxit, 'gtol':gtol, 'ftol':ftol, 'maxcor':10} )

        return result


    def cost_wuyang(self, x, P_ref, dim, nonscf):


        umat = tools.vec2mat(x, dim)
        #umat = tools.vec2mat_hchain(x,dim)
        print ("|umat| = ", np.linalg.norm(umat))

        if(self.use_sub_umat):
            s=self.s
            ao2sub = self.ao2sub
            umat = reduce(np.dot,(s,ao2sub,umat,ao2sub.T,s))

        if(nonscf == False):  #normal SCF
            tools.MatPrint(umat,"umat")
            #np.savetxt("umat.gz",umat)

            mf_frag = scf.EmbedSCF(self.mol_frag, umat+self.vnuc_bound_frag, self.smear_sigma)
            mf_frag.xc = self.mf_method
            mf_frag.scf()
            FRAG_energy = mf_frag.energy_elec()[0]
            FRAG_1RDM = mf_frag.make_rdm1()

            mf_env = scf.EmbedSCF(self.mol_env, umat+self.vnuc_bound_env, self.smear_sigma)
            mf_env.xc = self.mf_method
            mf_env.scf()
            ENV_energy = mf_env.energy_elec()[0]
            ENV_1RDM = mf_env.make_rdm1()

        else:  #non-self-consistent SCF
            mf_frag = scf.EmbedSCF_nonscf(self.mol_frag, self.P_imp, umat+self.vnuc_bound_frag, self.smear_sigma,self.fixed_occ,self.frag_occ)
            mf_frag.xc = self.mf_method
            mf_frag.scf()
            FRAG_energy = mf_frag.energy_elec()[0]
            FRAG_1RDM = mf_frag.make_rdm1()
            print ("orbital energy:")
            print (mf_frag.mo_energy)

            mf_env = scf.EmbedSCF_nonscf(self.mol_env, self.P_bath, umat+self.vnuc_bound_env, self.smear_sigma,self.fixed_occ,self.env_occ)
            mf_env.xc = self.mf_method
            mf_env.scf()
            ENV_energy = mf_env.energy_elec()[0]
            ENV_1RDM = mf_env.make_rdm1()
            print ("orbital energy:")
            print (mf_env.mo_energy)


        diffP = FRAG_1RDM + ENV_1RDM - P_ref
        energy = FRAG_energy + ENV_energy - np.trace(np.dot(P_ref,umat))


        if(self.use_sub_umat):
            diffP = grad_umat_sub(diffP,self.s,self.ao2sub,dim)

        #diffP = 2.0 * diffP
        #for i in range(diffP.shape[0]):
        #    diffP[i,i]=diffP[i,i] / 2.0
        

        grad = tools.mat2vec(diffP, dim)
        #grad = tools.mat2vec_hchain(diffP, dim)
        grad = -1.0 * grad

        print ("2-norm (grad),       max(grad):")
        print (np.linalg.norm(grad), ", ", np.amax(np.absolute(grad)))

        f = -energy
        print ('W = ', f)

        return (f, grad)


    def hess_wuyang(self, x, P_ref, dim, nonscf):

        size = dim*(dim+1)//2
        umat = tools.vec2mat(x, dim)
        print ("|umat| = ", np.linalg.norm(umat))

        if(nonscf):
            mf_frag = scf.EmbedSCF_nonscf(self.mol_frag, self.P_imp, umat+self.vnuc_bound_frag, self.smear_sigma,self.fixed_occ,self.frag_occ)
            mf_frag.xc = self.mf_method
            mf_frag.scf()
            hess_frag = oep_hess(mf_frag.mo_coeff,mf_frag.mo_energy,mf_frag.mo_occ,size,dim, self.Ne_frag//2, self.smear_sigma)

            mf_env = scf.EmbedSCF_nonscf(self.mol_env, self.P_bath, umat+self.vnuc_bound_env, self.smear_sigma,self.fixed_occ,self.env_occ)
            mf_env.xc = self.mf_method
            mf_env.scf()
            hess_env = oep_hess(mf_env.mo_coeff,mf_env.mo_energy,mf_env.mo_occ,size,dim, self.Ne_env//2, self.smear_sigma) 

            hess = hess_frag + hess_env

        else:
            raise Exception("NYI")


        return hess



    def cost_hess_wuyang(self,x,P_ref,dim,nonscf,calc_hess=False):

        size = dim*(dim+1)//2
        umat = tools.vec2mat(x, dim)

        if(nonscf):
            mf_frag = scf.EmbedSCF_nonscf(self.mol_frag, self.P_imp, umat+self.vnuc_bound_frag, self.smear_sigma,self.fixed_occ,self.frag_occ)
            mf_frag.xc = self.mf_method
            mf_frag.scf()
            FRAG_energy = mf_frag.energy_elec()[0]
            FRAG_1RDM = mf_frag.make_rdm1()
            print ("orbital energy:")
            print (mf_frag.mo_energy)

            mf_env = scf.EmbedSCF_nonscf(self.mol_env, self.P_bath, umat+self.vnuc_bound_env, self.smear_sigma,self.fixed_occ,self.env_occ)
            mf_env.xc = self.mf_method
            mf_env.scf()
            ENV_energy = mf_env.energy_elec()[0]
            ENV_1RDM = mf_env.make_rdm1()
            print ("orbital energy:")
            print (mf_env.mo_energy)

            if(calc_hess):
                hess_frag = oep_hess(mf_frag.mo_coeff,mf_frag.mo_energy,mf_frag.mo_occ,size,dim, self.Ne_frag//2, self.smear_sigma)
                hess_env = oep_hess(mf_env.mo_coeff,mf_env.mo_energy,mf_env.mo_occ,size,dim, self.Ne_env//2, self.smear_sigma)
                hess = hess_frag + hess_env

        else:
            raise Exception("NYI")

        diffP = FRAG_1RDM + ENV_1RDM - P_ref
        energy = FRAG_energy + ENV_energy - np.trace(np.dot(P_ref,umat))

        grad = tools.mat2vec(diffP, dim)
        #grad = tools.mat2vec_hchain(diffP, dim)
        grad = -1.0 * grad

        print ("2-norm (grad),       max(grad):")
        print (np.linalg.norm(grad), ", ", np.amax(np.absolute(grad)))

        f = -energy
        print ('W = ', f)

        #test
        #f = np.linalg.norm(grad)

        if(calc_hess):
            return (f,grad,hess)
        else:
            return (f,grad)

