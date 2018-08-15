import sys
import numpy as np
from pydmfet import tools
from pydmfet.dfet_ao import scf
from scipy import optimize


def init_density_partition(oep, umat=None, mol1=None, mol2=None, mf_method=None):

    if(umat is None): umat = oep.umat
    if(mol1 is None): mol1 = oep.mol_frag
    if(mol2 is None): mol2 = oep.mol_env
    if(mf_method is None): mf_method = oep.mf_method

    mf_frag = scf.EmbedSCF(mol1, umat+oep.vnuc_bound_frag, oep.smear_sigma)
    mf_frag.xc = mf_method
    mf_frag.scf()
    FRAG_1RDM = mf_frag.make_rdm1()

    mf_env = scf.EmbedSCF(mol2, umat+oep.vnuc_bound_env, oep.smear_sigma)
    mf_env.xc = mf_method
    mf_env.scf()
    ENV_1RDM = mf_env.make_rdm1()

    return (FRAG_1RDM, ENV_1RDM)


class OEPao:

    def __init__(self, dfet, params):

	self.params = params
	self.dim = dfet.dim
	self.umat = dfet.umat
	self.smear_sigma = dfet.smear_sigma

	self.mol_frag = dfet.mol_frag
	self.mol_env = dfet.mol_env

	self.vnuc_bound_frag = dfet.vnuc_bound_frag
	self.vnuc_bound_env  = dfet.vnuc_bound_env

	self.mf_method = dfet.mf_method
	self.P_ref = dfet.P_ref
	self.P_imp = dfet.P_imp
	self.P_bath = dfet.P_bath

	dim = self.dim
	if(self.umat is None): self.umat = np.zeros([dim,dim])


    def kernel(self):

	algorithm = self.params.algorithm
        if(algorithm == '2011'):
	    self.umat = self.oep_base(self.umat)
	elif(algorithm == 'split'):
	    self.umat = self.oep_loop(self.umat)

	self.P_imp, self.P_bath = self.verify_scf(self.umat)

	return self.umat


    init_density_partition = init_density_partition
    verify_scf = init_density_partition


    def oep_loop(self, _umat):

        umat = _umat.copy()
	self.P_imp, self.P_bath = self.init_density_partition(umat)

        threshold = self.params.diffP_tol
        maxit = self.params.outer_maxit
        it = 0
        while it < maxit:
            it += 1
            print " OEP iteration ", it

            P_imp_old = self.P_imp.copy()
            P_bath_old = self.P_bath.copy()

            umat = self.oep_base(umat, True)

	    #########
	    mf_frag = scf.EmbedSCF_nonscf(self.mol_frag, self.P_imp, umat+self.vnuc_bound_frag, self.smear_sigma)
            mf_frag.xc = self.mf_method
            mf_frag.scf()
            self.P_imp = mf_frag.make_rdm1()

            mf_env = scf.EmbedSCF_nonscf(self.mol_env, self.P_bath, umat+self.vnuc_bound_env, self.smear_sigma)
            mf_env.xc = self.mf_method
            mf_env.scf()
            self.P_bath = mf_env.make_rdm1()
	    #########

            diffP_imp = self.P_imp - P_imp_old
            diffP_bath = self.P_bath - P_bath_old
            gmax_imp = np.amax(np.absolute(diffP_imp))
            gmax_bath = np.amax(np.absolute(diffP_bath))
            print "diffP_max_imp, diffP_max_bath "
            print gmax_imp, gmax_bath

            sys.stdout.flush()
            if(gmax_imp < threshold and gmax_bath < threshold ):
                break

            P_imp_old = None
            P_bath_old = None


        return umat


    def oep_base(self, umat, nonscf = False):

        P_ref = self.P_ref
        dim = self.dim

        x = tools.mat2vec(umat, dim)

        _args = [P_ref, dim, nonscf]
        _args = tuple(_args)

        opt_method = self.params.opt_method
        result = None
        if( opt_method == 'BFGS' or opt_method == 'L-BFGS-B'):
            result = self.oep_bfgs(x, _args)

        x = result.x
        umat = tools.vec2mat(x, dim)

        return umat


    def oep_bfgs(self, x, _args):

        maxit = self.params.maxit
        gtol = self.params.gtol
        ftol = self.params.ftol
        algorithm = self.params.opt_method

        result = optimize.minimize(self.cost_wuyang,x,args=_args,method=algorithm, jac=True, \
                                   options={'disp': True, 'maxiter': maxit, 'gtol':gtol, 'ftol':ftol, 'maxcor':10} )

        return result


    def cost_wuyang(self, x, P_ref, dim, nonscf):


        umat = tools.vec2mat(x, dim)
        print "|umat| = ", np.linalg.norm(umat)

        if(nonscf == False):  #normal SCF

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
	    mf_frag = scf.EmbedSCF_nonscf(self.mol_frag, self.P_imp, umat+self.vnuc_bound_frag, self.smear_sigma)
	    mf_frag.xc = self.mf_method
	    mf_frag.scf()
            FRAG_energy = mf_frag.energy_elec()[0]
            FRAG_1RDM = mf_frag.make_rdm1()

	    mf_env = scf.EmbedSCF_nonscf(self.mol_env, self.P_bath, umat+self.vnuc_bound_env, self.smear_sigma)
            mf_env.xc = self.mf_method
            mf_env.scf()
            ENV_energy = mf_env.energy_elec()[0]
            ENV_1RDM = mf_env.make_rdm1()



        diffP = FRAG_1RDM + ENV_1RDM - P_ref
        energy = FRAG_energy + ENV_energy - np.trace(np.dot(P_ref,umat))

        grad = tools.mat2vec(diffP, dim)
        grad = -1.0 * grad

        print "2-norm (grad),       max(grad):"
        print np.linalg.norm(grad), ", ", np.amax(np.absolute(grad))

        f = -energy
        print 'W = ', f

        return (f, grad)
