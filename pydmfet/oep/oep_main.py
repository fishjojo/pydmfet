import numpy as np
from scipy import optimize
from pydmfet import qcwrap,tools,subspac

class OEP:

    def __init__(self, embedobj, params):

        self.P_ref = embedobj.P_ref_sub
        self.umat = embedobj.umat
        self.dim = embedobj.dim_sub
	self.dim_imp = embedobj.dim_imp
	self.dim_bath = embedobj.dim_bath
        self.Ne_frag = embedobj.Ne_frag
        self.Ne_env = embedobj.Ne_env
	self.charge = embedobj.charge
	self.spin = embedobj.spin 
        self.loc2sub = embedobj.loc2sub
        self.impAtom = embedobj.impAtom
        self.core1PDM_loc = embedobj.core1PDM_loc
        self.ints = embedobj.ints

        self.params = params

    def kernel(self):

        dim = self.dim
        if(self.umat is None):
            self.umat = np.zeros([dim,dim],dtype=float)

        self.umat = self.oep_loop()

    def oep_old(self):

	'''
	extended Wu-Yang 2011
	'''

	umat = self.umat.copy()
	P_ref = self.P_ref
	umat = self.oep_base(umat,P_ref)

	return umat



    def oep_loop(self):

	'''
	New OEP scheme
	Outer loop of OEP
	'''

	umat = self.umat.copy()
        dim = self.dim
	dim_imp = self.dim_imp

	P_ref = self.P_ref
	P_imp, P_bath = subspac.fullP_to_fragP(P_ref,dim_imp, dim)

	maxit = 50
	it = 0
	while it < maxit:
	    it += 1

	    P_imp_old = P_imp.copy()
            P_bath_old = P_bath.copy()
	    P_imp, P_bath, umat = self.oep_base(umat,P_ref, P_imp_old,P_bath_old)

	    diffP_imp = P_imp - P_imp_old
	    diffP_bath = P_bath - P_bath_old
	    gmax_imp = np.amax(np.absolute(diffP_imp)
	    gmax_bath = np.amax(np.absolute(diffP_bath)
	    if(gmax_imp < threshold and gmax_bath < threshold ):
		break

	    P_imp_old = None
	    P_bath_old = None

	return umat


    def oep_base(self,umat,P_ref,P_imp = None, P_bath = None):

        dim = self.dim
        x = tools.mat2vec(umat, dim)

        Ne_frag = self.Ne_frag
        Ne_env = self.Ne_env

        loc2sub = self.loc2sub
        impAtom = self.impAtom
        core1PDM_loc = self.core1PDM_loc

        ints = self.ints

	charge = self.charge
	spin = self.spin 

        subKin = ints.frag_kin_sub( impAtom, loc2sub, dim, charge[0],spin[0] )
        subVnuc1 = ints.frag_vnuc_sub( impAtom, loc2sub, dim, charge[0],spin[0] )
        subVnuc2 = ints.frag_vnuc_sub( 1-impAtom, loc2sub, dim, charge[1],spin[1] )
        subCoreJK = ints.coreJK_sub( loc2sub, dim, core1PDM_loc )
        subTEI = ints.dmet_tei( loc2sub, dim )

	impJK_sub  = None
	bathJK_sub = None
	if( P_imp is not None):
	    impJK_sub = ints.impJK_sub( P_imp, subTEI)
	    if(P_bath is None):
		print "P_bath is None!"
		assert(0==1)
	if( P_bath is not None):
	    bathJK_sub = ints.impJK_sub( P_bath, subTEI)


        _args = (P_ref,dim,Ne_frag,Ne_env,subKin,subVnuc1,subVnuc2,subCoreJK,impJK_sub,bathJK_sub,subTEI)

	opt_method = self.params.opt_method
	result = None
	if( opt_method = 'BFGS'):
	    result = oep_bfgs(x, _args)

        x = result.x
        umat = tools.vec2mat(x, dim)

        return umat

    def oep_bfgs(self, x, _args):

	maxit = self.params.maxit
        gtol = self.params.gtol
        ftol = self.params.ftol

        result = optimize.minimize(self.cost_wuyang,x,args=_args,method='L-BFGS-B', jac=True, options={'disp': True, 'maxiter': maxit, 'gtol':gtol, 'ftol':ftol} )

	return result

    def cost_wuyang(self, x, P_ref,dim, Ne_frag, Ne_env, subKin, subVnuc1, subVnuc2, subCoreJK, impJK_sub,bathJK_sub,subTEI):

        umat = tools.vec2mat(x, dim)
        print "umat"
        print umat

	FRAG_1RDM=None
	ENV_1RDM=None
	FRAG_energy=None
	ENV_energy=None

	if( impJK_sub is None):  #normal SCF

	    #guess density 
	    subOEI1 = subKin+subVnuc1+subCoreJK+umat
            subOEI2 = subKin+subVnuc2+subCoreJK+umat
            FRAG_energy, FRAG_1RDM = qcwrap.pyscf_rhf.scf( subOEI1, subTEI, dim, Ne_frag)
            ENV_energy, ENV_1RDM = qcwrap.pyscf_rhf.scf( subOEI2, subTEI, dim, Ne_env)
	    #real calculation
            #FRAG_energy, FRAG_1RDM = qcwrap.pyscf_rhf.scf( subOEI1+subVnuc2, subTEI, dim, Ne_frag, OneDM0=FRAG_1RDM)
            #ENV_energy, ENV_1RDM   = qcwrap.pyscf_rhf.scf( subOEI2+subVnuc1, subTEI, dim, Ne_env,  OneDM0=ENV_1RDM)

	else:  #non-self-consistent SCF
            subOEI1 = subKin+subVnuc1+subCoreJK+impJK_sub+umat
            subOEI2 = subKin+subVnuc2+subCoreJK+bathJK_sub+umat
	    FRAG_energy, FRAG_1RDM = qcwrap.pyscf_rhf.scf_oei( subOEI1, dim, Ne_frag)
	    ENV_energy, ENV_1RDM = qcwrap.pyscf_rhf.scf_oei( subOEI2, dim, Ne_env)


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

