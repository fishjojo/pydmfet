import numpy as np
from scipy import optimize
from pydmfet import qcwrap,tools,subspac
import time

class OEP:

    def __init__(self, embedobj, params):

        self.P_ref = embedobj.P_ref_sub
        self.umat = embedobj.umat
        self.dim = embedobj.dim_sub
	self.dim_imp = embedobj.dim_imp
	self.dim_bath = embedobj.dim_bath
        self.Ne_frag = embedobj.Ne_frag
        self.Ne_env = embedobj.Ne_env
        self.loc2sub = embedobj.loc2sub
        self.impAtom = embedobj.impAtom
	self.boundary_atoms = embedobj.boundary_atoms
        self.core1PDM_loc = embedobj.core1PDM_loc
        self.ints = embedobj.ints
	self.ops = embedobj.ops

        self.params = params

    def kernel(self):

	ops = self.ops

        dim = self.dim
        if(self.umat is None):
            self.umat = np.zeros([dim,dim],dtype=float)

	algorithm = self.params.algorithm
	if(algorithm == '2011'):
	    self.umat = self.oep_old(ops)
	elif(algorithm == 'split'):
            self.umat = self.oep_loop(ops)

	self.umat = self.umat - np.eye( self.umat.shape[ 0 ] ) * np.average( np.diag( self.umat ) )

    def oep_old(self,_ops):

	'''
	extended Wu-Yang 2011
	'''

	umat = self.umat.copy()
	P_ref = self.P_ref
	umat = self.oep_base(umat,P_ref,_ops)

	return umat


    def oep_loop(self,ops):

	'''
	New OEP scheme
	Outer loop of OEP
	'''

	umat = self.umat.copy()
        dim = self.dim
	dim_imp = self.dim_imp

        #calculate operators
	subTEI = ops[-1]

	P_ref = self.P_ref
	Nelec = self.Ne_frag + self.Ne_env
	P_imp, P_bath = subspac.fullP_to_fragP(self, subTEI, Nelec, P_ref, dim, dim_imp)

	#tools.MatPrint(P_imp, "P_imp")
	#tools.MatPrint(P_bath, "P_bath")

	threshold = self.params.diffP_tol
	maxit = self.params.outer_maxit
	it = 0
	t0 = (time.clock(),time.time())
	while it < maxit:
	    it += 1
	    print " OEP iteration ", it

	    P_imp_old = P_imp.copy()
            P_bath_old = P_bath.copy()
	    umat, P_imp, P_bath = self.oep_base(umat,P_ref, ops,P_imp_old,P_bath_old)

	    diffP_imp = P_imp - P_imp_old
	    diffP_bath = P_bath - P_bath_old
	    gmax_imp = np.amax(np.absolute(diffP_imp))
	    gmax_bath = np.amax(np.absolute(diffP_bath))
	    print "diffP_max_imp, diffP_max_bath "
	    print gmax_imp, gmax_bath
	    
	    if(gmax_imp < threshold and gmax_bath < threshold ):
		break

	    P_imp_old = None
	    P_bath_old = None
	t1 = tools.timer("oep", t0)

	return umat


    def oep_base(self,umat,P_ref,ops,P_imp = None, P_bath = None):

        dim = self.dim
        x = tools.mat2vec(umat, dim)

        Ne_frag = self.Ne_frag
        Ne_env = self.Ne_env

        ints = self.ints
	impJK_sub  = None
	bathJK_sub = None
	subTEI = ops[-1]
	if( P_imp is not None):
	    impJK_sub = ints.impJK_sub( P_imp, subTEI)
	    if(P_bath is None):
		raise Exception("P_imp is not None, but P_bath is None!")
	if( P_bath is not None):
	    bathJK_sub = ints.impJK_sub( P_bath, subTEI)

	_args = [P_ref,dim,Ne_frag,Ne_env]
	_args = _args + ops
	_args.append(impJK_sub)
	_args.append(bathJK_sub)
	_args = tuple(_args)
        #_args = (P_ref,dim,Ne_frag,Ne_env,subKin,subVnuc1,subVnuc2,subVnuc_bound,subCoreJK,subTEI,impJK_sub,bathJK_sub)

	opt_method = self.params.opt_method
	result = None
	if( opt_method == 'BFGS'):
	    result = self.oep_bfgs(x, _args)

        x = result.x
        umat = tools.vec2mat(x, dim)

	if (P_imp is None and P_bath is None):
            return umat
	else:
	    subKin = ops[0]
	    subVnuc1 = ops[1]
	    subVnuc2 = ops[2]
	    subVnuc_bound = ops[3]
	    subCoreJK = ops[4]
	    subOEI1 = subKin+subVnuc1+subVnuc_bound+subCoreJK+impJK_sub+umat
            subOEI2 = subKin+subVnuc2-subVnuc_bound+subCoreJK+bathJK_sub+umat
            E_imp, P_imp = qcwrap.pyscf_rhf.scf_oei( subOEI1, dim, Ne_frag)
            E_bath, P_bath = qcwrap.pyscf_rhf.scf_oei( subOEI2, dim, Ne_env)
	    return (umat, P_imp, P_bath)


    def oep_bfgs(self, x, _args):

	maxit = self.params.maxit
        gtol = self.params.gtol
        ftol = self.params.ftol

        result = optimize.minimize(self.cost_wuyang,x,args=_args,method='L-BFGS-B', jac=True, options={'disp': True, 'maxiter': maxit, 'gtol':gtol, 'ftol':ftol} )

	return result

    def cost_wuyang(self, x, P_ref,dim, Ne_frag, Ne_env, subKin, subVnuc1, subVnuc2, subVnuc_bound, subCoreJK, subTEI, impJK_sub,bathJK_sub):

        umat = tools.vec2mat(x, dim)

	if(self.params.oep_print == 3):
	    tools.MatPrint(umat, 'umat')

	FRAG_1RDM=None
	ENV_1RDM=None
	FRAG_energy=None
	ENV_energy=None

	if( impJK_sub is None):  #normal SCF

	    #guess density 
	    subOEI1 = subKin+subVnuc1+subVnuc_bound+subCoreJK+umat
            subOEI2 = subKin+subVnuc2-subVnuc_bound+subCoreJK+umat
            FRAG_energy, FRAG_1RDM, frag_mo = qcwrap.pyscf_rhf.scf( subOEI1, subTEI, dim, Ne_frag)
            ENV_energy, ENV_1RDM, env_mo = qcwrap.pyscf_rhf.scf( subOEI2, subTEI, dim, Ne_env)
	    #real calculation
            #FRAG_energy, FRAG_1RDM, frag_mo = qcwrap.pyscf_rhf.scf( subOEI1+subVnuc2, subTEI, dim, Ne_frag, OneDM0=FRAG_1RDM)
            #ENV_energy, ENV_1RDM, env_mo   = qcwrap.pyscf_rhf.scf( subOEI2+subVnuc1, subTEI, dim, Ne_env,  OneDM0=ENV_1RDM)

	else:  #non-self-consistent SCF
            subOEI1 = subKin + subVnuc1 + subVnuc_bound + subCoreJK + impJK_sub + umat
            subOEI2 = subKin + subVnuc2 - subVnuc_bound + subCoreJK + bathJK_sub + umat
	    FRAG_energy, FRAG_1RDM = qcwrap.pyscf_rhf.scf_oei( subOEI1, dim, Ne_frag)
	    ENV_energy, ENV_1RDM = qcwrap.pyscf_rhf.scf_oei( subOEI2, dim, Ne_env)


	if(self.params.oep_print == 3):
            tools.MatPrint(FRAG_1RDM, 'fragment density')
            tools.MatPrint(ENV_1RDM, 'environment density')


        diffP = FRAG_1RDM + ENV_1RDM - P_ref
        energy = FRAG_energy + ENV_energy - np.trace(np.dot(P_ref,umat))
        #print energy
        grad = tools.mat2vec(diffP, dim)
        grad = -1.0 * grad

	gtol = self.params.gtol
	size = dim*(dim+1)/2
	for i in range(size):
	    if(abs(grad[i]) < gtol): #numerical error may cumulate
		grad[i] = 0.0

        print "2-norm (grad),       max(grad):"
        print np.linalg.norm(grad), ", ", np.amax(np.absolute(grad))
        #print diffP
        #print umat
        return (-energy, grad)

