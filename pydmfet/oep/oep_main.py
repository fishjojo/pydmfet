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
	self.mf_method = embedobj.mf_method
	self.P_imp = None
	self.P_bath = None
	self.umat = None

        self.params = params

    def kernel(self, umat = None, P_imp=None, P_bath=None):

	dim = self.dim

	self.umat = umat
	self.P_imp = P_imp
	self.P_bath = P_bath

        if(self.umat is None):
            self.umat = np.zeros([dim,dim],dtype=float)

	algorithm = self.params.algorithm
	if(algorithm == '2011'):
	    self.umat = self.oep_old(self.umat)
	elif(algorithm == 'split'):
            self.umat = self.oep_loop(self.umat)

	self.umat = self.umat - np.eye( self.umat.shape[ 0 ] ) * np.average( np.diag( self.umat ) )

	return self

    def oep_old(self, _umat):

	'''
	extended Wu-Yang 2011
	'''

	umat = _umat.copy()
	if(self.P_imp is None):
	    self.init_density_partition()

	umat = self.oep_base(umat, False)

	return umat


    def init_density_partition(self):

	#initial density partition
	dim = self.dim
        dim_imp = self.dim_imp

        subTEI = self.ops[-1]

        P_ref = self.P_ref
        Nelec = self.Ne_frag + self.Ne_env
        self.P_imp, self.P_bath = subspac.fullP_to_fragP(self, subTEI, Nelec, P_ref, dim, dim_imp, self.mf_method)

	#tools.MatPrint(self.P_imp, "P_imp")
        #tools.MatPrint(self.P_bath, "P_bath")


    def oep_loop(self, _umat):

	'''
	New OEP scheme
	Outer loop of OEP
	'''

	umat = _umat.copy()
	self.init_density_partition()

	threshold = self.params.diffP_tol
	maxit = self.params.outer_maxit
	it = 0
	t0 = (time.clock(),time.time())
	while it < maxit:
	    it += 1
	    print " OEP iteration ", it

	    P_imp_old = self.P_imp.copy()
            P_bath_old = self.P_bath.copy()
	    umat = self.oep_base(umat, True)

	    diffP_imp = self.P_imp - P_imp_old
	    diffP_bath = self.P_bath - P_bath_old
	    gmax_imp = np.amax(np.absolute(diffP_imp))
	    gmax_bath = np.amax(np.absolute(diffP_bath))
	    print "diffP_max_imp, diffP_max_bath "
	    print gmax_imp, gmax_bath
	    
	    if(gmax_imp < threshold and gmax_bath < threshold ):
		break

	    P_imp_old = None
	    P_bath_old = None

	umat = self.oep_old(umat)

	t1 = tools.timer("oep", t0)

	return umat


    def oep_base(self, umat, nonscf = True):

	P_ref = self.P_ref
	ops = self.ops
        dim = self.dim

        x = tools.mat2vec(umat, dim)

        Ne_frag = self.Ne_frag
        Ne_env = self.Ne_env

        ints = self.ints
	impJK_sub  = None
	bathJK_sub = None
	subTEI = ops[-1]
	if( nonscf == True):
	    impJK_sub = ints.impJK_sub( self.P_imp, subTEI)
	    bathJK_sub = ints.impJK_sub( self.P_bath, subTEI)

	_args = [P_ref, dim,Ne_frag,Ne_env]
	_args = _args + ops
	_args.append(impJK_sub)
	_args.append(bathJK_sub)
	_args = tuple(_args)

	opt_method = self.params.opt_method
	result = None
	if( opt_method == 'BFGS'):
	    result = self.oep_bfgs(x, _args)

        x = result.x
        umat = tools.vec2mat(x, dim)

        return umat


    def oep_bfgs(self, x, _args):

	maxit = self.params.maxit
        gtol = self.params.gtol
        ftol = self.params.ftol

        result = optimize.minimize(self.cost_wuyang,x,args=_args,method='BFGS', jac=True, options={'disp': True, 'maxiter': maxit, 'gtol':gtol, 'ftol':ftol} )

	return result

    def cost_wuyang(self, x, P_ref, dim, Ne_frag, Ne_env, \
		    subKin, subVnuc1, subVnuc2, subVnuc_bound, subCoreJK, subTEI, \
		    impJK_sub, bathJK_sub):

        umat = tools.vec2mat(x, dim)

	if(self.params.oep_print == 3):
	    print "sum(diag(umat)) = ", np.sum(np.diag(umat))
	    tools.MatPrint(umat, 'umat')

	FRAG_1RDM=None
	ENV_1RDM=None
	FRAG_energy=None
	ENV_energy=None

	if( impJK_sub is None):  #normal SCF

	    subOEI1 = subKin+subVnuc1+subVnuc_bound+subCoreJK+umat
            subOEI2 = subKin+subVnuc2-subVnuc_bound+subCoreJK+umat
            FRAG_energy, FRAG_1RDM, frag_mo = qcwrap.pyscf_rhf.scf( subOEI1, subTEI, dim, Ne_frag, self.P_imp, mf_method = self.mf_method )
            ENV_energy, ENV_1RDM, env_mo = qcwrap.pyscf_rhf.scf( subOEI2, subTEI, dim, Ne_env, self.P_bath, mf_method = self.mf_method )

	    print np.linalg.norm(FRAG_1RDM-self.P_imp)
	    print np.linalg.norm(ENV_1RDM-self.P_bath)


	else:  #non-self-consistent SCF
            subOEI1 = subKin + subVnuc1 + subVnuc_bound + subCoreJK + impJK_sub + umat
            subOEI2 = subKin + subVnuc2 - subVnuc_bound + subCoreJK + bathJK_sub + umat
	    FRAG_energy, FRAG_1RDM = qcwrap.pyscf_rhf.scf_oei( subOEI1, dim, Ne_frag)
	    ENV_energy, ENV_1RDM = qcwrap.pyscf_rhf.scf_oei( subOEI2, dim, Ne_env)


	self.P_imp = FRAG_1RDM
        self.P_bath = ENV_1RDM

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

