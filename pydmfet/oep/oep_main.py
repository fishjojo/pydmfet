import numpy as np
from scipy import optimize
from pydmfet import qcwrap,tools,subspac
import time,copy

class OEP:

    def __init__(self, embedobj, params):

        self.P_ref = embedobj.P_ref_sub
        self.umat = embedobj.umat
        self.dim = embedobj.dim_sub
	self.dim_imp = embedobj.dim_imp
	#self.dim_imp_virt = embedobj.dim_imp_virt
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
	self.P_imp = embedobj.P_imp
	self.P_bath = embedobj.P_bath

	self.frag_mo = None
	self.env_mo = None

	self.P_frag_loc = embedobj.P_frag_loc
	self.P_env_loc = embedobj.P_env_loc

        self.params = params

    def kernel(self):

	dim = self.dim


	#debug
#	Ne_frag = copy.copy(self.Ne_frag)
#	P_ref = copy.copy(self.P_ref)
#        self.Ne_frag = 0
#        self.P_ref = self.P_bath
        if(self.umat is None):
            self.umat = np.zeros([dim,dim],dtype=float)

	algorithm = self.params.algorithm
	if(algorithm == '2011'):
	    self.umat = self.oep_old(self.umat)
	elif(algorithm == 'split'):
            self.umat = self.oep_loop(self.umat)
	elif(algorithm == 'leastsq'):
	    self.umat = self.oep_leastsq(self.umat)

	self.umat = self.umat - np.eye( self.umat.shape[ 0 ] ) * np.average( np.diag( self.umat ) )

#	self.Ne_frag = Ne_frag
#	self.P_ref = P_ref
#	self.umat = self.oep_old(self.umat)

	self.P_imp, self.P_bath = self.verify_scf(self.umat)

	#tools.MatPrint(self.P_imp,"P_imp")
        #tools.MatPrint(self.P_bath,"P_bath")
	#tools.MatPrint(self.umat,"umat")
	return self


    def oep_old(self, _umat):

	'''
	extended Wu-Yang 2011
	'''

	umat = _umat.copy()
#	if(self.P_imp is None):
#	    self.init_density_partition()

	umat = self.oep_base(umat, False)

	return umat


    def init_density_partition(self, method = 1):

	if(method == 1):
	    self.P_imp, self.P_bath = self.verify_scf(umat = None)
	elif(method == 2):
	    self.P_imp = np.dot(np.dot(self.loc2sub[:,:self.dim].T,self.P_frag_loc),self.loc2sub[:,:self.dim])
	    self.P_bath = np.dot(np.dot(self.loc2sub[:,:self.dim].T,self.P_env_loc),self.loc2sub[:,:self.dim])
	else:
	    dim = self.dim
            dim_imp = self.dim_imp
            subTEI = self.ops["subTEI"]
            P_ref = self.P_ref
            Nelec = self.Ne_frag + self.Ne_env
            self.P_imp, self.P_bath = subspac.fullP_to_fragP(self, subTEI, Nelec, P_ref, dim, dim_imp, self.mf_method)

	#tools.MatPrint(self.P_imp, "P_imp")
        #tools.MatPrint(self.P_bath, "P_bath")

	print "initialize density partition"
	print "Ne_imp = ", np.sum(np.diag(self.P_imp))
	print "Ne_bath = ", np.sum(np.diag(self.P_bath))
	print "|P_imp + P_bath - P_ref| = ", np.linalg.norm(self.P_imp + self.P_bath - self.P_ref)

	dim = self.dim
	frag_occ = np.zeros([dim],dtype = float)
        for i in range(dim):
            frag_occ[i] = self.P_imp[i,i]
        self.ints.sub_molden( self.loc2sub[:,:dim], 'frag_dens_guess.molden', frag_occ )

        env_occ = np.zeros([dim],dtype = float)
        for i in range(dim):
            env_occ[i] = self.P_bath[i,i]
        self.ints.sub_molden( self.loc2sub[:,:dim], 'env_dens_guess.molden', env_occ )



    def oep_loop(self, _umat):

	'''
	New OEP scheme
	Outer loop of OEP
	'''

	umat = _umat.copy()
	if(self.P_imp is None):
	    self.init_density_partition()

        tools.MatPrint(self.P_imp, "P_imp_guess")
        tools.MatPrint(self.P_bath, "P_bath_guess")


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

        #self.verify_scf(umat)
	#umat = self.oep_old(umat)
	#umat = self.oep_leastsq(umat)

	#tools.MatPrint(self.P_imp,"P_imp")
        #tools.MatPrint(self.P_bath,"P_bath")
	#tools.MatPrint(umat,"umat")

	#self.P_imp,self.P_bath = self.verify_scf(umat)

	t1 = tools.timer("oep", t0)

	return umat

    def verify_scf(self, umat):
	

	ops = self.ops
	Ne_frag = self.Ne_frag
	Ne_env = self.Ne_env
	dim = self.dim

	subTEI = ops["subTEI"]
	subOEI1 = ops["subKin"]+ops["subVnuc1"]+ops["subVnuc_bound1"]
        subOEI2 = ops["subKin"]+ops["subVnuc2"]+ops["subVnuc_bound2"]+ops["subCoreJK"]
	if(umat is not None):
	    subOEI1 = subOEI1 + umat
	    subOEI2 = subOEI2 + umat

	'''
	frag_coredm_guess = self.P_imp
	env_coredm_guess = self.P_bath
	FRAG_energy, FRAG_1RDM, frag_mo = qcwrap.pyscf_rhf.scf( subOEI1, subTEI, dim, Ne_frag, frag_coredm_guess, mf_method = self.mf_method )
        ENV_energy, ENV_1RDM, env_mo = qcwrap.pyscf_rhf.scf( subOEI2, subTEI, dim, Ne_env, env_coredm_guess, mf_method = self.mf_method )

	print "nonscf energies:"
	print FRAG_energy, ENV_energy

	frag_occ = np.zeros([dim],dtype = float)
        for i in range(Ne_frag/2):
            frag_occ[i] = 2.0
	self.ints.submo_molden( frag_mo[:,:dim], frag_occ, self.loc2sub, 'frag_dens_nonscf.molden' )

        env_occ = np.zeros([dim],dtype = float)
        for i in range(Ne_env/2):
            env_occ[i] = 2.0
        self.ints.submo_molden( env_mo[:,:dim], env_occ, self.loc2sub, 'env_dens_nonscf.molden' )
	'''

        frag_coredm_guess,mo_coeff = tools.fock2onedm(subOEI1, Ne_frag/2)
        FRAG_energy, FRAG_1RDM, frag_mo = qcwrap.pyscf_rhf.scf( subOEI1, subTEI, dim, Ne_frag, frag_coredm_guess, mf_method = self.mf_method )
        env_coredm_guess,mo_coeff = tools.fock2onedm(subOEI2, Ne_env/2)
        ENV_energy, ENV_1RDM, env_mo = qcwrap.pyscf_rhf.scf( subOEI2, subTEI, dim, Ne_env, env_coredm_guess, mf_method = self.mf_method )

	#print "scf energies:"
        #print FRAG_energy, ENV_energy
	#self.ints.submo_molden( frag_mo[:,:dim], frag_occ, self.loc2sub, 'frag_dens_scf.molden' )
        #self.ints.submo_molden( env_mo[:,:dim], env_occ, self.loc2sub, 'env_dens_scf.molden' )


	if(self.P_imp is not None):
	    print np.linalg.norm(FRAG_1RDM - self.P_imp)
	    print np.linalg.norm(ENV_1RDM - self.P_bath)
	diffP = FRAG_1RDM + ENV_1RDM - self.P_ref
	diffP_norm = np.linalg.norm(diffP)
	diffP_max = np.amax(np.absolute(diffP) )
	print "|P_frag + P_env - P_ref| = ", diffP_norm
	print "max element of (P_frag + P_env - P_ref) = ", diffP_max

	self.frag_mo = frag_mo
	self.env_mo = env_mo

	return (FRAG_1RDM, ENV_1RDM)

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
	subTEI = ops["subTEI"]
	if( nonscf == True):
	    impJK_sub = ints.impJK_sub( self.P_imp, subTEI)
	    bathJK_sub = ints.impJK_sub( self.P_bath, subTEI)

	_args = [P_ref, dim, Ne_frag, Ne_env]
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

        result = optimize.minimize(self.cost_wuyang,x,args=_args,method='L-BFGS-B', jac=True, options={'disp': True, 'maxiter': maxit, 'gtol':gtol, 'ftol':ftol} )

	return result

    def cost_wuyang(self, x, P_ref, dim, Ne_frag, Ne_env, impJK_sub, bathJK_sub):

	t0 = (time.clock(),time.time())

        umat = tools.vec2mat(x, dim)

	print "|umat| = ", np.linalg.norm(umat)
	if(self.params.oep_print >= 3):
	    print "sum(diag(umat)) = ", np.sum(np.diag(umat))
	    tools.MatPrint(umat, 'umat')

	FRAG_1RDM = np.zeros([dim,dim], dtype = float)
	ENV_1RDM = np.zeros([dim,dim], dtype = float)
	FRAG_energy = 0.0
	ENV_energy = 0.0

	ops = self.ops
	subKin = ops["subKin"]
	subVnuc1 = ops["subVnuc1"]
	subVnuc2 = ops["subVnuc2"]
	subVnuc_bound1 = ops["subVnuc_bound1"]
	subVnuc_bound2 = ops["subVnuc_bound2"]
	subCoreJK = ops["subCoreJK"]
	subTEI = ops["subTEI"]

	if( impJK_sub is None):  #normal SCF
	    subOEI1 = subKin+subVnuc1+subVnuc_bound1+umat
            subOEI2 = subKin+subVnuc2+subVnuc_bound2+subCoreJK+umat

	    frag_coredm_guess,mo_coeff = tools.fock2onedm(subOEI1, Ne_frag/2)
            FRAG_energy, FRAG_1RDM, frag_mo = qcwrap.pyscf_rhf.scf( subOEI1, subTEI, dim, Ne_frag, frag_coredm_guess, mf_method = self.mf_method )

	    if(Ne_env > 0):
	        env_coredm_guess,mo_coeff = tools.fock2onedm(subOEI2, Ne_env/2)
                ENV_energy, ENV_1RDM, env_mo = qcwrap.pyscf_rhf.scf( subOEI2, subTEI, dim, Ne_env, env_coredm_guess, mf_method = self.mf_method )

	else:  #non-self-consistent SCF
            subOEI1 = subKin + subVnuc1 + subVnuc_bound1 + impJK_sub + umat
            subOEI2 = subKin + subVnuc2 + subVnuc_bound2 + subCoreJK + bathJK_sub + umat
	    FRAG_energy, FRAG_1RDM = qcwrap.pyscf_rhf.scf_oei( subOEI1, dim, Ne_frag)
	    if(Ne_env > 0):
	        ENV_energy, ENV_1RDM = qcwrap.pyscf_rhf.scf_oei( subOEI2, dim, Ne_env)


	#tools.MatPrint(frag_mo,"frag_mo")
	#tools.MatPrint(env_mo,"env_mo")

	self.P_imp = FRAG_1RDM
        self.P_bath = ENV_1RDM

	if(self.params.oep_print >= 3):
	    print "number of electrons in fragment", np.sum(np.diag(FRAG_1RDM))
            tools.MatPrint(FRAG_1RDM, 'fragment density')
	    print "number of electrons in environment", np.sum(np.diag(ENV_1RDM))
            tools.MatPrint(ENV_1RDM, 'environment density')


        diffP = FRAG_1RDM + ENV_1RDM - P_ref
        energy = FRAG_energy + ENV_energy - np.trace(np.dot(P_ref,umat))

	#energy = np.trace(np.dot(subKin+subVnuc1+subVnuc_bound1+impJK_sub,FRAG_1RDM)) + np.trace(np.dot(umat,diffP))

        #print energy

	#istart = self.dim_imp - self.dim_imp_virt
	#iend = self.dim_imp
	#for i in range(istart,iend):
	#    for j in range(istart,iend):
	#	diffP[i][j] = 0.0

        grad = tools.mat2vec(diffP, dim)
        grad = -1.0 * grad

	gtol = self.params.gtol
	size = dim*(dim+1)/2
	#for i in range(size):
	#    if(abs(grad[i]) < gtol): #numerical error may accumulate
#		grad[i] = 0.0

        print "2-norm (grad),       max(grad):"
        print np.linalg.norm(grad), ", ", np.amax(np.absolute(grad))
        #print diffP
        #print umat


	l2_f, l2_g = self.l2_reg(x)

	f = -energy + l2_f
	grad = grad + l2_g

	t1 = tools.timer("wu-yang cost function", t0)


        return (f, grad)



    def l2_reg(self, x):

	x_norm = np.linalg.norm(x)
	f = self.params.l2_lambda * x_norm**2
	g = x*(2.0*self.params.l2_lambda)

	return (f,g)



    def oep_leastsq(self, _umat):

        umat = _umat.copy()

	umat = self.oep_leastsq_base(umat)

	return umat


    def oep_leastsq_base(self, umat):

	dim = self.dim
	ops = self.ops
	P_ref = self.P_ref
	Ne_frag = self.Ne_frag
	Ne_env = self.Ne_env

	x0 = tools.mat2vec(umat, dim)
	_args = (dim, Ne_frag, Ne_env, P_ref, ops)

	maxit = self.params.maxit

	result = optimize.least_squares(self.oep_calc_diffP, x0, jac=self.oep_calc_diffP_derivative, bounds=(-2.0, 2.0), method='trf', ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='soft_l1', f_scale=1.0, diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None, max_nfev=maxit, verbose=2, args=_args)

	x = result.x
        umat = tools.vec2mat(x, dim)
	return umat


    def oep_calc_diffP(self, x, dim, Ne_frag, Ne_env, P_ref, ops):

        umat = tools.vec2mat(x, dim)
        print "|umat| = ", np.linalg.norm(umat)
        if(self.params.oep_print >= 3):
            print "sum(diag(umat)) = ", np.sum(np.diag(umat))
            tools.MatPrint(umat, 'umat')

        FRAG_1RDM = np.zeros([dim,dim], dtype = float)
        ENV_1RDM = np.zeros([dim,dim], dtype = float)
        FRAG_energy = 0.0
        ENV_energy = 0.0
	frag_mo = np.zeros([dim,dim], dtype = float)
	env_mo = np.zeros([dim,dim], dtype = float)

        subKin = ops["subKin"]
        subVnuc1 = ops["subVnuc1"]
        subVnuc2 = ops["subVnuc2"]
        subVnuc_bound1 = ops["subVnuc_bound1"]
        subVnuc_bound2 = ops["subVnuc_bound2"]
        subCoreJK = ops["subCoreJK"]
        subTEI = ops["subTEI"]

        subOEI1 = subKin+subVnuc1+subVnuc_bound1+umat
        subOEI2 = subKin+subVnuc2+subVnuc_bound2+subCoreJK+umat

	if(Ne_frag > 0):
            frag_coredm_guess,mo_coeff = tools.fock2onedm(subOEI1, Ne_frag/2)
            FRAG_energy, FRAG_1RDM, frag_mo = qcwrap.pyscf_rhf.scf( subOEI1, subTEI, dim, Ne_frag, frag_coredm_guess, mf_method = self.mf_method )

        if(Ne_env > 0):
            env_coredm_guess,mo_coeff = tools.fock2onedm(subOEI2, Ne_env/2)
            ENV_energy, ENV_1RDM, env_mo = qcwrap.pyscf_rhf.scf( subOEI2, subTEI, dim, Ne_env, env_coredm_guess, mf_method = self.mf_method )

	self.frag_mo = frag_mo
	self.env_mo = env_mo
	self.P_imp = FRAG_1RDM
	self.P_bath = ENV_1RDM

	diffP = FRAG_1RDM + ENV_1RDM - P_ref
	grad = tools.mat2vec(diffP, dim)

	print "2-norm (grad),       max(grad):"
        print np.linalg.norm(grad), ", ", np.amax(np.absolute(grad))


	return grad


    def oep_calc_diffP_derivative(self, x, dim, Ne_frag, Ne_env, P_ref, ops):

	t0 = (time.clock(),time.time())
	size = dim*(dim+1)/2

	hess_frag = oep_calc_dPdV(self.frag_mo[:,:-1],self.frag_mo[:,-1],size,self.Ne_frag/2,dim)
	hess_env = oep_calc_dPdV(self.env_mo[:,:-1],self.env_mo[:,-1],size,self.Ne_env/2,dim)

	t1 = tools.timer("hessian", t0)
	return hess_frag + hess_env


def oep_calc_dPdV(jCa,orb_Ea,size,NOcc,NOrb):


	imax = NOcc
	amax = NOrb
	NOa = imax
	NVa = amax-imax
	NOVa = NOa*NVa
	N2 = NOrb*NOrb

	jt = np.zeros([N2*NOVa,1],dtype=float)
	for i in range(imax):
            index_munu = i*N2*NVa
            for a in range(imax,amax):
        	for mu in range(NOrb):
            	    Cmui=jCa[mu,i]
		    jt[index_munu:index_munu+NOrb,0] = Cmui * jCa[:,a] + jt[index_munu:index_munu+NOrb,0]
            	    index_munu = index_munu + NOrb

	jt_dia = np.zeros([N2*NOVa,1],dtype=float)
        for i in range(imax):
	    index_ia = i*NVa
            for a in range(imax,amax):
		eps_ia = orb_Ea[i] - orb_Ea[a]
                dia = 1.0/eps_ia
                ioff = N2*index_ia
		jt_dia[ioff:ioff+N2,0] = jt_dia[ioff:ioff+N2,0] + dia * jt[ioff:ioff+N2,0]
                index_ia = index_ia + 1

	jt = np.reshape(jt,(N2,NOVa),'F')
	jt_dia = np.reshape(jt_dia,(N2,NOVa),'F')
	jHfull = np.dot(jt,jt_dia.T)

	jt = None
	jt_dia = None


	hess = np.zeros([size*size,1],dtype=float)
	for mu in range(NOrb):
	    index = (2*NOrb-mu+1)*mu/2*size
	    for nu in range(mu,NOrb):
		jTemp = np.copy(jHfull[:,mu+nu*NOrb])
		jTemp = np.reshape(jTemp,(NOrb,NOrb))
		jTemp = jTemp + jTemp.T

		jTemp1 = np.copy(jHfull[:,nu+mu*NOrb])
		jTemp1 = np.reshape(jTemp1,(NOrb,NOrb))
		jTemp1 = jTemp1 + jTemp1.T

		jTemp = jTemp + jTemp1

		for i in range(NOrb):
		    jTemp[i,i] = jTemp[i,i]*0.5

		for lam in range(NOrb):
		    hess[index:index+NOrb-lam,0] = np.copy(jTemp[lam:NOrb, lam])
                    index = index + (NOrb-lam)


	hess = np.reshape(hess,(size,size),'F')
	hess = hess.T

	return 2.0*hess
