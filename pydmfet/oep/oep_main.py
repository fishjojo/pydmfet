import os,sys
import ctypes
import numpy as np
from scipy import optimize
from pydmfet import qcwrap,tools,subspac, libgen
import time,copy
from pyscf import lib, scf, mp
from pyscf.tools import cubegen

libhess = np.ctypeslib.load_library('libhess', os.path.dirname(__file__))
libsvd = np.ctypeslib.load_library('libsvd', os.path.dirname(__file__))

def init_umat(oep):

    dim = oep.dim
    Ne = oep.Ne_frag + oep.Ne_env
    ops = oep.ops

    subOEI = ops["subKin"]+ops["subVnuc1"]+ops["subVnuc2"]+ops["subCoreJK"]
    subTEI = ops["subTEI"]

    coredm = oep.core1PDM_ao
    ao2sub = oep.ao2sub[:,:dim]
    mf = qcwrap.qc_scf(Ne,dim,oep.mf_method,mol=oep.mol,oei=subOEI,tei=subTEI,dm0=np.eye(dim),coredm=coredm,ao2sub=ao2sub)
    mf.runscf()

    vks = mf.get_veff()

    #P_ref = oep.P_imp + oep.P_bath
    P_ref = oep.P_ref

    _args=[P_ref,dim,Ne, subOEI, subTEI,coredm,ao2sub,oep.mf_method,oep.mol]
    _args=tuple(_args)

    umat = np.zeros((dim,dim)) 
    x = tools.mat2vec(vks, dim)
    result = wy_oep(x,_args)
    voep = tools.vec2mat(result.x, dim)

    return vks-voep


def wy_oep(x,_args):


    maxit = 200
    gtol = 1e-6
    ftol = 1e-13
    algorithm = 'L-BFGS-B'

    result = optimize.minimize(wy_oep_cost,x,args=_args,method=algorithm, jac=True, \
                               options={'disp': True, 'maxiter': maxit, 'gtol':gtol, 'ftol':ftol, 'maxcor':10} )


    return result


def wy_oep_cost(x, P_ref, dim, Ne, subOEI, subTEI,coredm,ao2sub,mf_method,mol):


    umat = tools.vec2mat(x, dim)
    print "|umat| = ", np.linalg.norm(umat)

    oei = subOEI + umat
    tei = subTEI

    FRAG_energy, FRAG_1RDM = qcwrap.pyscf_rhf.scf_oei( oei, dim, Ne)

    energy = FRAG_energy - np.trace(np.dot(P_ref,umat))
    diffP = FRAG_1RDM - P_ref

    grad = tools.mat2vec(diffP, dim)
    grad = -1.0 * grad

    return (-energy,grad)

class OEP:

    def __init__(self, embedobj, params):

	self.ks = embedobj.mf_full
	self.ks_frag = None
	self.ks_env = None
	self.mol = embedobj.mol
	self.mol_frag = embedobj.mol_frag
	self.mol_env = embedobj.mol_env
        self.P_ref = embedobj.P_ref_sub
        self.umat = embedobj.umat
        self.dim = embedobj.dim_sub
	self.dim_imp = embedobj.dim_imp
	#self.dim_imp_virt = embedobj.dim_imp_virt
	self.dim_bath = embedobj.dim_bath
        self.Ne_frag = embedobj.Ne_frag
        self.Ne_env = embedobj.Ne_env
        self.loc2sub = embedobj.loc2sub
	self.ao2sub = embedobj.ao2sub
        self.impAtom = embedobj.impAtom
	self.boundary_atoms = embedobj.boundary_atoms
        self.core1PDM_loc = embedobj.core1PDM_loc
	self.core1PDM_ao = embedobj.core1PDM_ao
        self.ints = embedobj.ints
	self.ops = embedobj.ops
	self.mf_method = embedobj.mf_method
	self.P_imp = embedobj.P_imp
	self.P_bath = embedobj.P_bath
	self.impJK_sub = None
	self.bathJK_sub = None

	self.Kcoeff = embedobj.Kcoeff
	self.smear_sigma = embedobj.smear_sigma

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
	    self.init_density_partition()
	    self.P_imp0 = self.P_imp
	    self.P_bath0 = self.P_bath
	    #self.umat = init_umat(self)
	    self.umat = self.oep_old(self.umat)
	elif(algorithm == 'split'):
	    '''
	    maxit = 10
	    it = 0
	    umat = self.umat
	    while it<maxit:
		it += 1

                umat = self.oep_loop(umat)
		if(it > 1):
		    du = abs(np.linalg.norm(umat) - np.linalg.norm(umat_old))
                    print '|umat-umat_old| = ', du
                    if(du < 1e-2): 
			break
		    if(it==maxit):
			print 'umat not converge'
			break

		umat_old = umat.copy()
		umat = self.opt_umat_2(umat, tol=1e-6)
	    '''
	    umat = self.oep_loop(self.umat)

	    #tools.MatPrint(umat,"umat")
	    #umat = self.opt_umat_2(umat, tol=1e-9)
	    #tools.MatPrint(umat,"umat")

	    self.umat = umat

	elif(algorithm == 'leastsq'):
	    self.umat = self.oep_leastsq(self.umat)

	#self.umat = self.umat - np.eye( self.umat.shape[ 0 ] ) * np.average( np.diag( self.umat ) )

#	self.umat = self.oep_old(self.umat)

	self.P_imp, self.P_bath = self.verify_scf(self.umat)

	print '|P_imp-P_imp_0|', np.linalg.norm(self.P_imp - self.P_imp0), np.amax(np.absolute(self.P_imp - self.P_imp0))
	print '|P_bath-P_bath_0|', np.linalg.norm(self.P_bath - self.P_bath0), np.amax(np.absolute(self.P_bath - self.P_bath0))

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

	self.ks_frag = self.calc_energy_frag(self.umat, None, self.Ne_frag, self.dim)[5]
        self.ks_env = self.calc_energy_env(self.umat, None, self.Ne_env, self.dim)[5]
	
	if(method == 1):
	    self.P_imp = self.ks_frag.rdm1
	    self.P_bath = self.ks_env.rdm1
	elif(method == 2):
	    print "density partition from input"
	    #self.P_imp = np.dot(np.dot(self.loc2sub[:,:self.dim].T,self.P_frag_loc),self.loc2sub[:,:self.dim])
	    #self.P_bath = np.dot(np.dot(self.loc2sub[:,:self.dim].T,self.P_env_loc),self.loc2sub[:,:self.dim])
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
	diffP = self.P_imp + self.P_bath - self.P_ref
	print "|P_imp + P_bath - P_ref| = ", np.linalg.norm(diffP)
	print "max(P_imp + P_bath - P_ref)", np.amax(np.absolute(diffP))

	'''
	dim = self.dim
	frag_occ = np.zeros([dim],dtype = float)
        for i in range(self.Ne_frag/2):
            frag_occ[i] = 2.0
	self.ints.submo_molden(self.frag_mo, frag_occ, self.loc2sub[:,:dim], 'frag_dens_guess.molden' )

        env_occ = np.zeros([dim],dtype = float)
        for i in range(self.Ne_env/2):
            env_occ[i] = 2.0
	self.ints.submo_molden(self.env_mo, env_occ, self.loc2sub[:,:dim], 'env_dens_guess.molden' )
	'''

    def oep_loop(self, _umat):

	'''
	New OEP scheme
	Outer loop of OEP
	'''
	t0 = (time.clock(),time.time())

	umat = _umat.copy()
	#if(self.P_imp is None):
	self.init_density_partition()

	self.P_imp0 = self.P_imp.copy()
        self.P_bath0 = self.P_bath.copy()

	threshold = self.params.diffP_tol
	maxit = self.params.outer_maxit
	it = 0
	while it < maxit:
	    it += 1
	    print " OEP iteration ", it

	    P_imp_old = self.P_imp.copy()
            P_bath_old = self.P_bath.copy()

	    umat = self.oep_base(umat, True)

	    self.P_imp = self.calc_energy_frag(umat, self.impJK_sub, self.Ne_frag, self.dim)[1]
	    self.P_bath = self.calc_energy_env(umat, self.bathJK_sub, self.Ne_env, self.dim)[1]

	    print "P_imp idem = ", np.linalg.norm(np.dot(self.P_imp,self.P_imp) - 2.0*self.P_imp)

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
	'''
	#umat = umat - np.eye(umat.shape[ 0 ] ) * np.average( np.diag(umat ) )
	#self.P_imp, self.P_bath = self.verify_scf(umat)
	print '|P_imp-P_imp_0|', np.linalg.norm(self.P_imp - self.P_imp0), np.amax(np.absolute(self.P_imp - self.P_imp0))
        print '|P_bath-P_bath_0|', np.linalg.norm(self.P_bath - self.P_bath0), np.amax(np.absolute(self.P_bath - self.P_bath0))
	self.P_imp0 = self.P_imp.copy()
        self.P_bath0 = self.P_bath.copy()

	gtol = copy.copy(self.params.gtol)
	self.params.gtol = 1e-5
	opt_method = copy.copy(self.params.opt_method)
        self.params.opt_method = 'trust-ncg'
	threshold = 3e-5
	smear_sigma = copy.copy(self.smear_sigma)
	self.smear_sigma = 0.0
	l2_lambda = copy.copy(self.params.l2_lambda)
	self.params.l2_lambda = 0.0
        it = 0
	print 'restart OEP with tighter convergence criteria'
        while it < maxit:
            it += 1
            print " OEP iteration ", it

            P_imp_old = self.P_imp.copy()
            P_bath_old = self.P_bath.copy()

            umat = self.oep_base(umat, True)

            self.P_imp = self.calc_energy_frag(umat, self.impJK_sub, self.Ne_frag, self.dim)[1]
            self.P_bath = self.calc_energy_env(umat, self.bathJK_sub, self.Ne_env, self.dim)[1]

            diffP_imp = self.P_imp - P_imp_old
            diffP_bath = self.P_bath - P_bath_old
            #gmax_imp = np.amax(np.absolute(diffP_imp))
            #gmax_bath = np.amax(np.absolute(diffP_bath))
	    gmax_imp = np.linalg.norm(diffP_imp)
	    gmax_bath = np.linalg.norm(diffP_bath)
            print "diffP_max_imp, diffP_max_bath "
            print gmax_imp, gmax_bath

            sys.stdout.flush()
            if(gmax_imp < threshold and gmax_bath < threshold ):
                break

            P_imp_old = None
            P_bath_old = None
	
	self.params.gtol = gtol
	self.params.opt_method = opt_method
	self.smear_sigma = smear_sigma
	self.params.l2_lambda = l2_lambda
	'''
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
	
	print 'in verify_scf'
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


	coredm = self.core1PDM_ao
	ao2sub = self.ao2sub[:,:dim]

	'''
	impJK_sub = self.get_veff_sub(self.P_imp, self.ints, subTEI, self.Kcoeff, self.mf_method, ao2sub, self.ks, self.mol)
        bathJK_sub = self.get_veff_sub(self.P_bath, self.ints, subTEI, self.Kcoeff, self.mf_method, ao2sub, self.ks, self.mol, coredm)
	subOEI1 += impJK_sub
	subOEI2 += bathJK_sub

	P1 = tools.fock2onedm(subOEI1, Ne_frag/2)[0]
	P2 = tools.fock2onedm(subOEI2, Ne_env/2)[0]

	print np.linalg.norm(self.P_imp - P1)
	print np.linalg.norm(self.P_bath - P2)

	exit()
	'''

	frag_coredm_guess = None
        if(self.P_imp is None):
            frag_coredm_guess,mo_coeff = tools.fock2onedm(subOEI1, Ne_frag/2)
	else:
	    frag_coredm_guess = self.P_imp

        mf_frag = qcwrap.qc_scf(Ne_frag,dim,self.mf_method,mol=self.mol_frag,oei=subOEI1,tei=subTEI,\
				dm0=frag_coredm_guess,coredm=0.0,ao2sub=ao2sub, smear_sigma = 0.0)
	#mf_frag.init_guess =  'minao'
	mf_frag.conv_check = False
        mf_frag.runscf()
        FRAG_energy = mf_frag.elec_energy
        FRAG_1RDM = mf_frag.rdm1
	frag_mo = mf_frag.mo_coeff
	frag_occ = mf_frag.mo_occ

	#mf_frag.stability(internal=True, external=False, verbose=5)

	env_coredm_guess = None
	if(self.P_bath is None):
            env_coredm_guess,mo_coeff = tools.fock2onedm(subOEI2, Ne_env/2)
	else:
	    env_coredm_guess = self.P_bath

	mf_env = qcwrap.qc_scf(Ne_env,dim,self.mf_method,mol=self.mol_env,oei=subOEI2,tei=subTEI,\
			       dm0=env_coredm_guess,coredm=coredm,ao2sub=ao2sub, smear_sigma = 0.0)
	#mf_env.init_guess =  'minao'
	mf_env.conv_check = False #temp
        mf_env.runscf()
        ENV_energy = mf_env.elec_energy
        ENV_1RDM = mf_env.rdm1
	env_mo = mf_env.mo_coeff
	env_occ = mf_env.mo_occ

	#self.ints.submo_molden(env_mo, env_occ, self.loc2sub, "mo_env.molden" )
	#dm_ao = tools.dm_sub2ao(ENV_1RDM, ao2sub)
        #cubegen.density(self.mol, "env_dens.cube", dm_ao, nx=100, ny=100, nz=100)

	#tools.MatPrint(umat, "umat")
	#tools.MatPrint(FRAG_1RDM,"P_imp")
	#tools.MatPrint(tools.dm_sub2ao(FRAG_1RDM, ao2sub), "P_imp_ao")

	#tools.MatPrint(frag_mo,'frag_mo')

	#mo_i, mo_e = mf_env.stability(internal=True, external=False, verbose=5)
	#P_new = 2.0*np.dot(mo_i[:,:Ne_env/2],mo_i[:,:Ne_env/2].T)
	#print np.linalg.norm(ENV_1RDM - P_new)
	#mf_env = qcwrap.qc_scf(Ne_env,dim,self.mf_method,mol=None,oei=subOEI2,tei=subTEI,dm0=P_new,coredm=coredm,ao2sub=ao2sub)
        #mf_env.runscf()
        #ENV_energy = mf_env.elec_energy
        #ENV_1RDM = mf_env.rdm1

	print "scf energies:"
        print FRAG_energy-np.trace(np.dot(FRAG_1RDM,umat)),\
	      ENV_energy-np.trace(np.dot(ENV_1RDM,umat)), \
	      FRAG_energy+ENV_energy - np.trace(np.dot(FRAG_1RDM+ENV_1RDM,umat))
	#print np.trace(np.dot(FRAG_1RDM,umat)), np.trace(np.dot(ENV_1RDM,umat)),np.trace(np.dot(FRAG_1RDM+ENV_1RDM,umat))
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

	self.frag_mo = mf_frag.mo_coeff
	self.env_mo = mf_env.mo_coeff

	print 'check orthogonality'
	ortho = np.dot(mf_frag.mo_coeff[:,:Ne_frag/2].T, mf_env.mo_coeff[:,:Ne_env/2])
	print np.linalg.norm(ortho), np.amax(np.absolute(ortho))
        sys.stdout.flush()

	return (FRAG_1RDM, ENV_1RDM)


    def calc_energy_frag(self, umat, impJK_sub, Ne_frag, dim):

        FRAG_1RDM = np.zeros([dim,dim], dtype = float)
        FRAG_energy = 0.0
	mo_coeff = None
	mo_energy = None
	mo_occ = None

        ops = self.ops
        subKin = ops["subKin"]
        subVnuc1 = ops["subVnuc1"]
        subVnuc_bound1 = ops["subVnuc_bound1"]
        subCoreJK = ops["subCoreJK"]
        subTEI = ops["subTEI"]

	mf_frag = None
        if( impJK_sub is None):  #normal SCF
            subOEI1 = subKin+subVnuc1+subVnuc_bound1+umat

	    #initial guess
            #dm_guess,mo_coeff = tools.fock2onedm(subOEI1, Ne_frag/2)
            #dm_guess = self.P_imp
	    dm_guess = None
            coredm = self.core1PDM_ao
            ao2sub = self.ao2sub[:,:dim]
            mf_frag = qcwrap.qc_scf(Ne_frag,dim,self.mf_method,mol=self.mol_frag,oei=subOEI1,tei=subTEI,\
                                    dm0=dm_guess,coredm=0.0,ao2sub=ao2sub,smear_sigma = self.smear_sigma)
	    mf_frag.init_guess = 'minao'
            mf_frag.runscf()
            FRAG_energy = mf_frag.elec_energy
            FRAG_1RDM = mf_frag.rdm1
	    mo_coeff = mf_frag.mo_coeff
	    mo_energy = mf_frag.mo_energy
	    mo_occ = mf_frag.mo_occ

	    #self.ints.submo_molden(mo_coeff, mo_occ, self.loc2sub, 'frag_sub.molden' )

	else:  #non-self-consistent SCF
            subOEI1 = subKin + subVnuc1 + subVnuc_bound1 + impJK_sub + umat
            FRAG_energy, FRAG_1RDM, mo_coeff,mo_energy,mo_occ= qcwrap.pyscf_rhf.scf_oei( subOEI1, dim, Ne_frag, self.smear_sigma)

	return (FRAG_energy, FRAG_1RDM, mo_coeff,mo_energy,mo_occ, mf_frag)


    def calc_energy_env(self, umat, bathJK_sub, Ne_env, dim):

        ENV_1RDM = np.zeros([dim,dim], dtype = float)
        ENV_energy = 0.0
	mo_coeff = None
        mo_energy = None
        mo_occ = None

        ops = self.ops
        subKin = ops["subKin"]
        subVnuc2 = ops["subVnuc2"]
        subVnuc_bound2 = ops["subVnuc_bound2"]
        subCoreJK = ops["subCoreJK"]
        subTEI = ops["subTEI"]

	mf_env = None
        if( bathJK_sub is None):  #normal SCF
            subOEI2 = subKin+subVnuc2+subVnuc_bound2+subCoreJK+umat
            coredm = self.core1PDM_ao
            ao2sub = self.ao2sub[:,:dim]
            if(Ne_env > 0):
                #dm_guess,mo_coeff = tools.fock2onedm(subOEI2, Ne_env/2)
                #dm_guess = self.P_bath
		dm_guess = None

                mf_env = qcwrap.qc_scf(Ne_env,dim,self.mf_method,mol=self.mol_env,oei=subOEI2,tei=subTEI,\
                                       dm0=dm_guess,coredm=coredm,ao2sub=ao2sub, smear_sigma = self.smear_sigma)
		mf_env.init_guess = 'minao'
                mf_env.runscf()
                ENV_energy = mf_env.elec_energy
                ENV_1RDM = mf_env.rdm1
                mo_coeff = mf_env.mo_coeff
		mo_energy = mf_env.mo_energy
                mo_occ = mf_env.mo_occ

		#self.ints.submo_molden(mo_coeff, mo_occ, self.loc2sub, 'env_sub.molden' )

        else:  #non-self-consistent SCF
            subOEI2 = subKin + subVnuc2 + subVnuc_bound2 + subCoreJK + bathJK_sub + umat
            if(Ne_env > 0):
                ENV_energy, ENV_1RDM, mo_coeff,mo_energy,mo_occ = qcwrap.pyscf_rhf.scf_oei( subOEI2, dim, Ne_env, self.smear_sigma)

	return (ENV_energy, ENV_1RDM, mo_coeff,mo_energy,mo_occ, mf_env)



    def get_veff_sub(self, P, ints, subTEI, Kcoeff, mf_method, ao2sub, ks, mol, coredm_ao=0.0):

	JK_sub = ints.impJK_sub( P, subTEI, Kcoeff)
        if(mf_method != 'hf'):
            vxc_imp_ao = qcwrap.pyscf_rks.get_vxc(ks, mol, coredm_ao + tools.dm_sub2ao(np.asarray(P), ao2sub))[2]
            JK_sub += tools.op_ao2sub(vxc_imp_ao, ao2sub)

	return JK_sub

    def oep_base(self, umat, nonscf = True):

	P_ref = self.P_ref
	ops = self.ops
        dim = self.dim

        x = tools.mat2vec(umat, dim)

        Ne_frag = self.Ne_frag
        Ne_env = self.Ne_env

        ints = self.ints
	subTEI = ops["subTEI"]
	if( nonscf == True):
	    ao2sub = self.ao2sub[:,:dim]
            coredm = self.core1PDM_ao
	    self.impJK_sub = self.get_veff_sub(self.P_imp, ints, subTEI, self.Kcoeff, self.mf_method, ao2sub, self.ks_frag, self.mol_frag)
	    self.bathJK_sub = self.get_veff_sub(self.P_bath, ints, subTEI, self.Kcoeff, self.mf_method, ao2sub, self.ks_env, self.mol_env, coredm)


	_args = [P_ref, dim, Ne_frag, Ne_env]
	_args.append(self.impJK_sub)
	_args.append(self.bathJK_sub)
	_args = tuple(_args)

	opt_method = self.params.opt_method
	result = None
	if( opt_method == 'BFGS' or opt_method == 'L-BFGS-B'):
	    result = self.oep_bfgs(x, _args)
	elif( opt_method == 'trust-krylov' or opt_method == 'trust-ncg' or opt_method == 'trust-exact' or opt_method == 'Newton-CG'):
	    result = self.oep_cg(x, _args)

        x = result.x
        umat = tools.vec2mat(x, dim)

        return umat


    def oep_cg(self, x, _args):

	gtol = self.params.gtol
	maxit = self.params.maxit
	algorithm = self.params.opt_method

	res = optimize.minimize(self.cost_wuyang, x, args=_args, method=algorithm,jac=True, hess=self.hess_wuyang, \
		       options={'maxiter': maxit, 'gtol':gtol, 'disp': True})

	return res


    def oep_bfgs(self, x, _args):

	maxit = self.params.maxit
        gtol = self.params.gtol
        ftol = self.params.ftol
	algorithm = self.params.opt_method 

        result = optimize.minimize(self.cost_wuyang,x,args=_args,method=algorithm, jac=True, \
				   options={'disp': True, 'maxiter': maxit, 'gtol':gtol, 'ftol':ftol, 'maxcor':10} )

	return result

    def opt_umat_2(self,umat, tol=1e-6):

        ops = self.ops
        Ne_frag = self.Ne_frag
        Ne_env = self.Ne_env
        dim = self.dim

        subTEI = ops["subTEI"]
        subOEI1 = ops["subKin"]+ops["subVnuc1"]+ops["subVnuc_bound1"] + umat
        subOEI2 = ops["subKin"]+ops["subVnuc2"]+ops["subVnuc_bound2"]+ops["subCoreJK"] + umat

        coredm = self.core1PDM_ao
        ao2sub = self.ao2sub[:,:dim]

        #frag_coredm_guess = tools.fock2onedm(subOEI1, Ne_frag/2)[0]
	frag_coredm_guess = self.P_imp
        mf_frag = qcwrap.qc_scf(Ne_frag,dim,self.mf_method,mol=self.mol,oei=subOEI1,tei=subTEI,\
                                dm0=frag_coredm_guess,coredm=0.0,ao2sub=ao2sub, smear_sigma = 0.0)
        mf_frag.runscf()

        #env_coredm_guess = tools.fock2onedm(subOEI2, Ne_env/2)[0]
	env_coredm_guess = self.P_bath
        mf_env = qcwrap.qc_scf(Ne_env,dim,self.mf_method,mol=self.mol,oei=subOEI2,tei=subTEI,\
                               dm0=env_coredm_guess,coredm=coredm,ao2sub=ao2sub, smear_sigma = 0.0)
        mf_env.runscf()

	print np.linalg.norm(mf_frag.rdm1-self.P_imp)
	print np.linalg.norm(mf_env.rdm1-self.P_bath)

	diffP = mf_frag.rdm1 + mf_env.rdm1 - self.P_ref
        print "|P_frag + P_env - P_ref| = ", np.linalg.norm(diffP)
        print "max element of (P_frag + P_env - P_ref) = ", np.amax(np.absolute(diffP) )
	self.P_imp = mf_frag.rdm1
	self.P_bath = mf_env.rdm1

	#minimize |umat|^2
        x0 = tools.mat2vec(umat, dim)
        size = dim*(dim+1)/2
        hess_frag = calc_hess(mf_frag.mo_coeff, mf_frag.mo_energy, mf_frag.mo_occ, size, Ne_frag/2, dim)
        hess_env = calc_hess(mf_env.mo_coeff, mf_env.mo_energy, mf_env.mo_occ, size, Ne_env/2, dim)

        #u_f, s_f, vh_f = np.linalg.svd(hess_frag)
        #u_e, s_e, vh_e = np.linalg.svd(hess_env)
        u_f, s_f, vh_f = mkl_svd(hess_frag)
        u_e, s_e, vh_e = mkl_svd(hess_env)

        rankf = tools.rank(s_f,tol)
        ranke = tools.rank(s_e,tol)

        print 'svd of hess'
        print size-rankf, size-ranke
        print s_f[rankf-4:rankf+4]
        print s_e[ranke-4:ranke+4]

        if (rankf >= size or ranke >= size):
            return (FRAG_1RDM, ENV_1RDM)

        x = np.zeros((size))
        y = np.zeros((size))
        v_f = vh_f.T
        v_e = vh_e.T

        #tools.MatPrint(v_f, 'v_f')
        #tools.MatPrint(v_e, 'v_e')

        v_fe = np.concatenate((v_f[:,rankf:], -v_e[:,ranke:]), axis=1)
        v_fe_fort = np.require(v_fe, requirements=['A', 'O', 'W', 'F'])

        #uu, ss, vvh = np.linalg.svd(v_fe)
        uu, ss, vvh = mkl_svd(v_fe_fort, 2)
        rankfe = tools.rank(ss,tol)
        print v_fe.shape, rankfe
        #np.set_printoptions(threshold=np.inf)
        #print ss

        if (rankfe >= v_fe.shape[-1]):
            return (FRAG_1RDM, ENV_1RDM)

	print ss[rankfe-1:rankfe+1]

        vv = vvh[rankfe:,:].T
        #zero = np.dot(v_fe, vv)
        #print np.linalg.norm(zero),np.amax(np.absolute(zero)) 

        vint = np.dot(v_f[:,rankf:],vv[:(size-rankf),:])
	#vint1 = np.dot(v_e[:,ranke:],vv[(size-rankf):,:])
        for i in range(vint.shape[-1]):
            vint[:,i] = vint[:,i]/np.linalg.norm(vint[:,i])

	print 'ortho of vint'
        zero = np.dot(vint.T, vint) - np.eye(vint.shape[-1])
        print np.linalg.norm(zero),np.amax(np.absolute(zero)) 

	sys.stdout.flush()

        n = vint.shape[-1]
        c = np.zeros((n))
        res = self.minimize_umat_2(c,x0,vint)
        c = res.x
	print '|c| = ', np.linalg.norm(c)
        for i in range(n):
            x0 += c[i] * vint[:,i]

        umat = tools.vec2mat(x0, dim)
        #xmat = xmat - np.eye( xmat.shape[ 0 ] ) * np.average( np.diag( xmat ) )

        #tools.MatPrint(umat,'umat')
        print '|umat| = ', np.linalg.norm(umat)

	sys.stdout.flush()
	return umat



    def minimize_umat_2(self,c,x0,vint):
	
	_args = (x0,vint)
	ftol = 1e-9
	gtol = 1e-6
	maxit = 50

	#cons = ({'type': 'ineq', 'fun': self.constr_umat_2, 'jac':self.constr_umat_2_grad,'args': _args})
		
	#result = optimize.minimize(self.cost_umat_2,c,args=_args,method='SLSQP', jac=True, constraints=cons,\
        #                           options={'disp': True, 'maxiter': maxit,'ftol':ftol} )

	result = optimize.minimize(self.cost_umat_2, c, args=_args, method='L-BFGS-B', jac=True,\
                                   options={'disp': True, 'maxiter': maxit,'ftol':ftol, 'gtol':gtol} )

        return result

    def constr_umat_2(self, c, x0, vint):
	
	u = x0.copy()

        n = len(c)
        for i in range(n):
            u += c[i] * vint[:,i]

	u -= x0	
	delta_2 = np.dot(u,u)

	f = 0.01 - delta_2

	return f

    def constr_umat_2_grad(self, c, x0, vint):

	u = x0.copy()

        n = len(c)
        for i in range(n):
            u += c[i] * vint[:,i]

        u -= x0
        delta_2 = np.dot(u,u)

	g = np.zeros((n))
	for i in range(n):
	    g[i] = -2.0*np.dot(u,vint[:,i])


	return g

    def cost_umat_2(self,c,x0,vint):

	u = x0.copy()

	n = len(c)
	for i in range(n):
	    u += c[i] * vint[:,i]

	f = 2.0*np.dot(u,u)

	index = 0
	for i in range(self.dim):
	    f -= u[index]*u[index]
	    index += self.dim-i

	g = np.zeros((n))
	for i in range(n):
	    g[i] = 4.0*np.dot(u,vint[:,i])
	    index = 0
	    for j in range(self.dim):
		g[i] -= 2.0*u[index]*vint[index,i]
		index += self.dim-j

	return (f,g)


    def cost_wuyang(self, x, P_ref, dim, Ne_frag, Ne_env, impJK_sub, bathJK_sub):

	#t0 = (time.clock(),time.time())

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

	    #frag_coredm_guess,mo_coeff = tools.fock2onedm(subOEI1, Ne_frag/2)
	    frag_coredm_guess = self.P_imp

	    coredm = self.core1PDM_ao
            ao2sub = self.ao2sub[:,:dim]
            mf_frag = qcwrap.qc_scf(Ne_frag,dim,self.mf_method,mol=self.mol_frag,oei=subOEI1,tei=subTEI,\
				    dm0=frag_coredm_guess,coredm=0.0,ao2sub=ao2sub,smear_sigma = self.smear_sigma)
            mf_frag.runscf()
	    FRAG_energy = mf_frag.elec_energy
	    FRAG_1RDM = mf_frag.rdm1
	    #FRAG_mo = mf_frag.mo_coeff


	    #mf_frag.stability(internal=True, external=False, verbose=5)

	    if(Ne_env > 0):
	        #env_coredm_guess,mo_coeff = tools.fock2onedm(subOEI2, Ne_env/2)
		env_coredm_guess = self.P_bath
		#env_coredm_guess = None
		mf_env = qcwrap.qc_scf(Ne_env,dim,self.mf_method,mol=self.mol_env,oei=subOEI2,tei=subTEI,\
				       dm0=env_coredm_guess,coredm=coredm,ao2sub=ao2sub, smear_sigma = self.smear_sigma)
                mf_env.runscf()
                ENV_energy = mf_env.elec_energy
                ENV_1RDM = mf_env.rdm1
		#ENV_mo = mf_env.mo_coeff

		#mf_env.stability(internal=True, external=False, verbose=5)


	else:  #non-self-consistent SCF
            subOEI1 = subKin + subVnuc1 + subVnuc_bound1 + impJK_sub + umat
            subOEI2 = subKin + subVnuc2 + subVnuc_bound2 + subCoreJK + bathJK_sub + umat
	    FRAG_energy, FRAG_1RDM, tmp,tmp,tmp = qcwrap.pyscf_rhf.scf_oei( subOEI1, dim, Ne_frag, self.smear_sigma)
	    if(Ne_env > 0):
	        ENV_energy, ENV_1RDM, tmp,tmp,tmp = qcwrap.pyscf_rhf.scf_oei( subOEI2, dim, Ne_env, self.smear_sigma)


	#tools.MatPrint(frag_mo,"frag_mo")
	#tools.MatPrint(env_mo,"env_mo")

	self.P_imp = FRAG_1RDM
        self.P_bath = ENV_1RDM

	#self.frag_mo = FRAG_mo
        #self.env_mo = ENV_mo

	if(self.params.oep_print >= 3):
	    print "number of electrons in fragment", np.sum(np.diag(FRAG_1RDM))
            tools.MatPrint(FRAG_1RDM, 'fragment density')
	    print "number of electrons in environment", np.sum(np.diag(ENV_1RDM))
            tools.MatPrint(ENV_1RDM, 'environment density')


        diffP = FRAG_1RDM + ENV_1RDM - P_ref
        energy = FRAG_energy + ENV_energy - np.trace(np.dot(P_ref,umat))

        grad = tools.mat2vec(diffP, dim)
        grad = -1.0 * grad

	gtol = self.params.gtol
	#size = dim*(dim+1)/2
	#for i in range(size):
	#    if(abs(grad[i]) < gtol): #numerical error may accumulate
#		grad[i] = 0.0

        print "2-norm (grad),       max(grad):"
        print np.linalg.norm(grad), ", ", np.amax(np.absolute(grad))


	l2_f, l2_g = self.l2_reg(x)

	f = -energy + l2_f
	grad = grad + l2_g


	print 'W = ', f

        return (f, grad)



    def l2_reg(self, x):

	target = 2.5

	dim = self.dim
	u = tools.vec2mat(x, dim)
        u_norm = np.linalg.norm(u)
	f = self.params.l2_lambda * ((u_norm - target)**2)

	if(u_norm <1e-5):
            g = x*0.0
        else:
	    g = x*(2.0*self.params.l2_lambda/u_norm)

        index = 0
        for j in range(dim):
            g[index] = g[index]/2.0
            index += dim-j

	g *= 2.0*(u_norm-target)

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



    def hess_wuyang(self, x, P_ref, dim, Ne_frag, Ne_env, impJK_sub, bathJK_sub):

	size = dim*(dim+1)/2
        umat = tools.vec2mat(x, dim)

        ops = self.ops
        subKin = ops["subKin"]
        subVnuc1 = ops["subVnuc1"]
        subVnuc2 = ops["subVnuc2"]
        subVnuc_bound1 = ops["subVnuc_bound1"]
        subVnuc_bound2 = ops["subVnuc_bound2"]
        subCoreJK = ops["subCoreJK"]
        subTEI = ops["subTEI"]

        subOEI1 = subKin + subVnuc1 + subVnuc_bound1 + impJK_sub + umat
        subOEI2 = subKin + subVnuc2 + subVnuc_bound2 + subCoreJK + bathJK_sub + umat

        FRAG_energy, FRAG_1RDM, mo_coeff_frag, mo_energy_frag, mo_occ_frag = qcwrap.pyscf_rhf.scf_oei( subOEI1, dim, Ne_frag, self.smear_sigma)
	hess_frag = calc_hess(mo_coeff_frag, mo_energy_frag, mo_occ_frag, size, Ne_frag/2, dim)

        ENV_energy, ENV_1RDM, mo_coeff_env, mo_energy_env, mo_occ_env = qcwrap.pyscf_rhf.scf_oei( subOEI2, dim, Ne_env, self.smear_sigma)
	hess_env = calc_hess(mo_coeff_env, mo_energy_env, mo_occ_env, size, Ne_env/2, dim)
	
	hess = hess_frag + hess_env
	return hess


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
	hess = -2.0*hess.T

	return hess


def calc_hess(jCa,orb_Ea,mo_occ, size,NOcc,NOrb):

    t0 = (time.clock(),time.time())
    mo_coeff = np.reshape(jCa, (NOrb*NOrb), 'F')
    hess = np.ndarray((size,size),dtype=float, order='F')

    nthread  = lib.num_threads()

    lumo_occ = mo_occ[NOcc]
    if(lumo_occ < 1e-10):
        libhess.calc_hess_dm_fast(hess.ctypes.data_as(ctypes.c_void_p), \
			      mo_coeff.ctypes.data_as(ctypes.c_void_p), orb_Ea.ctypes.data_as(ctypes.c_void_p), \
			      ctypes.c_int(size), ctypes.c_int(NOrb), ctypes.c_int(NOcc), ctypes.c_int(nthread))
    else:
	libhess.calc_hess_dm_fast_frac(hess.ctypes.data_as(ctypes.c_void_p), \
                              mo_coeff.ctypes.data_as(ctypes.c_void_p), orb_Ea.ctypes.data_as(ctypes.c_void_p), \
			      mo_occ.ctypes.data_as(ctypes.c_void_p),\
                              ctypes.c_int(size), ctypes.c_int(NOrb), ctypes.c_int(NOcc), ctypes.c_int(nthread))


    t1 = tools.timer("hessian construction", t0)

    return hess


def mkl_svd(A, algorithm = 1):

    t0 = (time.clock(),time.time())

    m = A.shape[0]
    n = A.shape[1]

    U = np.ndarray((m,m), dtype=float, order='F')
    VT = np.ndarray((n,n), dtype=float, order='F')
    sigma = np.ndarray((min(m,n)), dtype=float, order='F')

    info = np.zeros((1),dtype = int)

    libsvd.mkl_svd(A.ctypes.data_as(ctypes.c_void_p),\
		   sigma.ctypes.data_as(ctypes.c_void_p),\
		   U.ctypes.data_as(ctypes.c_void_p),\
		   VT.ctypes.data_as(ctypes.c_void_p),\
		   ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(algorithm), \
		   info.ctypes.data_as(ctypes.c_void_p))

    t1 = tools.timer("mkl_svd", t0)
    if(info[0] != 0):
	print 'mkl_svd info = ',info[0] 
	raise Exception("mkl_svd failed!")

    return (U,sigma,VT)
