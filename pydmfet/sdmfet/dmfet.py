import time,copy
import numpy as np
from pydmfet import subspac,oep,tools,qcwrap
from pyscf import cc

class DMFET:

    def __init__(self, ints, cluster, impAtom, Ne_frag, boundary_atoms=None, umat = None,\
		 sub_threshold = 1e-13, oep_params = oep.OEPparams(), ecw_method = 'HF', mf_method = 'HF',do_dfet = False):

        self.ints = ints
        self.cluster = cluster
        self.impAtom = impAtom
        self.Ne_frag = Ne_frag
	self.Ne_frag_orig = copy.copy(self.Ne_frag)
	self.boundary_atoms = boundary_atoms

        self.dim_frag = np.sum(self.cluster)
        #self.dim_env = self.cluster.size - self.dim_frag

        self.ecw_method = ecw_method
	self.mf_method = mf_method
        self.do_dfet = do_dfet
	self.sub_threshold = sub_threshold

        #construct subspace
        self.OneDM_loc = self.ints.build_1pdm_loc()
        self.dim_imp, self.dim_bath, self.dim_imp_virt, self.Occupations, self.loc2sub, eignimp, eignbath = subspac.construct_subspace(self.OneDM_loc, self.cluster, self.sub_threshold)
        self.dim_sub = self.dim_imp + self.dim_bath

	print 'dimension of subspace: imp_occ, bath, imp_virt', 
	print self.dim_imp - self.dim_imp_virt, self.dim_bath, self.dim_imp_virt

        #construct core determinant
	idx = self.dim_frag + self.dim_bath
        self.core1PDM_loc, self.Nelec_core, Norb_imp_throw, self.frag_core1PDM_loc = subspac.build_core(self.Occupations, self.loc2sub, idx)

	self.Ne_frag = self.Ne_frag - Norb_imp_throw*2
        self.Ne_env = self.ints.Nelec - self.Ne_frag - self.Nelec_core
	
	print 'Ne_frag, Ne_env, Ne_core'
	print self.Ne_frag, self.Ne_env, self.Nelec_core

        self.umat = umat
	self.P_imp = None
	self.P_bath = None

        dim = self.dim_sub
        loc2sub = self.loc2sub
	self.P_ref_sub = np.dot(np.dot(loc2sub[:,:dim].T, self.OneDM_loc - self.core1PDM_loc),loc2sub[:,:dim])

        self.oep_params = oep_params
	self.ops = None

	#self.ops = self.build_ops( self.core1PDM_loc, dim)
	#self.total_scf_energy()
	#exit()


    def build_ops(self, core1PDM_loc, dim):

        t0 = (time.clock(), time.time())

        ints = self.ints
        loc2sub = self.loc2sub
        impAtom = self.impAtom
        boundary_atoms = self.boundary_atoms

        subKin = ints.frag_kin_sub( impAtom, loc2sub, dim )
        subVnuc1 = ints.frag_vnuc_sub( impAtom, loc2sub, dim)
        subVnuc2 = ints.frag_vnuc_sub( 1-impAtom, loc2sub, dim )

        subVnuc_bound = ints.bound_vnuc_sub(boundary_atoms, loc2sub, dim )

        subCoreJK = ints.coreJK_sub( loc2sub, dim, core1PDM_loc )
        subTEI = ints.dmet_tei( loc2sub, dim )

        ops = [subKin,subVnuc1,subVnuc2,subVnuc_bound,subCoreJK,subTEI]


        tools.timer("dmfet.build_ops",t0)
        return ops


    def calc_umat(self):
      
	dim = self.dim_sub
	self.ops = self.build_ops(self.core1PDM_loc, dim) 
        myoep = oep.OEP(self, self.oep_params)

	myoep.params.gtol = myoep.params.gtol * 100.0
	myoep.params.l2_lambda = myoep.params.gtol * 1.0 #test L2 regularization 
        myoep.kernel()
        self.umat = myoep.umat
	self.P_imp = myoep.P_imp
	self.P_bath = myoep.P_bath

	tools.MatPrint(self.P_imp,"P_imp")
	tools.MatPrint(self.P_bath,"P_bath")
	tools.MatPrint(self.P_imp+self.P_bath,"P_imp+P_bath")
	tools.MatPrint(self.umat,"umat")

	P1 = self.P_imp.copy()
	u1 = self.umat.copy()

	#myoep.umat = None
	myoep.params.algorithm = 'split'
	myoep.params.gtol = myoep.params.gtol * 0.01
	myoep.params.l2_lambda = 0.0
	myoep.kernel()
        self.umat = myoep.umat
        self.P_imp = myoep.P_imp
        self.P_bath = myoep.P_bath

        tools.MatPrint(self.P_imp,"P_imp")
        tools.MatPrint(self.P_bath,"P_bath")
        tools.MatPrint(self.P_imp+self.P_bath,"P_imp+P_bath")
        tools.MatPrint(self.umat,"umat")

	P2 = self.P_imp.copy()
	u2 = self.umat.copy()

	
	print np.linalg.norm(P1-P2)
	tools.MatPrint(P1-P2,"P_imp_2011 - P_imp_split")
	tools.MatPrint(u1-u2,"umat_2011 - umat_split")

    def embedding_potential(self):

        if(self.umat is None):
            self.calc_umat()
            
        return self.umat



    def correction_energy(self):
	self.total_scf_energy()

        energy = 0.0

        if(self.umat is None):
            self.calc_umat()

	print "Performing ECW energy calculation"

	#resize subspace
	dim = self.dim_frag + self.dim_bath
	if(dim != self.dim_sub):
	    coredm = self.core1PDM_loc - self.frag_core1PDM_loc
	    self.ops = self.build_ops(coredm, dim)

	umat = self.umat.copy()
	npad = dim - self.dim_sub
	umat = np.pad(umat, ((0,npad),(0,npad)), mode='constant', constant_values=0.0)


	if(self.ecw_method.lower() == 'hf'):
	    energy = self.hf_energy(umat,dim)
	elif(self.ecw_method.lower() == 'ccsd'):
	    energy = self.ccsd_energy()
	else:
	    raise Exception("ecw_method not supported!")


        return energy


    def core_energy(self):

        core1PDM_loc = self.core1PDM_loc
        oei_loc = self.ints.loc_oei()
        coreJK_loc = self.ints.coreJK_loc(core1PDM_loc)

        core_energy = np.trace(np.dot(core1PDM_loc,oei_loc)) + 0.5*np.trace(np.dot(core1PDM_loc,coreJK_loc))
        return core_energy


    def total_scf_energy(self):

        energy = 0.0

        dim = self.dim_sub
        Ne_frag = self.Ne_frag
        Ne_env = self.Ne_env
        ops = self.ops

        subKin = ops[0]
        subVnuc1 = ops[1]
        subVnuc2 = ops[2]
        subCoreJK = ops[4]
        subTEI = ops[-1]


        subOEI = subKin+subVnuc1+subVnuc2+subCoreJK
        energy, onedm, mo = qcwrap.pyscf_rhf.scf( subOEI, subTEI, dim, Ne_frag+Ne_env, self.P_ref_sub, self.mf_method)


        #print onedm - self.P_ref_sub
        energy = energy + self.core_energy() + self.ints.const()

        print "total scf energy = ",energy
        return energy


    def imp_scf_energy(self):

        energy = 0.0

        dim = self.dim_sub
        Ne_frag = self.Ne_frag
        ops = self.ops

        subKin = ops[0]
        subVnuc1 = ops[1]
        subVnuc_bound = ops[3]
        subCoreJK = ops[4]
        subTEI = ops[-1]
        umat = self.umat

        subOEI = subKin+subVnuc1+subVnuc_bound+subCoreJK+umat
        energy, onedm, mo = qcwrap.pyscf_rhf.scf( subOEI, subTEI, dim, Ne_frag, self.P_imp, self.mf_method)

	print "diffP = ",np.linalg.norm(self.P_imp - onedm)

        #energy = energy - np.trace(np.dot(onedm,umat+subVnuc_bound)) 

        print "embeded imp scf (electron) energy = ",energy
        return energy


    def imp_scf_energy2(self):

	dim = self.dim_frag + self.dim_bath
	Nelec = self.Ne_frag_orig


    def hf_energy(self, umat, dim):

        print "ECW method is HF"	
        energy = 0.0

        Ne_frag = self.Ne_frag_orig
        ops = self.ops

        subKin = ops[0]
        subVnuc1 = ops[1]
        subVnuc_bound = ops[3]
        subCoreJK = ops[4]
        subTEI = ops[-1]

        subOEI = subKin+subVnuc1+subVnuc_bound+subCoreJK+umat

	dim_sub = self.dim_sub
	npad = dim - dim_sub
	P_guess = np.pad(self.P_imp, ((0,npad),(0,npad)), mode='constant', constant_values=0.0)

	norb = (Ne_frag - self.Ne_frag)/2
	for i in range(norb):
	    index = dim_sub+i
	    P_guess[index][index] = 2.0

        energy, onedm, mo = qcwrap.pyscf_rhf.scf( subOEI, subTEI, dim, Ne_frag, P_guess)

	P = np.dot(np.dot(self.loc2sub[:,:dim_sub],self.P_imp),self.loc2sub[:,:dim_sub].T)
        P = np.dot(np.dot(self.loc2sub[:,:dim].T,P),self.loc2sub[:,:dim])
	print np.linalg.norm(P_guess-P)
	print np.linalg.norm(onedm-P)
	exit()

        tools.MatPrint(self.P_imp, "P_imp")
        tools.MatPrint(P,"P_imp+virt")
	print energy
	exit()
        imp_scf_energy = self.imp_scf_energy()

        energy = energy - imp_scf_energy

        print "dmfet correction energy = ",energy
        return energy

    def ccsd_energy(self):

        print "ECW method is CCSD(T)"
	energy = 0.0

        dim = self.dim_sub
        Ne_frag = self.Ne_frag
        ops = self.ops

        subKin = ops[0]
        subVnuc1 = ops[1]
        subVnuc_bound = ops[3]
        subCoreJK = ops[4]
        subTEI = ops[-1]
        umat = self.umat

        subOEI = subKin+subVnuc1+subVnuc_bound+subCoreJK+umat

        mf = qcwrap.pyscf_rhf.rhf( subOEI, subTEI, dim, Ne_frag, self.P_imp)

	onedm = mf.make_rdm1() 
	print "diffP = ",np.linalg.norm(self.P_imp - onedm)

	print mf.e_tot
        mycc = cc.CCSD(mf).run()
	et = 0.0
        et = mycc.ccsd_t()
        e_hf = mf.e_tot
	print mycc.e_corr + et

        e_ccsd = e_hf + mycc.e_corr + et
	print e_ccsd

        imp_scf_energy = self.imp_scf_energy()

        energy = e_ccsd - imp_scf_energy

        print "dmfet correction energy = ",energy
	return energy
