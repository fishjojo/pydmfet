import time,copy
import numpy as np
from pydmfet import subspac,oep,tools,qcwrap,libgen
import pyscf
from pyscf import mp,cc
from pyscf.tools import cubegen
from functools import reduce
#import scipy

class DMFET:

    def __init__(self, mf_full,mol_frag,mol_env, ints, cluster, impAtom, Ne_frag, boundary_atoms=None, boundary_atoms2=None, \
                 umat = None, P_frag_ao = None, P_env_ao = None, \
                 dim_imp =None, dim_bath = None, dim_big = None, smear_sigma = 0.0, \
                 sub_threshold = 1e-13, oep_params = oep.OEPparams(), ecw_method = 'HF', mf_method = 'HF', ex_nroots = 1, \
                 plot_dens=True, plot_mo = True, deproton=None, use_bath_virt = False, use_umat_ao = False, scf_max_cycle=50, frac_occ_tol=1e-6):

        self.use_suborb = True
        self.ints = ints
        self.mf_full = mf_full
        self.mol = self.mf_full.mol
        self.cluster = cluster
        self.ao_bas_tab_frag = self.cluster
        self.impAtom = impAtom
        self.Ne_frag = Ne_frag
        self.Ne_frag_orig = copy.copy(self.Ne_frag)
        self.boundary_atoms = boundary_atoms
        self.boundary_atoms2 = boundary_atoms2
        self.deproton = deproton
        self.plot_dens = plot_dens
        self.plot_mo = plot_mo
        self.scf_max_cycle = scf_max_cycle

        self.P_ref_ao = mf_full.make_rdm1()
        self.P_frag_ao = P_frag_ao
        self.P_env_ao = P_env_ao

        self.mol_frag = mol_frag
        self.mol_env = mol_env

        self.smear_sigma = smear_sigma

        self.use_umat_ao = use_umat_ao

        self.dim_frag = np.sum(self.cluster)
        #self.dim_env = self.cluster.size - self.dim_frag

        self.ecw_method = ecw_method.lower()
        self.mf_method = mf_method.lower()
        self.ex_nroots = ex_nroots
        self.sub_threshold = sub_threshold

        self.Kcoeff = 1.0
        if(self.mf_method != 'hf'):
            ks = self.mf_full
            self.Kcoeff = ks._numint.hybrid_coeff(ks.xc, spin=self.mol.spin)

        #construct subspace
        self.OneDM_loc, mo_coeff = self.ints.build_1pdm_loc()
        #tools.MatPrint(self.OneDM_loc,"self.OneDM_loc")

        '''
        ##########################
        # SCF with local orbitals
        ##########################
        ops_loc = libgen.build_locops(self.mol_frag, self.mol_env, self.ints, 0.0, self.Kcoeff, 0)
        oei_loc = ops_loc["locKin"]+ops_loc["locVnuc1"]+ops_loc["locVnuc2"]
        tei_loc = ops_loc["locTEI"]

        mf = qcwrap.qc_scf(True, mol=self.mol, Ne=self.ints.Nelec, Norb=self.ints.NOrb, method=self.mf_method, \
                           oei=oei_loc, tei=tei_loc, dm0=self.OneDM_loc, coredm=0.0,\
                           ao2sub=self.ints.ao2loc, smear_sigma = self.smear_sigma)
        #mf.init_guess = 'minao'
        mf.kernel()
        energy = mf.elec_energy + self.ints.energy_nuc()
        print('total scf energy = %.15g ' % energy)
        ##########################
        '''

        mo_coeff_loc,mo_occ_loc,_ = self.ints.get_loc_mo(self.smear_sigma)

        self.dim_imp, self.dim_bath, self.Occupations, self.loc2sub, occ_imp, occ_bath = \
        subspac.construct_subspace2(mo_coeff_loc,mo_occ_loc, self.mol_frag, self.ints, self.cluster,dim_imp,dim_bath,self.sub_threshold,\
                                    occ_tol = frac_occ_tol)


        self.P_frag_loc = None
        self.P_env_loc = None
        if(self.P_frag_ao is not None):
            s = self.mf_full.get_ovlp(self.mol)
            self.P_frag_loc = tools.dm_ao2loc(self.P_frag_ao, s, self.ints.ao2loc)
            self.P_env_loc = tools.dm_ao2loc(self.P_env_ao, s, self.ints.ao2loc)
        else:
            #self.P_frag_loc, self.P_env_loc, mo_frag_loc, mo_occ_loc  = subspac.loc_fullP_to_fragP(self.cluster,mo_coeff,self.ints.Nelec//2, self.ints.NOrb)
            self.P_frag_loc, self.P_env_loc = subspac.mulliken_partition_loc(self.cluster, self.OneDM_loc)


        #self.dim_imp, self.dim_bath, self.Occupations, self.loc2sub, occ_imp, occ_bath = \
        #    subspac.construct_subspace(self.ints,self.mol_frag,self.mol_env,self.OneDM_loc, self.cluster, self.sub_threshold, dim_bath, dim_imp)
        


        #test boys
        #self.loc2sub = np.eye(10)
        #self.dim_imp=1
        #self.dim_bath=2
        #self.Occupations=np.array([0.,0.,0.,2.,2.,0.,0.,0.,0.,0.])
        #self.Occupations[:]=0.0

        #tools.MatPrint(self.loc2sub,'self.loc2sub')
        #self.ints.sub_molden( self.loc2sub, 'loc2sub.molden', mo_occ=None )
        #exit()

        self.dim_sub = self.dim_imp + self.dim_bath
        print ('dimension of subspace: imp, bath',) 
        print (self.dim_imp, self.dim_bath)

        self.dim_big = dim_big
        if dim_big is None: self.dim_big =  self.dim_frag + self.dim_bath
        #construct core determinant
        idx = self.dim_frag + self.dim_bath
        self.core1PDM_loc, self.Nelec_core, Norb_imp_throw, self.frag_core1PDM_loc = subspac.build_core(self.Occupations, self.loc2sub, idx)
        self.core1PDM_ao = tools.dm_loc2ao(self.core1PDM_loc, self.ints.ao2loc) 

        if(use_bath_virt):
            #determine boundary orbitals
            ##################
            nbas = self.mol.nao_nr()
            natoms = self.mol.natm
            aoslice = self.mol.aoslice_by_atom()
            impurities = np.zeros([nbas], dtype = int)
            for i in range(natoms):
                if(abs(boundary_atoms[i]) >= 0.01):
                    impurities[aoslice[i,2]:aoslice[i,3]] = 1

            isbound = impurities ==1
            self.is_bound_orb = np.zeros([nbas], dtype = int)
            dim_active = self.dim_frag + self.dim_bath
            n_bound_orb = 0
            for i in range(dim_active + self.Nelec_core//2, nbas):
                weight = np.linalg.norm(self.loc2sub[isbound,i])
                if(weight > 0.5): 
                    self.is_bound_orb[i] = 1
                    n_bound_orb += 1

            self.is_bound_orb[:dim_active] = -1
            self.is_bound_orb[dim_active:dim_active+self.Nelec_core//2] =2
            _loc2sub = self.loc2sub.copy()
            self.loc2sub[:,dim_active:dim_active+n_bound_orb] = _loc2sub[:,self.is_bound_orb==1]
            self.loc2sub[:,dim_active+n_bound_orb:dim_active+n_bound_orb+self.Nelec_core//2] = _loc2sub[:,self.is_bound_orb==2]
            self.loc2sub[:,dim_active+n_bound_orb+self.Nelec_core//2:] = _loc2sub[:,self.is_bound_orb==0]

            self.dim_big += n_bound_orb
            print ("dim_big = ", self.dim_big)
            #################

        #debug
        self.dim_big = self.dim_sub

        self.ao2sub = np.dot(self.ints.ao2loc, self.loc2sub)
        #tools.MatPrint(self.ao2sub,'ao2sub')

        self.P_env_loc -= self.core1PDM_loc  #assume core does not have imp contribution

        self.Ne_frag = self.Ne_frag - Norb_imp_throw*2
        self.Ne_env = self.ints.Nelec - self.Ne_frag - self.Nelec_core
        
        print ('Ne_frag, Ne_env, Ne_core')
        print (self.Ne_frag, self.Ne_env, self.Nelec_core)

        self.umat = umat
        self.P_imp = None
        self.P_bath = None

        dim = self.dim_sub
        loc2sub = self.loc2sub
        self.P_ref_sub = tools.dm_loc2sub(self.OneDM_loc - self.core1PDM_loc, loc2sub[:,:dim])
        #tools.MatPrint(self.P_ref_sub, "P_ref")


        self.P_imp = np.dot(np.dot(self.loc2sub[:,:dim].T,self.P_frag_loc),self.loc2sub[:,:dim])
        self.P_bath = np.dot(np.dot(self.loc2sub[:,:dim].T,self.P_env_loc),self.loc2sub[:,:dim])
        #print ('|diffP| = ',  np.linalg.norm(self.P_imp + self.P_bath - self.P_ref_sub))
        #print ('P_imp idem:', np.linalg.norm(np.dot(self.P_imp,self.P_imp) - 2.0*self.P_imp))
        #print ('P_bath idem:',np.linalg.norm(np.dot(self.P_bath,self.P_bath) - 2.0*self.P_bath))


        #if(self.umat is not None):
        #    self.umat = tools.op_ao2sub(self.umat, self.ao2sub[:,:dim])
        if self.umat is None:
            #self.umat = np.random.rand(dim,dim)
            #self.umat = 0.5*(self.umat+self.umat.T)
            #self.umat = self.umat - np.eye( self.umat.shape[ 0 ] ) * np.average( np.diag( self.umat ) )
            self.umat = np.zeros((dim,dim))
        print ('|umat| = ', np.linalg.norm(self.umat))

        '''
        #density partition
        self.P_imp,P2 = subspac.subocc_to_dens_part(self.P_ref_sub,occ_imp, occ_bath, self.dim_imp, self.dim_bath)

        self.P_bath = self.P_ref_sub - self.P_imp
        print 'check idempotency for P_imp, P_bath'
        print np.linalg.norm(np.dot(self.P_imp,self.P_imp)-2.0*self.P_imp)
        print np.linalg.norm(np.dot(self.P_bath,self.P_bath)-2.0*self.P_bath)
        '''

        '''
        frag_occ = np.zeros([dim],dtype = float)
        for i in range(dim):
            frag_occ[i] = self.P_imp[i,i]
        self.ints.sub_molden( self.loc2sub[:,:dim], 'frag_dens_guess.molden', frag_occ )

        env_occ = np.zeros([dim],dtype = float)
        for i in range(dim):
            env_occ[i] = self.P_bath[i,i]
        self.ints.sub_molden( self.loc2sub[:,:dim], 'env_dens_guess.molden', env_occ )
        '''

        self.ints.sub_molden( self.loc2sub, "ao2sub.molden")

        self.oep_params = oep_params

        self.ops = None

        #test use
        self.P_ref = None
        self.vnuc_bound_frag = None
        self.vnuc_bound_env  = None
        if(self.use_umat_ao):
            self.P_ref = self.P_ref_ao
            self.vnuc_bound_frag = 0.0
            self.vnuc_bound_env  = 0.0
            if(self.umat is None):
                self.umat = np.zeros((dim,dim))
            s = self.mf_full.get_ovlp(self.mol)
            ao2sub = self.ao2sub[:,:self.dim_sub]
            self.P_imp = tools.dm_sub2ao(self.P_imp, ao2sub)
            self.P_bath = tools.dm_sub2ao(self.P_bath, ao2sub)
            return None
        #end test

        self.ops = libgen.build_subops(self.impAtom,self.mol_frag,self.mol_env,\
                                       self.boundary_atoms,self.boundary_atoms2,\
                                       self.ints, self.loc2sub, self.core1PDM_loc, self.dim_sub, self.Kcoeff, self.Nelec_core)

        self.total_scf_energy()
        self.P_ref = self.P_ref_sub
        self.P_imp, self.P_bath = oep.init_dens_par(self, self.dim_sub, True)

    def calc_umat(self):
      
        dim = self.dim_sub
        if(self.ops is None):
            self.ops = libgen.build_subops(self.impAtom, self.mol_frag,self.mol_env,\
                                           self.boundary_atoms,self.boundary_atoms2,\
                                           self.ints, self.loc2sub, self.core1PDM_loc, self.dim_sub, self.Kcoeff, self.Nelec_core) 
        myoep = oep.OEP(self, self.oep_params)

        #myoep.params.gtol = myoep.params.gtol * 100.0
        #myoep.params.l2_lambda = myoep.params.gtol * 1.0 #test L2 regularization 
        myoep.kernel()
        self.umat = myoep.umat
        self.P_imp = myoep.P_imp
        self.P_bath = myoep.P_bath
        self.frag_mo = myoep.frag_mo
        self.env_mo = myoep.env_mo

        #P_imp_loc = tools.dm_sub2loc(self.P_imp,self.loc2sub[:,:dim])
        #P_bath_loc = tools.dm_sub2loc(self.P_bath,self.loc2sub[:,:dim])
        #tools.MatPrint(P_imp_loc,"P_imp_loc")
        #tools.MatPrint(P_bath_loc+self.core1PDM_loc,"P_bath_loc+P_core_loc")
        #tools.MatPrint(P_imp_loc+P_bath_loc+self.core1PDM_loc-self.OneDM_loc,"P_sum_loc-P_tot_loc")

        #tools.MatPrint(self.P_imp,"P_imp")
        #tools.MatPrint(self.P_bath,"P_bath")
        #tools.MatPrint(self.P_imp+self.P_bath,"P_imp+P_bath")
        #tools.MatPrint(self.umat,"umat")

        #P1 = self.P_imp.copy()
        #u1 = self.umat.copy()
        '''
        myoep.params.algorithm = 'split'
        myoep.params.l2_lambda = 0.0
        myoep.kernel()
        self.umat = myoep.umat
        self.P_imp = myoep.P_imp
        self.P_bath = myoep.P_bath
        #tools.MatPrint(self.P_imp,"P_imp")
        #tools.MatPrint(self.P_bath,"P_bath")
        #tools.MatPrint(self.P_imp+self.P_bath,"P_imp+P_bath")
        #tools.MatPrint(self.umat,"umat")
        '''
        #P2 = self.P_imp.copy()
        #u2 = self.umat.copy()

        
        #print np.linalg.norm(P1-P2)
        #tools.MatPrint(P1-P2,"P_imp_2011 - P_imp_split")
        #tools.MatPrint(u1-u2,"umat_2011 - umat_split")
        '''
        ops = self.ops
        JK_imp = self.ints.impJK_sub(self.P_imp, ops["subTEI"],self.Kcoeff)
        JK_bath = self.ints.impJK_sub(self.P_bath, ops["subTEI"],self.Kcoeff)

        ao2sub = self.ao2sub[:,:self.dim_sub]
        mf1 = qcwrap.qc_scf(self.Ne_frag,self.dim_sub,self.mf_method,mol=self.mol_frag,oei=None,tei=None,dm0=self.P_imp,coredm=0.0,ao2sub=ao2sub)
        vxc_imp_ao = qcwrap.pyscf_rks.get_vxc(mf1, self.mol_frag, tools.dm_sub2ao(self.P_imp, ao2sub))[2]
        vxc_imp = tools.op_ao2sub(vxc_imp_ao, ao2sub)

        env_dm = tools.dm_sub2ao(self.P_bath, ao2sub)
        mf2 = qcwrap.qc_scf(self.Ne_env,self.dim_sub,self.mf_method,mol=self.mol_env,oei=None,tei=None,dm0=self.P_bath,coredm=0.0, ao2sub=ao2sub)
        vxc_bath_ao = qcwrap.pyscf_rks.get_vxc(mf2, self.mol_env, env_dm)[2]
        vxc_bath = tools.op_ao2sub(vxc_bath_ao, ao2sub)

        dm = tools.dm_sub2ao(self.P_imp+self.P_bath, ao2sub)
        mf = qcwrap.qc_scf(self.Ne_frag+self.Ne_env,self.dim_sub,self.mf_method,\
                           mol=self.mol,oei=None,tei=None,dm0=self.P_imp+self.P_bath,coredm=0.0, ao2sub=ao2sub)
        vxc_ao = qcwrap.pyscf_rks.get_vxc(mf, self.mol, dm)[2]
        vxc = tools.op_ao2sub(vxc_ao, ao2sub)

        #Vint_A = ops["subVnuc2"] + JK_bath - vxc_imp + vxc + 1e5*self.P_bath
        #Vint_B = ops["subVnuc1"] + JK_imp - vxc_bath + vxc + 1e5*self.P_imp

        #tools.MatPrint(Vint_A,"Vint_A")
        #tools.MatPrint(Vint_B,"Vint_B")
        '''

    def embedding_potential(self):

        #if(self.umat is None):
        self.calc_umat()

        #dim_big = self.dim_frag + self.dim_bath
        dim_big = self.dim_big
        '''
        occ_mo_imp = self.frag_mo[:,:self.Ne_frag//2]
        occ_mo_bath = self.env_mo[:,:self.Ne_env//2]
        self.umat = self.pad_umat(occ_mo_imp, occ_mo_bath, self.dim_sub, dim_big, self.core1PDM_loc)
        '''
        #tools.MatPrint(self.P_imp,'self.P_imp')
        #tools.MatPrint(self.P_bath,'self.P_bath')
        #tools.MatPrint(self.P_ref_sub,'P_ref')

        ao2sub = self.ao2sub[:,:dim_big]
        P_imp_ao = tools.dm_sub2ao(self.P_imp, ao2sub)
        P_bath_ao = tools.dm_sub2ao(self.P_bath, ao2sub)

        #tools.MatPrint(P_imp_ao,'P_frag')

        umat_ao = reduce(np.dot, (ao2sub, self.umat, ao2sub.T))

        diffP = P_imp_ao + P_bath_ao + self.core1PDM_ao - self.P_ref_ao
        print ('|diffP_ao| = ', np.linalg.norm(diffP), 'max(diffP_ao) = ', np.max(diffP))

        if(self.plot_dens):
            cubegen.density(self.mol, "tot_dens.cube", self.P_ref_ao, nx=100, ny=100, nz=100)
            cubegen.density(self.mol, "frag_dens.cube", P_imp_ao, nx=100, ny=100, nz=100)
            cubegen.density(self.mol, "bath_dens.cube", P_bath_ao, nx=100, ny=100, nz=100)
            cubegen.density(self.mol, "core_dens.cube", self.core1PDM_ao, nx=100, ny=100, nz=100)
            cubegen.density(self.mol, "env_dens.cube", P_bath_ao + self.core1PDM_ao, nx=100, ny=100, nz=100)
            cubegen.density(self.mol, "vemb.cube", umat_ao, nx=100, ny=100, nz=100)
            cubegen.density(self.mol, "error_dens.cube", diffP, nx=100, ny=100, nz=100)

        
        #return self.umat
        #s = self.mf_full.get_ovlp(self.mol)
        #return reduce(np.dot, (s,umat_plot,s))


    def correction_energy(self):

        energy = 0.0

        #if(self.umat is None):
        #    self.calc_umat()

        print ("Performing ECW energy calculation")
        dim = self.dim_big
        energy = self.ecw_energy(self.ecw_method, dim)

        '''
        if(self.ecw_method == 'hf'):
            energy = self.hf_energy(umat,dim)
        elif(self.ecw_method == 'mp2'):
            energy = self.mp2_energy(dim)
        elif(self.ecw_method == 'ccsd' or self.ecw_method == 'ccsd(t)'):
            energy = self.ccsd_energy(dim)
        else:
            raise Exception("ecw_method not supported!")

        if(self.deproton is not None):
            print 'deprotonated structure energy:'
            energy = self.ccsd_energy_beta(dim)
        '''

        return energy


    def ecw_energy(self, method, dim):

        mf_energy, Vemb, Vxc = self.imp_mf_energy(dim)

        Ne_frag = self.Ne_frag
        ops = self.ops

        subOEI = ops["subKin"] + ops["subVnuc1"] + Vemb
        subTEI = ops["subTEI"]

        mf = qcwrap.qc_scf(Ne_frag, dim, 'hf', mol=self.mol_frag, oei=subOEI, tei=subTEI, dm0=self.P_imp)
        mf.runscf()
        e_hf = mf.elec_energy# - np.trace(np.dot(mf.rdm1,Vemb))
        if(self.plot_mo):
            self.ints.submo_molden(mf.mo_coeff, mf.mo_occ, self.loc2sub, "mo_frag.molden",self.mol_frag)

        exc = 0.0
        if(Vxc is not None):
#           exc = np.einsum('ij,ji', Vxc, mf.rdm1-self.P_imp)
            exc = np.einsum('ij,ji', Vxc, self.P_imp)
        print ('exc = ', exc)

        if(method == 'hf'):
            energy = e_hf
        elif(method == 'mp2'):
            mp2 = mp.MP2(mf)
            mp2.kernel()
            energy = e_hf + mp2.e_corr
        elif(method == 'ccsd' or method == 'ccsd(t)'):
            mycc = cc.CCSD(mf)
            mycc.max_cycle = 200
            #mycc.conv_tol = 1e-6
            #mycc.conv_tol_normt = 1e-4
            mycc.kernel()

            et = 0.0
            if(method == 'ccsd(t)'):
                print ('CCSD(T) correction')
                et = mycc.ccsd_t()

            energy = e_hf + mycc.e_corr + et
        elif(method == 'eomccsd'):
            mf.verbose = 5
            mycc = cc.RCCSD(mf)
            mycc.kernel()
            es, vs = mycc.eomee_ccsd_singlet(nroots=self.ex_nroots)

            if (self.ex_nroots == 1):
                r1, r2 = mycc.vector_to_amplitudes(vs)
                print ('debug: ',r1.shape)
            else:
                for vn in vs:
                    r1, r2 = mycc.vector_to_amplitudes(vn)


            energy = e_hf + mycc.e_corr 
        elif(method == 'eomsfccsd'):
            self.mol_frag.spin = 2
            mf = qcwrap.pyscf_rks.rohf_pyscf(Ne_frag, dim, mol=self.mol_frag, oei=subOEI, tei=subTEI, dm0=np.array((.5*self.P_imp,.5*self.P_imp)) )
            mf.kernel(dm0 = mf.dm_guess)

            mf1 = qcwrap.pyscf_rks.uhf_pyscf(Ne_frag, dim, mol=self.mol_frag, oei=subOEI, tei=subTEI, dm0=None)
            mf1.convert_from_rhf(mf)

            mycc = cc.UCCSD(mf1)
            mycc.verbose = 5
            mycc.kernel()
            es,vs = mycc.eomsf_ccsd(nroots=self.ex_nroots)

            if (self.ex_nroots == 1):
                r1, r2 = cc.eom_uccsd.vector_to_amplitudes_eomsf(vs,mycc.nmo,mycc.nocc)
                tools.print_eomcc_t1(r1[0])
                tools.print_eomcc_t1(r1[1])
            else:
                for vn in vs:
                    r1, r2 = cc.eom_uccsd.vector_to_amplitudes_eomsf(vn,mycc.nmo,mycc.nocc)
                    tools.print_eomcc_t1(r1[0])
                    tools.print_eomcc_t1(r1[1])


            energy = mf1.e_tot + mycc.e_corr
        else:
            raise Exception("ecw_method not supported!")

        energy -= mf_energy
        energy -= exc

        return energy


    def core_energy(self):

        core1PDM_loc = self.core1PDM_loc
        oei_loc = self.ints.hcore_loc()
        coreJK_loc = self.ints.coreJK_loc(core1PDM_loc, self.Kcoeff)

        core_energy = np.trace(np.dot(core1PDM_loc,oei_loc)) + 0.5*np.trace(np.dot(core1PDM_loc,coreJK_loc))
        return core_energy


    def total_scf_energy(self):

        energy = 0.0

        Norb = self.dim_sub
        Ne = self.Ne_frag + self.Ne_env
        ops = self.ops

        subOEI = ops["subKin"]+ops["subVnuc1"]+ops["subVnuc2"]+ops["subCoreJK"]
        subTEI = ops["subTEI"]

        coredm = self.core1PDM_ao
        ao2sub = self.ao2sub[:,:Norb]
        
        #dm_guess = tools.fock2onedm(subOEI, Ne//2)[0]
        dm_guess = self.P_ref_sub
        mf = qcwrap.qc_scf(True, mol=self.mol, Ne=Ne, Norb=Norb, method=self.mf_method, \
                           oei=subOEI, tei=subTEI, dm0=dm_guess, coredm=coredm, ao2sub=ao2sub, smear_sigma = self.smear_sigma)
        #mf.init_guess =  'minao'
        mf.kernel()

        print ('max(P_tot - P_ref) = ', np.amax(np.absolute(mf.rdm1 - self.P_ref_sub)) )
        print ('|P_tot - P_ref| = ',    np.linalg.norm(mf.rdm1 - self.P_ref_sub))

        self.ints.submo_molden(mf.mo_coeff, mf.mo_occ, self.loc2sub, "total_system_mo.molden",self.mol)

        energy = mf.elec_energy + self.core_energy() + self.ints.energy_nuc()

        print('total scf energy = %.15g ' % energy)
        print('%.15g, %.15g, %.15g' % (mf.elec_energy, self.core_energy(), self.ints.energy_nuc()) )
        return energy


    def imp_mf_energy2(self, dim):

        Ne_frag = self.Ne_frag
        ops = self.ops
        umat = self.umat

        proj = 0.5*self.P_bath
        energy_shift = 1e5
        Vemb = umat + energy_shift*proj
        Vxc = None

        subOEI = ops["subKin"]+ops["subVnuc1"]+ops["subVnuc_bound1"]+Vemb
        subTEI = ops["subTEI"]

        ao2sub = self.ao2sub[:,:dim]

        mf = qcwrap.qc_scf(Ne_frag, dim, self.mf_method, mol=self.mol_frag, oei=subOEI, tei=subTEI, dm0=self.P_imp, coredm=0.0, ao2sub=ao2sub)
        mf.runscf()
        energy = mf.elec_energy

        print ('|diffP| = ', np.linalg.norm(mf.rdm1 + self.P_bath - self.P_ref_sub))
        print ("embeded imp scf (electron) energy = ",energy)

        self.P_imp = mf.rdm1

        return (energy, Vemb, Vxc)


    def imp_mf_energy_dfet(self, dim):

        Ne_frag = self.Ne_frag
        ops = self.ops
        Vemb = self.umat

        Vxc = None

        subOEI = ops["subKin"]+ops["subVnuc1"]+ops["subVnuc_bound1"]+Vemb
        subTEI = ops["subTEI"]

        ao2sub = self.ao2sub[:,:dim]

        mf = qcwrap.qc_scf(Ne_frag, dim, self.mf_method, mol=self.mol_frag, oei=subOEI, tei=subTEI, dm0=self.P_imp, coredm=0.0, ao2sub=ao2sub)
        #mf.init_guess =  'minao'
        mf.runscf()
        energy = mf.elec_energy #- np.trace(np.dot(mf.rdm1,Vemb))

        print ('|diffP| = ', np.linalg.norm(mf.rdm1 + self.P_bath - self.P_ref_sub))
        print ("embeded imp scf (electron) energy = ",energy)

        self.P_imp = mf.rdm1

        return (energy, Vemb, Vxc)



    def imp_mf_energy(self, dim):

        Ne_frag = self.Ne_frag
        ops = self.ops
        umat = self.umat

        Ne_env = self.Ne_env
        P_bath_loc = tools.dm_sub2ao(self.P_bath, self.loc2sub[:,:dim])
        P_bath_JK = libgen.coreJK_sub( self.ints, self.loc2sub, dim, P_bath_loc, Ne_env, self.Kcoeff)
        
        proj = 0.5*self.P_bath
        energy_shift = 1e5


        ao2sub = self.ao2sub[:,:dim]
        coredm = self.core1PDM_ao + tools.dm_sub2ao(self.P_bath, ao2sub)
        
        Vemb = ops["subVnuc2"] + P_bath_JK+ops["subCoreJK"] + energy_shift*proj
        Vxc = None
        if(self.mf_method != 'hf'):
            P_imp_ao = tools.dm_sub2ao(self.P_imp, ao2sub)
            mf_frag = qcwrap.qc_scf(Ne_frag,dim,self.mf_method,mol=self.mol_frag,oei=None,tei=None,dm0=self.P_imp,coredm=0.0,ao2sub=ao2sub)
            vxc_imp_ao = qcwrap.pyscf_rks.get_vxc(mf_frag, self.mol_frag, P_imp_ao)[2]
            vxc_full_ao = qcwrap.pyscf_rks.get_vxc(self.mf_full, self.mol, coredm + P_imp_ao)[2]

            Vxc = tools.op_ao2sub(vxc_full_ao, ao2sub) - tools.op_ao2sub(vxc_imp_ao, ao2sub)
            Vemb += Vxc


        subOEI = ops["subKin"]+ops["subVnuc1"]+Vemb
        #subOEI = ops["subKin"]+ops["subVnuc1"]+ops["subVnuc2"]+P_bath_JK+ops["subCoreJK"]+ energy_shift*proj
        #subOEI = ops["subKin"]+ops["subVnuc1"]+ops["subVnuc_bound1"]+umat +0.5*1e4*self.P_bath
        subTEI = ops["subTEI"]

        mf = qcwrap.qc_scf(Ne_frag, dim, self.mf_method, mol=self.mol_frag, oei=subOEI, tei=subTEI, dm0=self.P_imp, coredm=0.0, ao2sub=ao2sub,\
                           smear_sigma = self.smear_sigma)
        mf.runscf()
        energy = mf.elec_energy

        #self.ints.submo_molden(mf.mo_coeff, mf.mo_occ, self.loc2sub, "mo_frag.molden" )

        print ('|diffP| = ', np.linalg.norm(mf.rdm1 + self.P_bath - self.P_ref_sub))
        print ("embeded imp scf (electron) energy = ",energy)

        self.P_imp = mf.rdm1

        print ('level shift energy contribution: ',np.trace(np.dot(energy_shift*proj,self.P_imp)))

        return (energy, Vemb, Vxc)


    def imp_mf_energy_beta(self, dim):

        mol = self.mol_frag
        natm = mol.natm

        mol1 = mol.copy()
        for i in range(natm):
            if(self.deproton[i] == 1):
                mol1._atm[i][0] = 0 #make proton ghost

        vnuc_sub = libgen.frag_vnuc_sub(mol1, self.ints, self.loc2sub, dim)


        Ne_frag = self.Ne_frag
        ops = self.ops
        umat = self.umat

        subOEI = ops["subKin"]+vnuc_sub+ops["subVnuc_bound1"]+umat
        subTEI = ops["subTEI"]

        ao2sub = self.ao2sub[:,:dim]
        mf = qcwrap.qc_scf(Ne_frag,dim,self.mf_method,mol=self.mol,oei=subOEI,tei=subTEI,dm0=self.P_imp,coredm=0.0,ao2sub=ao2sub)
        mf.runscf()
        energy = mf.elec_energy

        e_emb = 0.0
        #onedm = mf.make_rdm1()
        #e_emb = np.trace(np.dot(umat,onedm))

        energy -= e_emb

        print ("embeded imp scf (electron) energy = ",energy)
        return energy


    def hf_energy(self, umat, dim):

        print ("ECW method is HF") 
        energy = 0.0

        Ne_frag = self.Ne_frag_orig
        ops = self.ops

        subOEI = ops["subKin"]+ops["subVnuc1"]+ops["subVnuc_bound1"]+ops["subCoreJK"]+umat

        dim_sub = self.dim_sub
        npad = dim - dim_sub
        P_guess = np.pad(self.P_imp, ((0,npad),(0,npad)), mode='constant', constant_values=0.0)

        norb = (Ne_frag - self.Ne_frag)//2
        for i in range(norb):
            index = dim_sub+i
            P_guess[index][index] = 2.0

        energy, onedm, mo = qcwrap.pyscf_rhf.scf( subOEI, ops["subTEI"], dim, Ne_frag, P_guess)

        P = np.dot(np.dot(self.loc2sub[:,:dim_sub],self.P_imp),self.loc2sub[:,:dim_sub].T)
        P = np.dot(np.dot(self.loc2sub[:,:dim].T,P),self.loc2sub[:,:dim])
        print (np.linalg.norm(P_guess-P))
        print (np.linalg.norm(onedm-P))
        exit()

        tools.MatPrint(self.P_imp, "P_imp")
        tools.MatPrint(P,"P_imp+virt")
        print (energy)
        exit()
        imp_mf_energy = self.imp_mf_energy()

        energy = energy - imp_mf_energy

        print ("dmfet correction energy = ",energy)
        return energy


    def ks_ccsd_energy(self,dim):

        print ("ECW method is CCSD")
        energy = 0.0

        Ne_frag = self.Ne_frag
        ops = self.ops
        umat = self.umat

        subOEI = ops["subKin"]+ops["subVnuc1"]+ops["subVnuc_bound1"]+umat
        subTEI = ops["subTEI"]

        ao2sub = self.ao2sub[:,:dim]
        mf = qcwrap.qc_scf(Ne_frag,dim,self.mf_method,mol=self.mol,oei=subOEI,tei=subTEI,dm0=self.P_imp,coredm=0.0,ao2sub=ao2sub)
        mf.runscf()
        e_ks =  mf.elec_energy
        print ("e_ks = ", e_ks)

        #HF JK
        veff = self.ints.impJK_sub( mf.make_rdm1(), subTEI)
        mf.get_veff = lambda *args: veff

        print ("incore_anyway: ",mf.mol.incore_anyway)

        e_emb = 0.0
        #onedm = mf.make_rdm1() 
        #e_emb=np.trace(np.dot(umat,onedm))

        #print "diffP = ",np.linalg.norm(self.P_imp - onedm)

        e_hf =  pyscf.scf.hf.energy_elec(mf,mf.make_rdm1(),subOEI,veff)[0]
        print ("e_hf = ", e_hf)

        mycc = cc.CCSD(mf).run()
        et = 0.0
        if(self.ecw_method == 'ccsd(t)'):
            print ('CCSD(T) correction')
            et = mycc.ccsd_t()

        #ccsd_dm_mo = mycc.make_rdm1()
        #ccsd_dm_sub = tools.dm_loc2ao(ccsd_dm_mo,mf.mo_coeff) 
        #e_emb=np.trace(np.dot(umat,ccsd_dm_sub))

        print (mycc.e_corr + et)
        e_ccsd = e_hf + mycc.e_corr + et
        e_ccsd -= e_emb
        print (e_ccsd)

        energy = e_ccsd - e_ks

        print ("dmfet correction energy = ",energy)
        return energy


    def mp2_energy(self,dim):

        print ("ECW method is MP2")

        energy = 0.0

        Ne_frag = self.Ne_frag
        ops = self.ops
        umat = self.umat

        Ne_env = self.Ne_env
        P_bath_JK = libgen.coreJK_sub( self.ints, self.loc2sub, dim, tools.dm_sub2ao(self.P_bath, self.loc2sub[:,:dim]), Ne_env, 1.0)

        subOEI = ops["subKin"]+ops["subVnuc1"]+ops["subVnuc2"]+P_bath_JK+0.5*1e4*self.P_bath
        #subOEI = ops["subKin"]+ops["subVnuc1"]+ops["subVnuc_bound1"]+umat
        subTEI = ops["subTEI"]

        mf = qcwrap.qc_scf(Ne_frag, dim, 'hf', mol=self.mol, oei=subOEI, tei=subTEI, dm0=self.P_imp)
        mf.runscf()

        #self.ints.submo_molden( mf.mo_coeff, mf.mo_occ, self.loc2sub, 'mo_imp.molden' )
        print ('|diffP| = ', np.linalg.norm(mf.rdm1 + self.P_bath - self.P_ref_sub))

        print ("mo_energy")
        print (mf.mo_energy)

        e_emb = 0.0
        #onedm = mf.make_rdm1() 
        #e_emb=np.trace(np.dot(umat,onedm))
        #print "diffP = ",np.linalg.norm(self.P_imp - onedm)

        mp2 = mp.MP2(mf)
        mp2.kernel()

        e_mp2 = mf.elec_energy + mp2.e_corr
        e_mp2 -= e_emb
        
        imp_mf_energy = self.imp_mf_energy(dim)

        energy = e_mp2 - imp_mf_energy


        print ("dmfet correction energy = ",energy)
        return energy




    def ccsd_energy(self,dim):

        print ("ECW method is CCSD")
        energy = 0.0

        Ne_frag = self.Ne_frag
        ops = self.ops
        umat = self.umat

        Ne_env = self.Ne_env
        P_bath_JK = libgen.coreJK_sub( self.ints, self.loc2sub, dim, tools.dm_sub2ao(self.P_bath, self.loc2sub[:,:dim]), Ne_env, 1.0)

        subOEI = ops["subKin"]+ops["subVnuc1"]+ops["subVnuc2"]+P_bath_JK+ops["subCoreJK"]+0.5*1e4*self.P_bath
        #subOEI = ops["subKin"]+ops["subVnuc1"]+ops["subVnuc_bound1"]+umat + 0.5*1e4*self.P_bath
        subTEI = ops["subTEI"]

        mf = qcwrap.qc_scf(Ne_frag, dim, 'hf', mol=self.mol, oei=subOEI, tei=subTEI, dm0=self.P_imp)
        mf.runscf()

        e_emb = 0.0
        #onedm = mf.make_rdm1() 
        #e_emb=np.trace(np.dot(umat,onedm))

        #print "diffP = ",np.linalg.norm(self.P_imp - onedm)

        print (mf.elec_energy)
        mycc = cc.CCSD(mf)
        mycc.max_cycle = 200
        #mycc.conv_tol = 1e-6
        #mycc.conv_tol_normt = 1e-4
        mycc.kernel()

        et = 0.0
        if(self.ecw_method == 'ccsd(t)'):
            print ('CCSD(T) correction')
            et = mycc.ccsd_t()

        #ccsd_dm_mo = mycc.make_rdm1()
        #ccsd_dm_sub = tools.dm_loc2ao(ccsd_dm_mo,mf.mo_coeff) 
        #e_emb=np.trace(np.dot(umat,ccsd_dm_sub))

        e_hf = mf.elec_energy
        print (mycc.e_corr + et)

        e_ccsd = e_hf + mycc.e_corr + et
        e_ccsd -= e_emb
        print (e_ccsd)

        imp_mf_energy = self.imp_mf_energy(dim)

        energy = e_ccsd - imp_mf_energy


        print ("dmfet correction energy = ",energy)
        return energy

    def ccsd_energy_beta(self, dim):

        mol = self.mol_frag
        natm = mol.natm

        mol1 = mol.copy()
        for i in range(natm):
            if(self.deproton[i] == 1):
                mol1._atm[i][0] = 0 #make proton ghost
        print ('mol_frag de:')
        print (mol1._atm)
        vnuc_sub = libgen.frag_vnuc_sub(mol1, self.ints, self.loc2sub, dim)

        print ("ECW method is CCSD")
        energy = 0.0

        Ne_frag = self.Ne_frag
        ops = self.ops
        umat = self.umat

        subOEI = ops["subKin"]+vnuc_sub+ops["subVnuc_bound1"]+umat
        subTEI = ops["subTEI"]

        mf = qcwrap.qc_scf(Ne_frag,dim,'hf',mol=None,oei=subOEI,tei=subTEI,dm0=self.P_imp)
        mf.runscf()

        e_emb = 0.0
        #onedm = mf.make_rdm1()
        #e_emb=np.trace(np.dot(umat,onedm))

        #print "diffP = ",np.linalg.norm(self.P_imp - onedm)

        print (mf.elec_energy)
        mycc = cc.CCSD(mf).run()
        et = 0.0
        if(self.ecw_method == 'ccsd(t)'):
            print ('CCSD(T) correction')
            et = mycc.ccsd_t()

        e_hf = mf.elec_energy
        print (mycc.e_corr + et)

        e_ccsd = e_hf + mycc.e_corr + et
        e_ccsd -= e_emb
        print (e_ccsd)

        imp_mf_energy = self.imp_mf_energy_beta(dim)

        energy = e_ccsd - imp_mf_energy


        print ("dmfet correction energy = ",energy)
        return energy


    def tot_ccsd_energy(self):

        energy = 0.0

        dim = self.dim_frag+self.dim_bath
        Ne = self.Ne_frag+self.Ne_env
        ops = libgen.build_subops(self.impAtom,self.mol_frag,self.mol_env, self.boundary_atoms, self.boundary_atoms2, self.ints, self.loc2sub, self.core1PDM_loc, self.dim_sub)

        subOEI = ops["subKin"]+ops["subVnuc1"]+ops["subVnuc2"]+ops["subCoreJK"]

        mf = qcwrap.pyscf_rhf.rhf( subOEI, ops["subTEI"], dim, Ne)

        e_hf = mf.e_tot + self.core_energy() + self.ints.const()
        print ("hf energy = ", e_hf)

        mycc = cc.CCSD(mf).run()
        et = 0.0
        et = mycc.ccsd_t()

        print ("correlation energy = ", mycc.e_corr + et)

        e_ccsd = e_hf + mycc.e_corr + et
        print ("total CCSD(T) energy = ", e_ccsd)
        return energy


    def pad_umat(self, occ_mo_imp, occ_mo_bath, dim_small, dim_big, core1dm_loc):

        if(dim_small == dim_big):
            return self.umat

        ops = libgen.build_subops(self.impAtom, self.mol_frag,self.mol_env, \
                                  self.boundary_atoms,self.boundary_atoms2, self.ints, self.loc2sub, core1dm_loc, dim_big,self.Kcoeff, self.Nelec_core)
        self.ops = ops

        P_imp_small = 2.0*np.dot(occ_mo_imp,occ_mo_imp.T)
        P_bath_small = 2.0*np.dot(occ_mo_bath,occ_mo_bath.T)

        dim_ext = dim_big - dim_small

        P_imp_big = np.pad(P_imp_small, ((0,dim_ext),(0,dim_ext)), 'constant', constant_values=0 )
        P_bath_big = np.pad(P_bath_small, ((0,dim_ext),(0,dim_ext)), 'constant', constant_values=0 )

        self.P_imp = P_imp_big
        self.P_bath = P_bath_big
        self.P_ref_sub = tools.dm_loc2sub(self.OneDM_loc - self.core1PDM_loc, self.loc2sub[:,:dim_big])

        JK_imp = self.ints.impJK_sub( P_imp_big, ops["subTEI"],self.Kcoeff)
        JK_bath = self.ints.impJK_sub( P_bath_big, ops["subTEI"],self.Kcoeff)


        ao2sub = self.ao2sub[:,:dim_big]
        coredm = self.core1PDM_ao
        if(self.mf_method != 'hf'):
            mf1 = qcwrap.qc_scf(True, mol=self.mol_frag, Ne=self.Ne_frag, Norb=dim_big, method=self.mf_method, \
                                oei=None,tei=None,dm0=P_imp_big,coredm=0.0,ao2sub=ao2sub)
            vxc_imp_ao = qcwrap.pyscf_rks.get_vxc(mf1, self.mol_frag, tools.dm_sub2ao(P_imp_big, ao2sub))[2]
            JK_imp += tools.op_ao2sub(vxc_imp_ao, ao2sub)

            env_dm = coredm+tools.dm_sub2ao(P_bath_big, ao2sub)
            mf2 = qcwrap.qc_scf(True, mol=self.mol_env, Ne=self.Ne_env, Norb=dim_big, method=self.mf_method, \
                                oei=None,tei=None,dm0=P_bath_big,coredm=coredm, ao2sub=ao2sub)
            vxc_bath_ao = qcwrap.pyscf_rks.get_vxc(mf2, self.mol_env, env_dm, n_core_elec=self.Nelec_core)[2]
            JK_bath += tools.op_ao2sub(vxc_bath_ao, ao2sub)


        oei_imp = ops["subKin"]+ops["subVnuc1"]+ops["subVnuc_bound1"] + JK_imp
        ext_oei_imp = oei_imp[dim_small:,:dim_small]

        oei_bath = ops["subKin"]+ops["subVnuc2"]+ops["subVnuc_bound2"]+ops["subCoreJK"] + JK_bath
        ext_oei_bath = oei_bath[dim_small:,:dim_small]

        b_imp = -np.dot(ext_oei_imp, occ_mo_imp)
        b_bath = -np.dot(ext_oei_bath, occ_mo_bath)


        #solve uv = b for u
        v = np.concatenate((occ_mo_imp, occ_mo_bath), axis=1)
        b = np.concatenate((b_imp, b_bath), axis=1)

        AAT = np.dot(v.T, v)
        #tools.MatPrint(AAT,'AAT')
        #tmp = np.dot(np.linalg.inv(AAT), b.T)
        tmp = b.T
        uT = np.dot(v,tmp)
        u = uT.T

        #u = scipy.linalg.lstsq(v.T, b.T, cond=None, overwrite_a=False, overwrite_b=False, check_finite=True, lapack_driver=None)[0]
        #u = np.linalg.lstsq(v.T, b.T, rcond=1e-9)[0]
        #tools.MatPrint(u.T,"u")
        #zero = np.dot(u.T,v)-b
        #tools.MatPrint(zero,"zero")

        #tools.MatPrint(self.umat,"umat")

        umat = np.pad(self.umat, ((0,dim_ext),(0,dim_ext)), 'constant', constant_values=0 )
        umat[dim_small:, :dim_small] = u
        umat[:dim_small, dim_small:] = u.T


        #umat1=umat.copy()
        #umat1[dim_small:, :dim_small] = -ext_oei_bath
        #umat1[:dim_small, dim_small:] = -ext_oei_bath.T
        
        #print "u-u1=",np.linalg.norm(umat-umat1)

        #debug
        subOEI = ops["subKin"]+ops["subVnuc1"]+ops["subVnuc_bound1"]
        #energy, onedm, mo = qcwrap.pyscf_rhf.scf( subOEI, ops["subTEI"], dim_big, self.Ne_frag, P_imp_big, self.mf_method)
        mf1 = qcwrap.qc_scf(True, mol=self.mol_frag, Ne=self.Ne_frag, Norb=dim_big, method=self.mf_method,\
                            vext_1e = umat, oei=subOEI,tei=ops["subTEI"],dm0=P_imp_big,coredm=0.0,ao2sub=ao2sub)
        mf1.runscf()

        subOEI = ops["subKin"]+ops["subVnuc2"]+ops["subVnuc_bound2"]+ops["subCoreJK"]
        #energy2, onedm2, mo2 = qcwrap.pyscf_rhf.scf( subOEI, ops["subTEI"], dim_big, self.Ne_env, P_bath_big, self.mf_method)
        mf2 = qcwrap.qc_scf(True, mol=self.mol_env, Ne=self.Ne_env, Norb=dim_big, method=self.mf_method,\
                            vext_1e = umat, oei=subOEI,tei=ops["subTEI"],dm0=P_bath_big,coredm=coredm,ao2sub=ao2sub)
        #mf2.conv_check = False #temp

        #subOEI = ops["subKin"]+ops["subVnuc2"]+ops["subVnuc_bound2"]+umat1
        #s = self.mf_full.get_ovlp(self.mol)
        #coredm_sub = tools.dm_ao2sub(coredm,s,ao2sub)
        #mf2 = qcwrap.qc_scf(self.Ne_env+self.Nelec_core,dim_big,self.mf_method,mol=self.mol_env,oei=subOEI,tei=ops["subTEI"],dm0=P_bath_big+coredm_sub,coredm=0.0,ao2sub=ao2sub)
        mf2.runscf()

        diffP = mf1.rdm1 + mf2.rdm1 - self.P_ref_sub
        #diffP = mf1.rdm1 + mf2.rdm1 - tools.dm_loc2sub(self.OneDM_loc, self.loc2sub[:,:dim_big])

        print (np.linalg.norm(P_imp_big-mf1.rdm1), np.linalg.norm(P_bath_big-mf2.rdm1))
        print ('|diffP| = ',np.linalg.norm(diffP), 'max(diffP) = ',np.amax(np.absolute(diffP)))
        #tools.MatPrint(diffP,"diffP")
        #tools.MatPrint(mf1.rdm1,"P1")
        #tools.MatPrint(mf2.rdm1,"P2")
        #tools.MatPrint(P_imp_small,"P_imp_small")
        #tools.MatPrint(P_bath_small,"P_bath_small")
        return umat
