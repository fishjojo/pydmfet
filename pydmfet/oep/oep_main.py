import os,sys
import ctypes
import numpy as np
from scipy import optimize
from pyscf import lib, scf, mp, lo
from pyscf.tools import cubegen
import time,copy
from pydmfet import qcwrap,tools,subspac, libgen
from pydmfet.libcpp import oep_hess, oep_hess_old, mkl_svd
from pydmfet.opt import newton
from pydmfet.oep import OEPparams, ObjFunc_WuYang
from pydmfet.oep.oep_optimize import OEP_Optimize

def init_umat_invks(oep):

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
    print ("|umat| = ", np.linalg.norm(umat))

    oei = subOEI + umat
    tei = subTEI

    FRAG_energy, FRAG_1RDM = qcwrap.pyscf_rhf.scf_oei( oei, dim, Ne)

    energy = FRAG_energy - np.trace(np.dot(P_ref,umat))
    diffP = FRAG_1RDM - P_ref

    grad = tools.mat2vec(diffP, dim)
    grad = -1.0 * grad

    return (-energy,grad)


def init_umat(oep, dim, umat_init_method = "zero"):

    if umat_init_method.upper() == "ZERO":
        zero_mat = np.zeros([dim,dim], dtype = float)
        if isinstance(oep.P_ref, np.ndarray):
            return zero_mat
        else:
            return [zero_mat,zero_mat]
    else:
        raise NotImplementedError("umat init method: %s, is not implemented" % umat_init_method)

class OEP:

    def __init__(self, embedobj, params):

        if not isinstance(params, OEPparams):
            raise TypeError("%s is not an instance of pydmfet.oep.OEPparams" % params)
        self.params   = params

        self.mol_full = getattr(embedobj, 'mol_full', None)
        self.mol_frag = getattr(embedobj, 'mol_frag', None)
        self.mol_env  = getattr(embedobj, 'mol_env', None)

        self.Ne_frag  = getattr(embedobj, 'Ne_frag', None)
        self.Ne_env   = getattr(embedobj, 'Ne_env', None)

        self.mf_full  = getattr(embedobj, 'mf_full', None)
        self.mf_frag  = None
        self.mf_env   = None

        self.P_ref    = getattr(embedobj, 'P_ref', None)
        if self.P_ref is None:
            raise RuntimeError("reference density not defined")

        self.dim      = 0
        if isinstance(self.P_ref, np.ndarray):
            self.dim  = self.P_ref.shape[0]
        else:
            self.dim  = self.P_ref[0].shape[0]

        self.P_imp    = getattr(embedobj, 'P_imp', None)
        self.P_bath   = getattr(embedobj, 'P_bath', None)
        if self.P_imp is None or self.P_bath is None:
            raise RuntimeError("initial density partition not defined")


        self.umat     = getattr(embedobj, 'umat', None)
        if self.umat is None:
            self.umat = init_umat(self, dim, params.umat_init_method)

        self.ao_bas_tab_frag = getattr(embedobj, 'ao_bas_tab_frag', None)

        #v2m function, can use symmetry if required
        self.v2m      = tools.vec2mat 
        self.sym_tab  = getattr(embedobj, 'sym_tab', None)

        #attributes associated with scf calculations
        self.use_suborb  = getattr(embedobj, 'use_suborb', False)
        self.mf_method   = getattr(embedobj, 'mf_method', 'lda,vwn')
        self.smear_sigma = getattr(embedobj, 'smear_sigma', 0.0)
        self.ops         = getattr(embedobj, 'ops', None)
        self.ao2sub      = getattr(embedobj, 'ao2sub', None)
        self.core1PDM_ao = getattr(embedobj, 'core1PDM_ao', None)
        self.ints        = getattr(embedobj, 'ints', None)

        self.tei      = None
        self.oei_frag = None
        self.oei_env  = None
        if self.use_suborb:
            #frequently used operators
            ops = self.ops
            kin = ops["subKin"]
            vnuc_frag = ops["subVnuc1"]
            vnuc_env  = ops["subVnuc2"]
            vnuc_bound_frag = ops["subVnuc_bound1"]
            vnuc_bound_env  = ops["subVnuc_bound2"]
            coreJK = ops["subCoreJK"]

            self.tei = ops["subTEI"]
            self.oei_frag = kin + vnuc_frag + vnuc_bound_frag
            self.oei_env  = kin + vnuc_env  + vnuc_bound_env + coreJK


        '''
        self.dim_imp = embedobj.dim_imp
        #self.dim_imp_virt = embedobj.dim_imp_virt
        self.dim_bath = embedobj.dim_bath

        self.loc2sub = embedobj.loc2sub
        self.impAtom = embedobj.impAtom
        self.impJK_sub = None
        self.bathJK_sub = None

        self.Kcoeff = embedobj.Kcoeff

        self.frag_mo = None
        self.env_mo = None

        self.P_frag_loc = embedobj.P_frag_loc
        self.P_env_loc = embedobj.P_env_loc

        self.gtol_dyn = self.params.gtol
        '''

    def kernel(self):

        dim = self.dim
        params = self.params

        P_imp_0  = copy.copy(self.P_imp)
        P_bath_0 = copy.copy(self.P_bath)

        algorithm = params.algorithm
        if(algorithm == '2011'):
            self.umat = self.oep_old(self.umat)
        elif(algorithm.lower() == 'split'):
            self.umat = self.oep_loop(self.umat)
        elif(algorithm.lower() == 'leastsq'):
            self.umat = self.oep_leastsq(self.umat)

        #print ('sum (diag(umat)) = ', np.sum( np.diag( self.umat )) )
        #self.umat = self.umat - np.eye( self.umat.shape[ 0 ] ) * np.average( np.diag( self.umat ) )

        self.P_imp, self.P_bath = self.verify_scf(self.umat)

        print ('|P_imp-P_imp_0| = ', tools.mat_diff_norm(self.P_imp, P_imp_0)) 
        print ('max(P_imp-P_imp_0) = ', tools.mat_diff_max(self.P_imp, P_imp_0))

        print ('|P_bath-P_bath_0| = ', tools.mat_diff_norm(self.P_bath, P_bath_0)) 
        print ('max(P_bath-P_bath_0) = ', tools.mat_diff_max(self.P_bath, P_bath_0))

        return self


    def oep_old(self, umat0, nonscf=False, dm0_frag=None, dm0_env=None):

        '''
        Extended Wu-Yang method
        JCP, 134, 154110 (2011)
        '''

        t0 = (time.clock(),time.time())
       
        params = self.params
        dim = self.dim
        sym_tab = self.sym_tab

        x0 = self.v2m(umat0, dim, False, sym_tab)
        func = ObjFunc_WuYang 
        scf_solver = qcwrap.qc_scf 

        if self.use_suborb:
            #sdmfet
            ao2sub = self.ao2sub[:,:dim]

            scf_args_frag = {'mol':self.mol_frag, 'Ne':self.Ne_frag, 'Norb':dim, 'method':self.mf_method,\
                             'oei':self.oei_frag, 'tei':self.tei, 'dm0':dm0_frag, 'coredm':0.0, \
                             'ao2sub':ao2sub, 'smear_sigma':self.smear_sigma,} 

            scf_args_env  = {'mol':self.mol_env, 'Ne':self.Ne_env, 'Norb':dim, 'method':self.mf_method,\
                             'oei':self.oei_env, 'tei':self.tei, 'dm0':dm0_env, 'coredm':self.core1PDM_ao, \
                             'ao2sub':ao2sub, 'smear_sigma':self.smear_sigma,}

        else:
            #dmfet
            raise NotImplementedError("dmfet NYI")

        func_args = (self.v2m, sym_tab, scf_solver, self.P_ref, dim, self.use_suborb, nonscf, scf_args_frag, scf_args_env,) 
 
        optimizer = OEP_Optimize(params.opt_method, params.options, x0, func, func_args)
        x = optimizer.kernel() 

        umat = self.v2m(x, dim, True, sym_tab)

        t1 = tools.timer("embedding potential optimization (extended Wu-Yang)", t0)

        return umat


    def init_density_partition(self, method = 1):

        print ("Initial density partition")

        self.ks_frag = self.calc_energy_frag(self.umat, None, self.Ne_frag, self.dim)[5]
        self.ks_env = self.calc_energy_env(self.umat, None, self.Ne_env, self.dim)[5]
        
        if(method == 1):
            self.P_imp = self.ks_frag.rdm1
            self.P_bath = self.ks_env.rdm1
        elif(method == 2):
            print ("density partition from input")
            self.P_imp = np.dot(np.dot(self.loc2sub[:,:self.dim].T,self.P_frag_loc),self.loc2sub[:,:self.dim])
            self.P_bath = np.dot(np.dot(self.loc2sub[:,:self.dim].T,self.P_env_loc),self.loc2sub[:,:self.dim])
        elif(method == 3):
            print ("Pipek-Mezey density partition")
            nocc = 0
            nbas = self.ao_bas_tab_frag.size
            for i in range(nbas):
                if(abs(self.ks.mo_occ[i] - 2.0)<1e-8):
                    nocc += 1
            mo_pipek = lo.pipek.PM(self.mol).kernel(self.ks.mo_coeff[:,:nocc], verbose=4)
            s = self.ks.get_ovlp(self.mol)
            self.P_imp[:,:] = 0.0
            self.P_bath[:,:] = 0.0
            Pi = np.zeros([nocc,nbas,nbas])
            pop = np.zeros(nocc)
            for i in range(nocc):
                Pi[i,:,:] = 2.0*np.outer(mo_pipek[:,i],mo_pipek[:,i])
                PiS = np.dot(Pi[i,:,:],s)
                for j in range(nbas):
                    if(self.ao_bas_tab_frag[j] == 1):
                        pop[i] += PiS[j,j]

            print ("Mulliken pop.:")
            print (pop)
            idx = pop.argsort()
            for i in range(nocc):
                if(i<self.Ne_env//2):
                    self.P_bath += tools.dm_ao2sub(Pi[idx[i],:,:], s, self.ao2sub[:,:self.dim])
                else:
                    self.P_imp += tools.dm_ao2sub(Pi[idx[i],:,:], s, self.ao2sub[:,:self.dim])
            for i in range(nocc,nbas):
                if(self.ks.mo_occ[i] > 1e-8):
                    P = self.ks.mo_occ[i]*np.outer(self.ks.mo_coeff[:,i],self.ks.mo_coeff[:,i])
                    self.P_imp += tools.dm_ao2sub(P, s, self.ao2sub[:,:self.dim])
        else:
            dim = self.dim
            dim_imp = self.dim_imp
            subTEI = self.ops["subTEI"]
            P_ref = self.P_ref
            Nelec = self.Ne_frag + self.Ne_env
            self.P_imp, self.P_bath = subspac.fullP_to_fragP(self, subTEI, Nelec, P_ref, dim, dim_imp, self.mf_method)

        #tools.MatPrint(self.P_imp, "P_imp")
        #tools.MatPrint(self.P_bath, "P_bath")

        print ("Ne_imp = ", np.sum(np.diag(self.P_imp)))
        print ("Ne_bath = ", np.sum(np.diag(self.P_bath)))
        diffP = self.P_imp + self.P_bath - self.P_ref
        print ("|P_imp + P_bath - P_ref| = ", np.linalg.norm(diffP))
        print ("max(P_imp + P_bath - P_ref)", np.amax(np.absolute(diffP)))

        P_imp_ao = tools.dm_sub2ao(self.P_imp, self.ao2sub[:,:self.dim])
        P_bath_ao = tools.dm_sub2ao(self.P_bath, self.ao2sub[:,:self.dim])
        cubegen.density(self.mol, "frag_dens_init.cube", P_imp_ao, nx=100, ny=100, nz=100)
        cubegen.density(self.mol, "bath_dens_init.cube", P_bath_ao, nx=100, ny=100, nz=100)


        '''
        dim = self.dim
        frag_occ = np.zeros([dim],dtype = float)
        for i in range(self.Ne_frag//2):
            frag_occ[i] = 2.0
        self.ints.submo_molden(self.frag_mo, frag_occ, self.loc2sub[:,:dim], 'frag_dens_guess.molden' )

        env_occ = np.zeros([dim],dtype = float)
        for i in range(self.Ne_env//2):
            env_occ[i] = 2.0
        self.ints.submo_molden(self.env_mo, env_occ, self.loc2sub[:,:dim], 'env_dens_guess.molden' )
        '''

    def oep_loop(self, umat0):

        '''
        oep with split loops
        '''

        t0 = (time.clock(),time.time())

        params = self.params
        dim = self.dim

        threshold = params.diffP_tol
        maxit = params.outer_maxit
        it = 0
        umat = copy.copy(umat0)
        while it < maxit:
            it += 1
            print (" OEP iteration ", it)

            P_imp_old  = copy.copy(self.P_imp)
            P_bath_old = copy.copy(self.P_bath)

            umat = self.oep_old(umat, nonscf=True, dm0_frag=P_imp_old, dm0_env=P_bath_old)

            if self.use_suborb:
                ao2sub = self.ao2sub[:,:dim]
                scf_args_frag = {'mol':self.mol_frag, 'Ne':self.Ne_frag, 'Norb':dim, 'method':self.mf_method,\
                                 'vext_1e':umat, 'oei':self.oei_frag, 'tei':self.tei, 'dm0':P_imp_old, 'coredm':0.0, \
                                 'ao2sub':ao2sub, 'smear_sigma':self.smear_sigma,}
                mf_frag = qcwrap.qc_scf(True, nonscf=True, **scf_args_frag)
                mf_frag.kernel()
                self.P_imp = mf_frag.rdm1

                scf_args_env  = {'mol':self.mol_env, 'Ne':self.Ne_env, 'Norb':dim, 'method':self.mf_method,\
                                 'vext_1e':umat, 'oei':self.oei_env, 'tei':self.tei, 'dm0':P_bath_old, 'coredm':self.core1PDM_ao, \
                                 'ao2sub':ao2sub, 'smear_sigma':self.smear_sigma,}
                mf_env = qcwrap.qc_scf(True, nonscf=True, **scf_args_env)
                mf_env.kernel()
                self.P_bath = mf_env.rdm1
            else:
                raise NotImplementedError("NYI")


            gmax_imp  = tools.mat_diff_max(self.P_imp, P_imp_old)
            gmax_bath = tools.mat_diff_max(self.P_bath, P_bath_old)
            print ("diffP_max_imp, diffP_max_bath ")
            print (gmax_imp, gmax_bath)

            del(P_imp_old)
            del(P_bath_old)

            #convergence check
            if gmax_imp < threshold and gmax_bath < threshold:
                print ("embedding potential optimization converged")
                break
            if it == maxit:
                print ("STOP: embedding potential optimization exceeds max No. of iterations")

        t1 = tools.timer("embedding potential optimization (split loops)", t0)

        return umat


    def verify_scf(self, umat):

        '''
        Extra SCF calculations with final umat
        '''
        print ('========================================')
        print (' SCF with converged embedding potential ')
        print ('========================================')

        ops = self.ops
        Ne_frag = self.Ne_frag
        Ne_env = self.Ne_env
        dim = self.dim

        coredm = self.core1PDM_ao
        ao2sub = self.ao2sub[:,:dim]

        dm0_frag = None
        if(self.P_imp is None):
            dm0_frag,mo_coeff = tools.fock2onedm(self.oei_frag, Ne_frag//2)
        else:
            dm0_frag = self.P_imp

        mf_frag = qcwrap.qc_scf(True, mol=self.mol_frag, Ne=Ne_frag, Norb=dim, method=self.mf_method,\
                                vext_1e=umat, oei=self.oei_frag, tei=self.tei,\
                                dm0=dm0_frag, coredm=0.0, ao2sub=ao2sub, smear_sigma = self.smear_sigma)
        #mf_frag.init_guess =  'minao'
        #mf_frag.conv_check = False
        mf_frag.kernel()
        FRAG_energy = mf_frag.elec_energy
        FRAG_1RDM = mf_frag.rdm1
        frag_mo = mf_frag.mo_coeff
        frag_occ = mf_frag.mo_occ

        #mf_frag.stability(internal=True, external=False, verbose=5)

        dm0_env = None
        if(self.P_bath is None):
            dm0_env,mo_coeff = tools.fock2onedm(subOEI2, Ne_env//2)
        else:
            dm0_env = self.P_bath

        mf_env = qcwrap.qc_scf(True, mol=self.mol_env, Ne=Ne_env, Norb=dim, method=self.mf_method, \
                               vext_1e=umat, oei=self.oei_env, tei=self.tei,\
                               dm0=dm0_env, coredm=coredm, ao2sub=ao2sub, smear_sigma = self.smear_sigma)
        #mf_env.init_guess =  'minao'
        #mf_env.conv_check = False
        mf_env.kernel()
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
        #P_new = 2.0*np.dot(mo_i[:,:Ne_env//2],mo_i[:,:Ne_env//2].T)
        #print np.linalg.norm(ENV_1RDM - P_new)
        #mf_env = qcwrap.qc_scf(Ne_env,dim,self.mf_method,mol=None,oei=subOEI2,tei=subTEI,dm0=P_new,coredm=coredm,ao2sub=ao2sub)
        #mf_env.runscf()
        #ENV_energy = mf_env.elec_energy
        #ENV_1RDM = mf_env.rdm1

        print ("scf energies:")
        print ("EA (no vemb contrib.) = ",FRAG_energy-np.trace(np.dot(FRAG_1RDM,umat)) )
        print ("EB (no vemb contrib.) = ",ENV_energy-np.trace(np.dot(ENV_1RDM,umat)) )
        #print np.trace(np.dot(FRAG_1RDM,umat)), np.trace(np.dot(ENV_1RDM,umat)),np.trace(np.dot(FRAG_1RDM+ENV_1RDM,umat))
        #self.ints.submo_molden( frag_mo[:,:dim], frag_occ, self.loc2sub, 'frag_dens_scf.molden' )
        #self.ints.submo_molden( env_mo[:,:dim], env_occ, self.loc2sub, 'env_dens_scf.molden' )
        W = FRAG_energy + ENV_energy - np.trace(np.dot(self.P_ref,umat))
        print ("W = ", W)

        deltaN = 0.001
        mf_frag.mo_occ[Ne_frag//2] = deltaN
        mu_A = (mf_frag.energy_tot() - FRAG_energy)/deltaN
        mf_env.mo_occ[Ne_env//2] = deltaN
        mu_B = (mf_env.energy_tot() - ENV_energy)/deltaN
        print ('mu_A = ',mu_A)
        print ('mu_B = ',mu_B)

        mf_frag.mo_occ[Ne_frag//2-1] = mf_frag.mo_occ[Ne_frag//2-1] - deltaN
        mu_A = (-mf_frag.energy_tot() + FRAG_energy)/deltaN
        mf_env.mo_occ[Ne_env//2-1] = mf_env.mo_occ[Ne_env//2-1] - deltaN
        mu_B = (-mf_env.energy_tot() + ENV_energy)/deltaN
        print ('mu_A = ',mu_A)
        print ('mu_B = ',mu_B)

        if(self.P_imp is not None):
            print ("P_imp change (scf - non-scf): ",  tools.mat_diff_norm(FRAG_1RDM, self.P_imp) )
            print ("P_bath change (scf - non-scf): ", tools.mat_diff_norm(ENV_1RDM, self.P_bath) )
        diffP = FRAG_1RDM + ENV_1RDM - self.P_ref
        diffP_norm = np.linalg.norm(diffP)
        diffP_max = np.amax(np.absolute(diffP) )
        print ("|P_frag + P_env - P_ref| = ", diffP_norm)
        print ("max element of (P_frag + P_env - P_ref) = ", diffP_max)

        self.frag_mo = mf_frag.mo_coeff
        self.env_mo = mf_env.mo_coeff

        print ('check orthogonality')
        ortho = np.dot(mf_frag.mo_coeff[:,:Ne_frag//2].T, mf_env.mo_coeff[:,:Ne_env//2])
        print (np.linalg.norm(ortho), np.amax(np.absolute(ortho)))

        return (FRAG_1RDM, ENV_1RDM)




    def oep_newton(self, x, _args):

        ftol = self.params.ftol
        #gtol = self.params.gtol
        gtol = self.gtol_dyn
        print ('gtol = ', gtol)
        maxit = self.params.maxit
        svd_thresh = self.params.svd_thresh

        x_new = newton(self.cost_hess_wuyang,x,args=_args,ftol=ftol,gtol=gtol,maxit=maxit,svd_thresh=svd_thresh)

        return x_new


    def oep_cg(self, x, _args):

        gtol = self.params.gtol
        #gtol = self.gtol_dyn
        print ('gtol = ', gtol)
        maxit = self.params.maxit
        algorithm = self.params.opt_method

        res = optimize.minimize(self.cost_wuyang, x, args=_args, method=algorithm,jac=True, hess=self.hess_wuyang, \
                       options={'maxiter': maxit, 'gtol':gtol, 'disp': True})

        return res



################################################################
#   build_null_space
################################################################
    def build_null_space(self,mf_frag,mf_env,Ne_frag,Ne_env,dim,tol=1e-9):

        size = dim*(dim+1)//2
        hess_frag = oep_hess(mf_frag.mo_coeff, mf_frag.mo_energy, mf_frag.mo_occ, size, dim, self.Ne_frag//2,self.smear_sigma)
        hess_env = oep_hess(mf_env.mo_coeff, mf_env.mo_energy, mf_env.mo_occ, size, dim, self.Ne_env//2,self.smear_sigma)

        #u_f, s_f, vh_f = np.linalg.svd(hess_frag)
        #u_e, s_e, vh_e = np.linalg.svd(hess_env)
        u_f, s_f, vh_f = mkl_svd(hess_frag)
        u_e, s_e, vh_e = mkl_svd(hess_env)

        rankf = tools.rank(s_f,tol)
        ranke = tools.rank(s_e,tol)

        print ('svd of hess')
        print (size-rankf, size-ranke)
        print (s_f[rankf-4:rankf+4])
        print (s_e[ranke-4:ranke+4])

        if (rankf >= size or ranke >= size):
            print ('null space not found!')
            return None

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
        print ('null space shape:',v_fe.shape)
        print ('null space rank:', rankfe)
        #np.set_printoptions(threshold=np.inf)
        #print ss

        if (rankfe >= v_fe.shape[-1]):
            print ('null space not found!')
            return None

        print ('singular value:',ss[rankfe-1:rankfe+1])

        vv = vvh[rankfe:,:].T
        #zero = np.dot(v_fe, vv)
        #print np.linalg.norm(zero),np.amax(np.absolute(zero)) 

        vint = np.dot(v_f[:,rankf:],vv[:(size-rankf),:])
        #vint1 = np.dot(v_e[:,ranke:],vv[(size-rankf):,:])
        for i in range(vint.shape[-1]):
            vint[:,i] = vint[:,i]/np.linalg.norm(vint[:,i])

        print ('check orthogonality of vint:')
        zero = np.dot(vint.T, vint) - np.eye(vint.shape[-1])
        print (np.linalg.norm(zero),np.amax(np.absolute(zero)))

        return vint


#####################################
    def opt_umat_homo_diff(self,umat,gtol=1e-6):

        ops = self.ops
        Ne_frag = self.Ne_frag
        Ne_env = self.Ne_env
        dim = self.dim

        subTEI = ops["subTEI"]
        subOEI1 = ops["subKin"]+ops["subVnuc1"]+ops["subVnuc_bound1"] + umat
        subOEI2 = ops["subKin"]+ops["subVnuc2"]+ops["subVnuc_bound2"]+ops["subCoreJK"] + umat

        coredm = self.core1PDM_ao
        ao2sub = self.ao2sub[:,:dim]

        #frag_coredm_guess = tools.fock2onedm(subOEI1, Ne_frag//2)[0]
        frag_coredm_guess = self.P_imp
        mf_frag = qcwrap.qc_scf(Ne_frag,dim,self.mf_method,mol=self.mol,oei=subOEI1,tei=subTEI,\
                                dm0=frag_coredm_guess,coredm=0.0,ao2sub=ao2sub, smear_sigma = 0.0)
        mf_frag.runscf()

        #env_coredm_guess = tools.fock2onedm(subOEI2, Ne_env//2)[0]
        env_coredm_guess = self.P_bath
        mf_env = qcwrap.qc_scf(Ne_env,dim,self.mf_method,mol=self.mol,oei=subOEI2,tei=subTEI,\
                               dm0=env_coredm_guess,coredm=coredm,ao2sub=ao2sub, smear_sigma = 0.0)
        mf_env.runscf()

        print (np.linalg.norm(mf_frag.rdm1-self.P_imp) )
        print (np.linalg.norm(mf_env.rdm1-self.P_bath) )

        diffP = mf_frag.rdm1 + mf_env.rdm1 - self.P_ref
        print ("|P_frag + P_env - P_ref| = ", np.linalg.norm(diffP) )
        print ("max element of (P_frag + P_env - P_ref) = ", np.amax(np.absolute(diffP) ) )
        self.P_imp = mf_frag.rdm1
        self.P_bath = mf_env.rdm1

        #build null space
        vint = self.build_null_space(mf_frag,mf_env,Ne_frag,Ne_env,dim)
        if vint is None:
            return umat

        #minimize |ehomo_A-ehomo_B|^2
        x0 = tools.mat2vec(umat, dim)
        n = vint.shape[-1]
        c = np.zeros((n))
        res = self.minimize_ehomo_2(c,x0,vint,gtol)
        c = res.x
        #print '|c| = ', np.linalg.norm(c)
        for i in range(n):
            x0 += c[i] * vint[:,i]

        umat = tools.vec2mat(x0, dim)
        #xmat = xmat - np.eye( xmat.shape[ 0 ] ) * np.average( np.diag( xmat ) )

        #tools.MatPrint(umat,'umat')
        #print '|umat| = ', np.linalg.norm(umat)

        return umat


    def minimize_ehomo_2(self,c,x0,vint,gtol):

        _args = (x0,vint)
        ftol = 1e-9
        #gtol = 1e-6
        maxit = 50

        result = optimize.minimize(self.cost_ehomo_2, c, args=_args, method='L-BFGS-B', jac=True,\
                                   options={'disp': True, 'maxiter': maxit,'ftol':ftol, 'gtol':gtol} )

        return result


    def cost_ehomo_2(self,c,x0,vint):

        x = x0.copy()

        n = len(c)
        for i in range(n):
            x += c[i] * vint[:,i]


        ops = self.ops
        Ne_frag = self.Ne_frag
        Ne_env = self.Ne_env
        dim = self.dim

        umat = tools.vec2mat(x, dim)
        subTEI = ops["subTEI"]
        subOEI1 = ops["subKin"]+ops["subVnuc1"]+ops["subVnuc_bound1"] + umat
        subOEI2 = ops["subKin"]+ops["subVnuc2"]+ops["subVnuc_bound2"]+ops["subCoreJK"] + umat

        coredm = self.core1PDM_ao
        ao2sub = self.ao2sub[:,:dim]

        frag_coredm_guess = self.P_imp
        mf_frag = qcwrap.qc_scf(Ne_frag,dim,self.mf_method,mol=self.mol,oei=subOEI1,tei=subTEI,\
                                dm0=frag_coredm_guess,coredm=0.0,ao2sub=ao2sub, smear_sigma = 0.0)
        mf_frag.runscf()

        env_coredm_guess = self.P_bath
        mf_env = qcwrap.qc_scf(Ne_env,dim,self.mf_method,mol=self.mol,oei=subOEI2,tei=subTEI,\
                               dm0=env_coredm_guess,coredm=coredm,ao2sub=ao2sub, smear_sigma = 0.0)
        mf_env.runscf()

        diffP = mf_frag.rdm1 + mf_env.rdm1 - self.P_ref
        #print "|P_frag + P_env - P_ref| = ", np.linalg.norm(diffP)
        #print "max element of (P_frag + P_env - P_ref) = ", np.amax(np.absolute(diffP) )

        ehomo_A = mf_frag.mo_energy[Ne_frag//2-1]
        ehomo_B = mf_env.mo_energy[Ne_env//2-1]
        delta_ehomo = ehomo_A-ehomo_B
        #print 'ehomo_A=',ehomo_A,'  ehomo_B=',ehomo_B, '  delta_ehomo=', delta_ehomo

        elumo_A = mf_frag.mo_energy[Ne_frag//2]
        elumo_B = mf_env.mo_energy[Ne_env//2]
        delta_elumo = elumo_A-elumo_B
        #print 'elumo_A=',elumo_A,'  elumo_B=',elumo_B, '  delta_elumo=', delta_elumo

        #objective function
        f = delta_ehomo*delta_ehomo + delta_elumo*delta_elumo #+ (elumo_A-ehomo_A-0.3)**2


        homo_A = mf_frag.mo_coeff[:,Ne_frag//2-1]
        homo_B = mf_env.mo_coeff[:,Ne_env//2-1]
        lumo_A = mf_frag.mo_coeff[:,Ne_frag//2]
        lumo_B = mf_env.mo_coeff[:,Ne_env//2]
        #gradient
        g = np.zeros((n))
        for i in range(n):
            tmp = tools.vec2mat(vint[:,i],dim)
            EA=np.dot(homo_A,np.dot(tmp,homo_A))
            EB=np.dot(homo_B,np.dot(tmp,homo_B))
            g[i] = 2.0*delta_ehomo*(EA-EB)

            EA=np.dot(lumo_A,np.dot(tmp,lumo_A))
            EB=np.dot(lumo_B,np.dot(tmp,lumo_B))
            g[i] += 2.0*delta_elumo*(EA-EB)

            #EB=np.dot(homo_A,np.dot(tmp,homo_A))
            #g[i] += 2.0*(elumo_A-ehomo_A-0.3)*(EA-EB)

        return (f,g)


#####################################

    def opt_umat_2(self,umat, tol=1e-6):

        #print '|umat| = ', np.linalg.norm(umat)

        ops = self.ops
        Ne_frag = self.Ne_frag
        Ne_env = self.Ne_env
        dim = self.dim

        subTEI = ops["subTEI"]
        subOEI1 = ops["subKin"]+ops["subVnuc1"]+ops["subVnuc_bound1"] + umat
        subOEI2 = ops["subKin"]+ops["subVnuc2"]+ops["subVnuc_bound2"]+ops["subCoreJK"] + umat

        coredm = self.core1PDM_ao
        ao2sub = self.ao2sub[:,:dim]

        #frag_coredm_guess = tools.fock2onedm(subOEI1, Ne_frag//2)[0]
        frag_coredm_guess = self.P_imp
        mf_frag = qcwrap.qc_scf(Ne_frag,dim,self.mf_method,mol=self.mol,oei=subOEI1,tei=subTEI,\
                                dm0=frag_coredm_guess,coredm=0.0,ao2sub=ao2sub, smear_sigma = 0.0)
        mf_frag.runscf()

        #env_coredm_guess = tools.fock2onedm(subOEI2, Ne_env//2)[0]
        env_coredm_guess = self.P_bath
        mf_env = qcwrap.qc_scf(Ne_env,dim,self.mf_method,mol=self.mol,oei=subOEI2,tei=subTEI,\
                               dm0=env_coredm_guess,coredm=coredm,ao2sub=ao2sub, smear_sigma = 0.0)
        mf_env.runscf()

        #print np.linalg.norm(mf_frag.rdm1-self.P_imp)
        #print np.linalg.norm(mf_env.rdm1-self.P_bath)

        diffP = mf_frag.rdm1 + mf_env.rdm1 - self.P_ref
        #print "|P_frag + P_env - P_ref| = ", np.linalg.norm(diffP)
        #print "max element of (P_frag + P_env - P_ref) = ", np.amax(np.absolute(diffP) )
        self.P_imp = mf_frag.rdm1
        self.P_bath = mf_env.rdm1

        #build null space
        vint = self.build_null_space(mf_frag,mf_env,Ne_frag,Ne_env,dim)
        if vint is None:
            return umat

        #minimize |umat|^2
        x0 = tools.mat2vec(umat, dim)
        n = vint.shape[-1]
        c = np.zeros((n))
        res = self.minimize_umat_2(c,x0,vint)
        c = res.x
        #print '|c| = ', np.linalg.norm(c)
        for i in range(n):
            x0 += c[i] * vint[:,i]
            #x0 += 0.1*np.random.random() * vint[:,i]

        umat = tools.vec2mat(x0, dim)
        #xmat = xmat - np.eye( xmat.shape[ 0 ] ) * np.average( np.diag( xmat ) )

        #tools.MatPrint(umat,'umat')
        #print '|umat| = ', np.linalg.norm(umat)

        return umat



    def minimize_umat_2(self,c,x0,vint):
        
        _args = (x0,vint)
        ftol = 1e-12
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
        #print "|umat| = ", np.linalg.norm(umat)
        if(self.params.oep_print >= 3):
            #print "sum(diag(umat)) = ", np.sum(np.diag(umat))
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
            frag_coredm_guess,mo_coeff = tools.fock2onedm(subOEI1, Ne_frag//2)
            FRAG_energy, FRAG_1RDM, frag_mo = qcwrap.pyscf_rhf.scf( subOEI1, subTEI, dim, Ne_frag, frag_coredm_guess, mf_method = self.mf_method )

        if(Ne_env > 0):
            env_coredm_guess,mo_coeff = tools.fock2onedm(subOEI2, Ne_env//2)
            ENV_energy, ENV_1RDM, env_mo = qcwrap.pyscf_rhf.scf( subOEI2, subTEI, dim, Ne_env, env_coredm_guess, mf_method = self.mf_method )

        self.frag_mo = frag_mo
        self.env_mo = env_mo
        self.P_imp = FRAG_1RDM
        self.P_bath = ENV_1RDM

        diffP = FRAG_1RDM + ENV_1RDM - P_ref
        grad = tools.mat2vec(diffP, dim)

        #print "2-norm (grad),       max(grad):"
        #print np.linalg.norm(grad), ", ", np.amax(np.absolute(grad))


        return grad


    def oep_calc_diffP_derivative(self, x, dim, Ne_frag, Ne_env, P_ref, ops):

        t0 = (time.clock(),time.time())
        size = dim*(dim+1)//2

        hess_frag = oep_calc_dPdV(self.frag_mo[:,:-1],self.frag_mo[:,-1],size,self.Ne_frag//2,dim)
        hess_env = oep_calc_dPdV(self.env_mo[:,:-1],self.env_mo[:,-1],size,self.Ne_env//2,dim)

        t1 = tools.timer("hessian", t0)
        return hess_frag + hess_env



    def hess_wuyang(self, x, P_ref, dim, Ne_frag, Ne_env, impJK_sub, bathJK_sub):

        size = dim*(dim+1)//2
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
        hess_frag = oep_hess(mo_coeff_frag, mo_energy_frag, mo_occ_frag, size, dim,self.Ne_frag//2,self.smear_sigma)

        ENV_energy, ENV_1RDM, mo_coeff_env, mo_energy_env, mo_occ_env = qcwrap.pyscf_rhf.scf_oei( subOEI2, dim, Ne_env, self.smear_sigma)
        hess_env = oep_hess(mo_coeff_env, mo_energy_env, mo_occ_env, size, dim,self.Ne_env//2, self.smear_sigma)

        #hess_frag = oep_hess_old(mo_coeff_frag,mo_energy_frag,size,dim,Ne_frag//2)
        #hess_env  = oep_hess_old(mo_coeff_env, mo_energy_env, size,dim,Ne_env//2)
        
        hess = hess_frag + hess_env
        return hess



    def cost_hess_wuyang(self, x, P_ref, dim, Ne_frag, Ne_env, impJK_sub, bathJK_sub, calc_hess=False):

        size = dim*(dim+1)//2
        umat = tools.vec2mat(x, dim)

        print ("|umat| = ", np.linalg.norm(umat) )
        #if(self.params.oep_print >= 3):
        #    print "sum(diag(umat)) = ", np.sum(np.diag(umat))
        #    tools.MatPrint(umat, 'umat')

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

        if(impJK_sub is not None):
            subOEI1 = subKin + subVnuc1 + subVnuc_bound1 + impJK_sub + umat
            subOEI2 = subKin + subVnuc2 + subVnuc_bound2 + subCoreJK + bathJK_sub + umat
            FRAG_energy, FRAG_1RDM, frag_mo_coeff, frag_mo_energy, frag_mo_occ = \
                qcwrap.pyscf_rhf.scf_oei( subOEI1, dim, Ne_frag, self.smear_sigma)
            if(Ne_env > 0):
                ENV_energy, ENV_1RDM, env_mo_coeff, env_mo_energy, env_mo_occ =\
                    qcwrap.pyscf_rhf.scf_oei( subOEI2, dim, Ne_env, self.smear_sigma)

            if(calc_hess):
                #hess_frag = oep_hess(frag_mo_coeff, frag_mo_energy, frag_mo_occ, size, dim, Ne_frag//2, self.smear_sigma)
                #hess_env  = oep_hess(env_mo_coeff, env_mo_energy, env_mo_occ, size, dim, Ne_env//2, self.smear_sigma)
                hess_frag = oep_hess_old(frag_mo_coeff,frag_mo_energy,size,dim,Ne_frag//2)
                hess_env  = oep_hess_old(env_mo_coeff, env_mo_energy, size,dim,Ne_env//2)
                hess = hess_frag + hess_env
        else:
            raise Exception("NYI")

        diffP = FRAG_1RDM + ENV_1RDM - P_ref
        energy = FRAG_energy + ENV_energy - np.trace(np.dot(P_ref,umat))

        grad = tools.mat2vec(diffP, dim)
        #grad = tools.mat2vec_hchain(diffP, dim)
        grad = -1.0 * grad

        print ("2-norm (grad),       max(grad):" )
        print (np.linalg.norm(grad), ", ", np.amax(np.absolute(grad)) )

        f = -energy
        print ('-W = ', f)

        #test
        #f = np.linalg.norm(grad)

        if(calc_hess):
            return (f,grad,hess)
        else:
            return (f,grad)


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

