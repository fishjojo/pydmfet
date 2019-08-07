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


class foo:
    def __init__(self):

        return None

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

