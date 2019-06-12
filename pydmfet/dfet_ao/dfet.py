import numpy as np
import pydmfet
from pydmfet.dfet_ao import scf,oep
from pyscf.tools import cubegen
from pydmfet import tools

def bound_vnuc_ao(dfet, boundary_atoms, mol=None):

    if mol is None: mol = dfet.mol
    NAtom = mol.natm

    vnuc = 0.0
    mol1 = mol.copy()
    for i in range(NAtom):
        #mol1._atm[i][0] = boundary_atoms[i]
        if( abs(boundary_atoms[i]) > 1e-6 ):
            mol1.set_rinv_origin(mol.atom_coord(i))
            vnuc += -1.0*boundary_atoms[i] * mol1.intor_symmetric('int1e_rinv_sph')

    return vnuc



class DFET:

    def __init__(self, mf_full,mol_frag, mol_env, Ne_frag, Ne_env, boundary_atoms=None, boundary_atoms2=None, \
		 umat = None, oep_params = pydmfet.oep.OEPparams(), smear_sigma = 0.0,\
		 ecw_method = 'HF', mf_method = 'HF', ex_nroots = 1, \
		 plot_dens = True):

	self.mf_full = mf_full
        self.mol = self.mf_full.mol
	self.boundary_atoms = boundary_atoms
	self.boundary_atoms2 = boundary_atoms2

	self.umat = umat
	self.smear_sigma = smear_sigma

	self.P_ref = None#self.mf_full.make_rdm1()
	self.P_imp = None
	self.P_bath = None

	self.mol_frag = mol_frag
	self.mol_env = mol_env

	self.Ne_frag = Ne_frag
	self.Ne_env = Ne_env

        self.ecw_method = ecw_method.lower()
	self.mf_method = mf_method.lower()
	self.ex_nroots = ex_nroots

        self.oep_params = oep_params

	self.dim = self.mol.nao_nr()
	
	if(self.umat is None):
            self.umat = np.zeros((self.dim,self.dim))

	self.plot_dens = plot_dens

	self.vnuc_bound_frag = 0.0
	self.vnuc_bound_env  = 0.0

	if(boundary_atoms is not None): 
	    self.vnuc_bound_frag = self.bound_vnuc_ao(boundary_atoms)
	if(boundary_atoms2 is not None):
	    self.vnuc_bound_env  = self.bound_vnuc_ao(boundary_atoms2)

	#test total system mf calculation with smearing
	mf = scf.EmbedSCF(self.mol, 0.0, smear_sigma = self.smear_sigma)
        mf.xc = self.mf_method
        mf.scf()
	self.P_ref = mf.make_rdm1()
	#tools.MatPrint(self.P_ref,"P_ref")
	print "mo_energy:"
	print mf.mo_energy
	print "mo_occ:"
	print mf.mo_occ

	if(self.plot_dens):
            cubegen.density(self.mol, "tot_dens.cube", self.P_ref, nx=100, ny=100, nz=100)


    bound_vnuc_ao = bound_vnuc_ao


    def calc_umat(self):
      
        myoep = oep.OEPao(self, self.oep_params)
        self.umat = myoep.kernel()
	self.P_imp = myoep.P_imp
	self.P_bath = myoep.P_bath


    def embedding_potential(self):

        self.calc_umat()

	ovlp = self.mol.intor_symmetric('int1e_ovlp')
	inv_S = np.linalg.inv(ovlp)
	#print(np.dot(ovlp,inv_S))
	umat_ao = reduce(np.dot,(inv_S,self.umat,inv_S))

	#tools.MatPrint(umat_ao,'S^-1*umat_ao*S^-1')
	#tools.MatPrint(self.P_imp,'P_frag')

	if(self.plot_dens):
            cubegen.density(self.mol, "frag_dens.cube", self.P_imp, nx=100, ny=100, nz=100)
            cubegen.density(self.mol, "env_dens.cube", self.P_bath, nx=100, ny=100, nz=100)
	    cubegen.density(self.mol, "vemb.cube", umat_ao, nx=100, ny=100, nz=100)

        return self.umat

    '''
    def correction_energy(self):

        energy = 0.0

        #if(self.umat is None):
        #    self.calc_umat()

	print "Performing ECW energy calculation"
	energy = self.ecw_energy(self.ecw_method)


        return energy


    def ecw_energy(self, method):

	mf_energy, Vemb, Vxc = self.imp_mf_energy(dim)

	Ne_frag = self.Ne_frag
        ops = self.ops

        subOEI = ops["subKin"] + ops["subVnuc1"] + Vemb
        subTEI = ops["subTEI"]

	mf = scf.EmbedSCF(self.mol_frag, Vemb)
        mf.xc = 'hf'
        mf.scf()
	e_hf = mf.e_tot

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
                print 'CCSD(T) correction'
                et = mycc.ccsd_t()

	    energy = e_hf + mycc.e_corr + et
	elif(method == 'eomccsd'):
	    mf.verbose = 5
	    mycc = cc.RCCSD(mf)
            mycc.kernel()
            mycc.eomee_ccsd_singlet(nroots=self.ex_nroots)
	    energy = e_hf + mycc.e_corr 
        else:
            raise Exception("ecw_method not supported!")

	energy -= mf_energy

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

        print '|diffP| = ', np.linalg.norm(mf.rdm1 + self.P_bath - self.P_ref_sub)
        print "embeded imp scf (electron) energy = ",energy

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
        energy = mf.elec_energy

        print '|diffP| = ', np.linalg.norm(mf.rdm1 + self.P_bath - self.P_ref_sub)
        print "embeded imp scf (electron) energy = ",energy

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

	mf = qcwrap.qc_scf(Ne_frag, dim, self.mf_method, mol=self.mol_frag, oei=subOEI, tei=subTEI, dm0=self.P_imp, coredm=0.0, ao2sub=ao2sub)
        mf.runscf()
	energy = mf.elec_energy

	#self.ints.submo_molden(mf.mo_coeff, mf.mo_occ, self.loc2sub, "mo_frag.molden" )
	dm_ao = tools.dm_sub2ao(mf.rdm1, ao2sub)
	cubegen.density(self.mol, "frag_dens.cube", dm_ao, nx=100, ny=100, nz=100)
	dm_ao = tools.dm_sub2ao(self.P_bath, ao2sub)
        cubegen.density(self.mol, "env_dens.cube", dm_ao, nx=100, ny=100, nz=100)

	print '|diffP| = ', np.linalg.norm(mf.rdm1 + self.P_bath - self.P_ref_sub)
        print "embeded imp scf (electron) energy = ",energy

	self.P_imp = mf.rdm1

        return (energy, Vemb, Vxc)

    '''
