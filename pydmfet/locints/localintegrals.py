from pyscf import gto, scf, ao2mo, tools
from pyscf import lo
from pyscf.lo import nao, orth
from pyscf.lo.boys import Boys
from pyscf.tools import molden
from pydmfet.locints import iao_helper
from pydmfet import tools
import numpy as np
import copy
from functools import reduce


class LocalIntegrals:

    def __init__( self, the_mf, active_orbs, localizationtype, ao_rotation=None, use_full_hessian=True, localization_threshold=1e-6 ):

        assert (( localizationtype == 'meta_lowdin' ) or ( localizationtype == 'boys' ) or ( localizationtype == 'lowdin' ) or ( localizationtype == 'iao' ))
        
        # Information on the full SCF problem
        self.mo_occ = the_mf.mo_occ
        self.mol        = the_mf.mol
        self.s1e_ao = the_mf.get_ovlp()
        self.fullEhf    = the_mf.e_tot
        #self.fullDMao   = np.dot(np.dot( the_mf.mo_coeff, np.diag( the_mf.mo_occ )), the_mf.mo_coeff.T )
        self.fullDMao = the_mf.make_rdm1()
        #self.fullJKao   = scf.hf.get_veff( self.mol, self.fullDMao, 0, 0, 1 ) #Last 3 numbers: dm_last, vhf_last, hermi
        #self.fullJKao = the_mf.get_veff()
        #self.fullFOCKao = self.mol.intor_symmetric('int1e_kin') + self.mol.intor_symmetric('int1e_nuc') + self.fullJKao
        self.fullFOCKao = the_mf.get_fock()
        self.fullOEIao = the_mf.get_hcore()


        # Active space information
        self._which   = localizationtype
        self.active   = np.zeros( [ self.mol.nao_nr() ], dtype=int )
        self.active[ active_orbs ] = 1
        self.NOrb    = np.sum( self.active ) # Number of active space orbitals
        self.Nelec    = int(np.rint( self.mol.nelectron - np.sum( the_mf.mo_occ[ self.active==0 ] ))) # Total number of electrons minus frozen part
        
        # Localize the orbitals
        if (( self._which == 'meta_lowdin' ) or ( self._which == 'boys' )):
            if ( self._which == 'meta_lowdin' ):
                assert( self.NOrb == self.mol.nao_nr() ) # Full active space required
            if ( self._which == 'boys' ):
                self.ao2loc = the_mf.mo_coeff[ : , self.active==1 ]
            if ( self.NOrb == self.mol.nao_nr() ): # If you want the full active, do meta-Lowdin
                nao.AOSHELL[4] = ['1s0p0d0f', '2s1p0d0f'] # redefine the valence shell for Be
                self.ao2loc = orth.orth_ao( self.mol, 'meta_lowdin' )
                if ( ao_rotation is not None ):
                    self.ao2loc = np.dot( self.ao2loc, ao_rotation.T )
            if ( self._which == 'boys' ):
                old_verbose = self.mol.verbose
                self.mol.verbose = 5
                #loc = localizer.localizer( self.mol, self.ao2loc, self._which, use_full_hessian )
                boys = Boys(self.mol, self.ao2loc )
                self.mol.verbose = old_verbose
                #self.ao2loc = loc.optimize( threshold=localization_threshold )
                self.ao2loc = boys.kernel()
            self.TI_OK = False # Check yourself if OK, then overwrite
        if ( self._which == 'lowdin' ):
            assert( self.NOrb == self.mol.nao_nr() ) # Full active space required
            ovlp = self.mol.intor_symmetric('int1e_ovlp')
            ovlp_eigs, ovlp_vecs = np.linalg.eigh( ovlp )
            assert ( np.linalg.norm( np.dot( np.dot( ovlp_vecs, np.diag( ovlp_eigs ) ), ovlp_vecs.T ) - ovlp ) < 1e-10 )
            self.ao2loc = np.dot( np.dot( ovlp_vecs, np.diag( np.power( ovlp_eigs, -0.5 ) ) ), ovlp_vecs.T )
            self.TI_OK  = False # Check yourself if OK, then overwrite
        if ( self._which == 'iao' ):
            assert( self.NOrb == self.mol.nao_nr() ) # Full active space assumed
            self.ao2loc = iao_helper.localize_iao( self.mol, the_mf )
            if ( ao_rotation is not None ):
                self.ao2loc = np.dot( self.ao2loc, ao_rotation.T )
            self.TI_OK = False # Check yourself if OK, then overwrite
            #self.molden( 'dump.molden' ) # Debugging mode
        assert( self.loc_ortho() < 1e-9 )

        '''
        # Effective Hamiltonian due to frozen part
        self.frozenDMmo  = np.array( the_mf.mo_occ, copy=True )
        self.frozenDMmo[ self.active==1 ] = 0 # Only the frozen MO occupancies nonzero
        self.frozenDMao  = np.dot(np.dot( the_mf.mo_coeff, np.diag( self.frozenDMmo )), the_mf.mo_coeff.T )
        self.frozenJKao  = scf.hf.get_veff( self.mol, self.frozenDMao, 0, 0, 1 ) #Last 3 numbers: dm_last, vhf_last, hermi
        self.frozenOEIao = self.fullFOCKao - self.fullJKao + self.frozenJKao
        '''
        self.frozenOEIao = self.fullOEIao

        # Active space OEI and ERI
        self.activeCONST = self.mol.energy_nuc() #+ np.einsum( 'ij,ij->', self.frozenOEIao - 0.5*self.frozenJKao, self.frozenDMao )
        self.activeOEI   = np.dot( np.dot( self.ao2loc.T, self.frozenOEIao ), self.ao2loc )
        self.activeFOCK  = np.dot( np.dot( self.ao2loc.T, self.fullFOCKao  ), self.ao2loc )
        if ( self.NOrb <= 150 ):
            self.ERIinMEM  = True

            if(self.mol.cart):
                intor='int2e_cart'
            else:
                intor='int2e_sph'

            self.activeERI = ao2mo.outcore.full_iofree( self.mol, self.ao2loc, intor, compact=False ).reshape(self.NOrb, self.NOrb, self.NOrb, self.NOrb)
        else:
            self.ERIinMEM  = False
            self.activeERI = None
        
        #self.debug_matrixelements()
        
    def loc_molden( self, filename ):
    
        with open( filename, 'w' ) as thefile:
            molden.header( self.mol, thefile )
            molden.orbital_coeff( self.mol, thefile, self.ao2loc )

    def locmo_molden( self, mo_coeff, mo_occ, filename):

        transfo = np.dot( self.ao2loc, mo_coeff )
        with open( filename, 'w' ) as thefile:
            molden.header( self.mol, thefile )
            molden.orbital_coeff( self.mol, thefile, transfo, occ=mo_occ )


    def sub_molden( self, loc2sub, filename, mo_occ=None ):

        transfo = np.dot( self.ao2loc, loc2sub )
        with open( filename, 'w' ) as thefile:
            molden.header( self.mol, thefile )
            molden.orbital_coeff( self.mol, thefile, transfo, occ=mo_occ )

    def submo_molden( self, mo_coeff, mo_occ, loc2sub, filename, mol = None ):

        if mol is None: mol = self.mol

        dim = mo_coeff.shape[0]
        mo_loc = np.dot( loc2sub[:,:dim], mo_coeff )
        transfo = np.dot( self.ao2loc, mo_loc )
        with open( filename, 'w' ) as thefile:
            molden.header( mol, thefile )
            molden.orbital_coeff( mol, thefile, transfo, occ=mo_occ )
            
    def loc_ortho( self ):
    
        ShouldBeI = np.dot( np.dot( self.ao2loc.T , self.mol.intor_symmetric('int1e_ovlp') ) , self.ao2loc )
        return np.linalg.norm( ShouldBeI - np.eye( ShouldBeI.shape[0] ) )
       
    def debug_matrixelements( self ):
    
        eigvals, eigvecs = np.linalg.eigh( self.activeFOCK )
        eigvecs = eigvecs[ :, eigvals.argsort() ]
        assert( self.Nelec % 2 == 0 )
        numPairs = self.Nelec // 2
        DMguess = 2 * np.dot( eigvecs[ :, :numPairs ], eigvecs[ :, :numPairs ].T )
        if ( self.ERIinMEM == True ):
            DMloc = rhf.solve_ERI( self.activeOEI, self.activeERI, DMguess, numPairs )
        else:
            DMloc = rhf.solve_JK( self.activeOEI, self.mol, self.ao2loc, DMguess, numPairs )
        newFOCKloc = self.loc_rhf_fock_bis( DMloc )
        newRHFener = self.activeCONST + 0.5 * np.einsum( 'ij,ij->', DMloc, self.activeOEI + newFOCKloc )
        print ("2-norm difference of RDM(self.activeFOCK) and RDM(self.active{OEI,ERI})  =", np.linalg.norm( DMguess - DMloc ))
        print ("2-norm difference of self.activeFOCK and FOCK(RDM(self.active{OEI,ERI})) =", np.linalg.norm( self.activeFOCK - newFOCKloc ))
        print ("RHF energy of mean-field input           =", self.fullEhf)
        print ("RHF energy based on self.active{OEI,ERI} =", newRHFener)
 
    def const( self ):
    
        return self.activeCONST
        
    def hcore_loc( self ):
        
        return self.activeOEI
        
    def loc_rhf_fock( self ):
    
        return self.activeFOCK
        
    def loc_rhf_fock_bis( self, DMloc ):
    
        if ( self.ERIinMEM == False ):
            DM_ao = np.dot( np.dot( self.ao2loc, DMloc ), self.ao2loc.T )
            JK_ao = scf.hf.get_veff( self.mol, DM_ao, 0, 0, 1 ) #Last 3 numbers: dm_last, vhf_last, hermi
            JK_loc = np.dot( np.dot( self.ao2loc.T, JK_ao ), self.ao2loc )
        else:
            JK_loc = np.einsum( 'ijkl,ij->kl', self.activeERI, DMloc ) - 0.5 * np.einsum( 'ijkl,ik->jl', self.activeERI, DMloc )
        FOCKloc = self.activeOEI + JK_loc
        return FOCKloc

    def get_loc_mo(self, smear_sigma):

        from pydmfet.qcwrap.fermi import find_efermi       
 
        fock = self.activeFOCK
        eigenvals, eigenvecs = np.linalg.eigh( fock )
        idx = np.argmax(abs(eigenvecs), axis=0)
        eigenvecs[:,eigenvecs[ idx, np.arange(len(eigenvals)) ]<0] *= -1
        Nocc = self.Nelec//2  #closed shell

        e_homo = eigenvals[Nocc-1]
        e_lumo = eigenvals[Nocc]
        print ('HOMO: ', e_homo, 'LUMO: ', e_lumo)
        print ("mo_energy:")
        print (eigenvals[:Nocc+5])

        e_fermi = e_homo
        mo_occ = np.zeros((self.NOrb))

        if(smear_sigma < 1e-8): #T=0
            mo_occ[:Nocc] = 1.0
        else: #finite T
            e_fermi, mo_occ = find_efermi(eigenvals, smear_sigma, Nocc, self.NOrb)

        mo_occ*=2.0 #closed shell

        Ne_error = np.sum(mo_occ) - self.Nelec
        if(Ne_error > 1e-8):
            print ('Ne error = ', Ne_error)
        print ("fermi energy: ", e_fermi)
        np.set_printoptions(precision=4)
        flag = mo_occ > 1e-4
        print (mo_occ[flag])
        np.set_printoptions()

        mo_coeff = eigenvecs
        mo_energy = eigenvals

        return mo_coeff, mo_occ, mo_energy


    def tei_loc( self ):

        if ( self.ERIinMEM == False ):
            t0 = tools.time0()
            #print "LocalIntegrals.tei_loc : ERI of the localized orbitals are not stored in memory."
            if(self.mol.cart):
                intor='int2e_cart'
            else:
                intor='int2e_sph'
            TEI_4 = ao2mo.outcore.full_iofree(self.mol, self.ao2loc, intor)
            TEI_8 = ao2mo.restore(8, TEI_4, self.NOrb)
            TEI_4 = None
            t1 = tools.timer("LocalIntegrals.tei_loc",t0)
            return TEI_8
        else:
            return self.activeERI

        
    def dmet_oei( self, loc2dmet, numActive ):
    
        OEIdmet = np.dot( np.dot( loc2dmet[:,:numActive].T, self.activeOEI ), loc2dmet[:,:numActive] )
        return OEIdmet
        
    def dmet_fock( self, loc2dmet, numActive, coreDMloc ):
    
        FOCKdmet = np.dot( np.dot( loc2dmet[:,:numActive].T, self.loc_rhf_fock_bis( coreDMloc ) ), loc2dmet[:,:numActive] )

        return FOCKdmet
        
    def dmet_init_guess_rhf( self, loc2dmet, numActive, numPairs, Nimp, chempot_imp ):
    
        Fock_small = np.dot( np.dot( loc2dmet[:,:numActive].T, self.activeFOCK ), loc2dmet[:,:numActive] )
        if (chempot_imp != 0.0):
            for orb in range(Nimp):
                Fock_small[ orb, orb ] -= chempot_imp
        eigvals, eigvecs = np.linalg.eigh( Fock_small )
        eigvecs = eigvecs[ :, eigvals.argsort() ]
        DMguess = 2 * np.dot( eigvecs[ :, :numPairs ], eigvecs[ :, :numPairs ].T )
        return DMguess
        
    def tei_sub( self, loc2dmet, numAct ):

        TEIdmet_8 = None

        t0 = tools.time0() 
        if ( self.ERIinMEM == False ):
            transfo = np.dot( self.ao2loc, loc2dmet[:,:numAct] )
            #TEIdmet = ao2mo.outcore.full_iofree(self.mol, transfo, compact=False).reshape(numAct, numAct, numAct, numAct)
            if(self.mol.cart):
                intor='int2e_cart'
            else:
                intor='int2e_sph'
            TEIdmet_4 = ao2mo.outcore.full_iofree(self.mol, transfo, intor)
            TEIdmet_8 = ao2mo.restore(8, TEIdmet_4, numAct)
            TEIdmet_4 = None
        else:
            #TEIdmet = ao2mo.incore.full(ao2mo.restore(8, self.activeERI, self.NOrb), loc2dmet[:,:numAct], compact=False).reshape(numAct, numAct, numAct, numAct)
            TEIdmet_8 = ao2mo.incore.full(ao2mo.restore(8, self.activeERI, self.NOrb), loc2dmet[:,:numAct])

        t1 = tools.timer("LocalIntegrals.tei_sub",t0)
        return TEIdmet_8
        
    def frag_mol_ao(self, impAtom):
        '''
            Xing: get fragment Vnuc
        '''
        mol = self.mol
        NAtom = mol.natm

        mol1 = mol.copy()
        for i in range(NAtom):
            if(impAtom[i] == 0):
                mol1._atm[i][0] = 0 #make environment atoms ghost

        return mol1


    def bound_vnuc_ao(self, boundary_atoms):

        mol = self.mol
        NAtom = mol.natm

        vnuc = 0.0
        mol1 = mol.copy()
        for i in range(NAtom):
            #mol1._atm[i][0] = boundary_atoms[i]
            if( abs(boundary_atoms[i]) > 1e-6 ):
                mol1.set_rinv_origin(mol.atom_coord(i))
                vnuc += -1.0*boundary_atoms[i] * mol1.intor_symmetric('int1e_rinv_sph')

        return vnuc


    def bound_vnuc_sub(self, boundary_atoms, loc2sub, numActive):

        vnuc_sub = np.zeros((numActive,numActive),dtype=float)
        if (boundary_atoms is None):
            return vnuc_sub 
        else:
            vnuc_ao = self.bound_vnuc_ao(boundary_atoms)
            vnuc_loc = reduce( np.dot, ( self.ao2loc.T, vnuc_ao, self.ao2loc ))
            vnuc_sub = reduce( np.dot, (loc2sub[:,:numActive].T, vnuc_loc, loc2sub[:,:numActive] ))

            return vnuc_sub


    def frag_oei_loc(self, impAtom):

        mol = self.frag_mol_ao(impAtom)
        oei_ao = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')

        oei_loc = np.dot( np.dot( self.ao2loc.T, oei_ao ), self.ao2loc )
        return oei_loc

    def frag_oei_sub( self, impAtom, loc2sub, numActive):

        oei_loc = self.frag_oei_loc(impAtom )
        subOEI = np.dot( np.dot( loc2sub[:,:numActive].T, oei_loc ), loc2sub[:,:numActive] )
        return subOEI

    '''
    def frag_vnuc_loc(self, impAtom):

        mol = self.frag_mol_ao(impAtom)

        oei_ao = mol.intor_symmetric('int1e_nuc')
        oei_loc = reduce( np.dot, ( self.ao2loc.T, oei_ao, self.ao2loc ) )
        return oei_loc
    '''

    def frag_vnuc_loc(self, mol):

        oei_ao = mol.intor_symmetric('int1e_nuc')
        if mol.has_ecp():
            oei_ao += mol.intor_symmetric('ECPscalar')
        oei_loc = reduce( np.dot, ( self.ao2loc.T, oei_ao, self.ao2loc ) )
        return oei_loc


    def frag_kin_loc(self):

        oei_ao = self.mol.intor_symmetric('int1e_kin')
        oei_loc = reduce( np.dot, (self.ao2loc.T, oei_ao, self.ao2loc) )
        return oei_loc


    def frag_fock_sub( self, impAtom, loc2sub, numActive, coreOneDM_loc):
        
        oei_sub = self.frag_oei_sub( impAtom, loc2sub, numActive)
        coreJK_sub = self.coreJK_sub( loc2sub, numActive, coreOneDM_loc )
        fock_sub = coreJK_sub + oei_sub
        return fock_sub


    def build_1pdm_loc(self):

        mo_coeff = None
        #NOcc = self.Nelec//2  #closed shell 
        #fock = self.loc_rhf_fock()
        #onedm,mo_coeff = tools.fock2onedm(fock, NOcc)

        dm_loc = tools.dm_ao2loc(self.fullDMao, self.s1e_ao, self.ao2loc)
        #print 'dm_loc - dm_ao = ',np.linalg.norm(onedm-dm_loc)

        return (dm_loc,mo_coeff)


    def coreJK_loc( self, DMloc, Kcoeff=1.0 ):

        if ( self.ERIinMEM == False ):
            DM_ao = np.dot( np.dot( self.ao2loc, DMloc ), self.ao2loc.T )
            vj,vk = scf.hf.get_jk(self.mol, DM_ao, hermi=1)
            JK_ao = vj - vk * 0.5 * Kcoeff
            #JK_ao = scf.hf.get_veff( self.mol, DM_ao, 0, 0, 1 ) #Last 3 numbers: dm_last, vhf_last, hermi
            JK_loc = np.dot( np.dot( self.ao2loc.T, JK_ao ), self.ao2loc )
        else:
            JK_loc = np.einsum( 'ijkl,ij->kl', self.activeERI, DMloc ) - 0.5 * Kcoeff * np.einsum( 'ijkl,ik->jl', self.activeERI, DMloc )

        return JK_loc


    def impJK_sub( self, DMsub, ERIsub, Kcoeff=1.0):

        #impJK_sub = np.einsum( 'ijkl,ij->kl', ERIsub, DMsub ) - 0.5 * np.einsum( 'ijkl,ik->jl', ERIsub, DMsub )

        j, k=scf.hf.dot_eri_dm(ERIsub, DMsub, hermi=1)
        impJK_sub = j - 0.5*Kcoeff*k

        return impJK_sub


    def fock_sub( self, loc2sub, dim, coreDMloc ):

        fock_sub = np.dot( np.dot( loc2sub[:,:dim].T, self.loc_rhf_fock_bis( coreDMloc ) ), loc2sub[:,:dim] )

        return fock_sub


    def energy_nuc(self):

        return self.mol.energy_nuc()
