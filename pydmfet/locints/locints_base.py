import numpy as np
from pyscf import cc
from pydmfet import tools, qcwrap
from functools import reduce

CHARGE_OF = 0

class LocInts:

    def __init__(self, mol, the_mf):

        self.mol = mol
        self.mf = the_mf
        self.Nelec = int(np.rint(np.sum(self.mf.mo_occ)))
        self.NOrb = self.mol.nao_nr()

        self.s1e_ao = self.mf.get_ovlp()
        self.hcore_ao = self.mf.get_hcore()
        self.fock_ao = self.mf.get_fock()

        self.vnuc_ao = self.hcore_ao - self.t_ao


    def energy_nuc(self):
        
        return self.mol.energy_nuc()


    def vnuc_loc(self):

        vnuc = reduce(np.dot, (self.ao2loc.T, self.vnuc_ao, self.ao2loc) )
        return vnuc

    def hcore_loc( self ):

        oei = reduce(np.dot, (self.ao2loc.T, self.hcore_ao, self.ao2loc) )
        return oei

    def fock_loc(self):

        fock = reduce(np.dot, (self.ao2loc.T, self.fock_ao, self.ao2loc) )
        return fock


    def frag_kin_loc(self):

        t_loc = reduce( np.dot, (self.ao2loc.T, self.t_ao, self.ao2loc) )
        return t_loc


    def build_1pdm_loc(self):

        NOcc = self.Nelec/2  #closed shell 
        fock = self.fock_loc()
        onedm, mo_coeff = tools.fock2onedm(fock, NOcc)

        #fullDMao = self.mf.make_rdm1()
        #dm_loc = tools.dm_ao2loc(fullDMao, self.s1e_ao, self.ao2loc)
        #print 'dm_loc - dm_ao2loc(dm_ao) = ',np.linalg.norm(onedm-dm_loc)

        return (onedm, mo_coeff)


    def frag_mol_ao(self, impAtom):

        mol = self.mol
        NAtom = mol.natm

        mol1 = mol.copy()

        for i in range(NAtom):
            if(impAtom[i] == 0):
                mol1._atm[i,CHARGE_OF] = 0 #make environment atoms 0 charge

                symbol = mol1._atom[i][0]
                coord = mol1._atom[i][1]
                mol1._atom[i] = tuple(['ghost_'+symbol,coord]) #write symbol  

        return mol1

    def bound_mol_ao(self, boundary_atoms):

        mol = self.mol
        NAtom = mol.natm

        mol1 = mol.copy()

        for i in range(NAtom):
            mol1._atm[i,CHARGE_OF] = boundary_atoms[i]
            if (boundary_atoms[i] == 0) :
                coord = mol1._atom[i][1]
                mol1._atom[i] = tuple(['ghost',coord])

        return mol1


    def debug_loc_orb(self):

        #fock = self.fock_loc()
        #eigvals, eigvecs = np.linalg.eigh(fock)
        ##eigvecs = eigvecs[ :, eigvals.argsort() ]
        #print eigvals

        oei = self.hcore_loc()
        tei = self.tei_loc()

        Ne = self.Nelec
        NOrb = self.NOrb
        mf = qcwrap.qc_scf(Ne, NOrb, 'hf', mol=self.mol, oei=oei, tei=tei)
        mf.runscf()

        #print mf.elec_energy
        mycc = cc.CCSD(mf).run()
        et = 0.0
        et = mycc.ccsd_t()
        e_hf = mf.elec_energy + self.energy_nuc()
        print (e_hf)
        print (mycc.e_corr + et)

        e_ccsd = e_hf + mycc.e_corr + et
        print ('total ccsd(t) energy = ',e_ccsd)
        exit()



    def check_loc_ortho( self ):

        ShouldBeI = reduce(np.dot, (self.ao2loc.T, self.s1e_ao, self.ao2loc) )
        return np.linalg.norm( ShouldBeI - np.eye( ShouldBeI.shape[0] ) )


