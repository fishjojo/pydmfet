import numpy as np
from pyscf import ao2mo
from pyscf.pbc.df import fft_ao2mo
from pyscf.pbc.scf import hf
from pyscf.pbc.gto.pseudo import get_pp
from pydmfet.locints import locints_base
from functools import reduce

def get_ao2loc(mf,umat):

    mo_coeff = mf.mo_coeff
    ao2loc = np.dot(mo_coeff,umat[0].real)
    return ao2loc


class LocInts_wf(locints_base.LocInts):

    def __init__(self, the_mf, umat):

        cell = the_mf.cell
        self.t_ao = hf.get_t(cell)
        self.ao2loc = get_ao2loc(the_mf,umat)

        locints_base.LocInts.__init__(self, cell, the_mf)

        assert( self.check_loc_ortho() < 1e-8 )
        #self.debug_loc_orb()


    def tei_loc(self):

        tei = None
        if(self.mf._eri is not None):
            tei = ao2mo.incore.full(ao2mo.restore(4, self.mf._eri, self.NOrb), self.ao2loc)
        else:
            tei = fft_ao2mo.general(self.mf.with_df, self.ao2loc, kpts=None, compact=True)

        return tei


    def tei_sub( self, loc2sub, numAct ):

        transfo = np.dot( self.ao2loc, loc2sub[:,:numAct] )
        tei = None
        if(self.mf._eri is not None):
            tei = ao2mo.incore.full(ao2mo.restore(4, self.mf._eri, self.NOrb), transfo)
        else:
            tei = fft_ao2mo.general(self.mf.with_df, transfo, kpts=None, compact=True)

        return tei


    def coreJK_loc( self, DMloc, Kcoeff=1.0 ):

        vj, vk = hf.dot_eri_dm(self.tei_loc(), DMloc, hermi=1)
        vjk = vj - 0.5 * Kcoeff * vk
        return vjk

    def impJK_sub( self, DMsub, ERIsub, Kcoeff=1.0):

        vj, vk=hf.dot_eri_dm(ERIsub, DMsub, hermi=1)
        impJK_sub = vj - 0.5*Kcoeff*vk

        return impJK_sub



    def frag_vnuc_loc(self, impAtom):
        '''
        get fragment Vnuc
        '''

        mol = self.frag_mol_ao(impAtom)
        if not mol.pseudo: raise Exception("no pseudo-potential applied")

        vnuc_ao = get_pp(mol, np.zeros(3)) #gamma point
        vnuc_loc = reduce( np.dot, ( self.ao2loc.T, vnuc_ao, self.ao2loc ) )
        return vnuc_loc


    def bound_vnuc_sub(self, boundary_atoms, loc2sub, numActive):

        vnuc_sub = np.zeros((numActive,numActive),dtype=float)
        if (boundary_atoms is None):
            return vnuc_sub
        else:
            mol = self.bound_mol_ao(boundary_atoms)
            if not mol.pseudo: raise Exception("no pseudo-potential applied")

            vnuc_ao = get_pp(mol, np.zeros(3)) #gamma point
            vnuc_loc = reduce( np.dot, ( self.ao2loc.T, vnuc_ao, self.ao2loc ))
            vnuc_sub = reduce( np.dot, (loc2sub[:,:numActive].T, vnuc_loc, loc2sub[:,:numActive] ))
            return vnuc_sub

