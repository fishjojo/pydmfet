import numpy
from pyscf.dft import rks
from pydmfet.qcwrap import fermi

def entropy_corr(mo_occ, smear_sigma):

    if mo_occ is None:
        return 0.0

    S = 0.0
    if(smear_sigma >= 1e-8):
        nmo = mo_occ.size
        for i in range(nmo):
            occ_i = mo_occ[i]/2.0 #closed shell
            if(occ_i > 1e-8 and occ_i < 1.0-1e-8):
                S += occ_i * numpy.log(occ_i) + (1.0-occ_i) * numpy.log(1.0-occ_i)
            else:
                S += 0.0

    energy = 2.0*S*smear_sigma
    print 'entropy correction: ',energy
    return energy


def energy_elec(ks, dm=None, h1e=None, vhf=None):

    tot_e,e_hxc = rks.energy_elec(ks, dm=dm, h1e=h1e, vhf=vhf)

    if(ks.smear_sigma > 1e-8 ):
	tot_e += entropy_corr(ks.mo_occ, ks.smear_sigma)

    return tot_e, e_hxc



def get_occ(mf, mo_energy=None, mo_coeff=None, smear_sigma = None):

    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if smear_sigma is None: smear_sigma = mf.smear_sigma

    nmo = mo_energy.size
    mo_occ = numpy.zeros(nmo)
    Nocc = mf.mol.nelectron // 2

    e_idx = numpy.argsort(mo_energy)
    e_sort = mo_energy[e_idx]
    mo_occ[e_idx[:Nocc]] = 1.0

    e_homo = e_sort[Nocc-1]
    e_lumo = e_sort[Nocc]
    print 'HOMO: ',e_homo, 'LUMO: ', e_lumo

    e_fermi = e_homo

    if(smear_sigma > 1e-8):
        e_fermi, mo_occ = fermi.find_efermi(mo_energy, smear_sigma, Nocc, nmo)

    mo_occ *= 2.0  #closed shell

    ne = numpy.sum(mo_occ)
    Ne_error = ne - mf.mol.nelectron
    if(abs(Ne_error) > 1e-8):
        print 'Ne error = ', Ne_error
    print "fermi energy: ", e_fermi
    numpy.set_printoptions(precision=4)
    flag = mo_occ > 1e-4
    print mo_occ[flag]
    numpy.set_printoptions()

    return mo_occ


class rks_ao(rks.RKS):

    '''
        rks with smearing
        wrapper for dft.rks module of pyscf 
    '''

    def __init__(self, mol, smear_sigma = 0.0):

	self.smear_sigma = smear_sigma
        rks.RKS.__init__(self, mol)

    energy_elec = energy_elec
    get_occ = get_occ
