from pydmfet import proj_ao
from pydmfet.qcwrap.pyscf_rks_ao import rks_ao
from pyscf import gto,scf
import numpy as np
from pyscf.tools import molden
from pyscf import lo
from pyscf.lo import iao,orth
from functools import reduce
import math

DMguess  = None

bondlengths = np.arange(0.74, 0.79, 0.1)
energies = []

bas = 'sto-6g'
#bas = 'cc-pvdz'

temp = 0.005

for bondlength in bondlengths:

    nat = 20
    mol = gto.Mole()
    mol.atom = []
    r = 0.5 * bondlength / np.sin(np.pi/nat)
    for i in range(nat):
        theta = i * (2*np.pi/nat)
        mol.atom.append(('H', (r*np.cos(theta), r*np.sin(theta), 0)))

    mol.basis = bas
    mol.build(max_memory = 4000,verbose=4)

    #mf = scf.RKS(mol)
    mf = rks_ao(mol,smear_sigma = temp)
    mf.xc = 'pbe,pbe'
    mf.max_cycle = 50
    mf.scf(dm0=DMguess)

    '''
    proj = np.empty((mol.natm,11,11))
    s = mol.intor_symmetric('int1e_ovlp')
    mo_coeff = mf.mo_coeff[:,:11]
    c = iao.iao(mol, mo_coeff)
    c = np.dot(c, orth.lowdin(reduce(np.dot, (c.conj().T,s,c))))
    with open( 'iao.molden', 'w' ) as thefile:
        molden.header(mol, thefile)
        molden.orbital_coeff(mol, thefile, c)

    for i in range(11):
        mo_coeff[:,i] *= math.sqrt(mf.mo_occ[i])
    for i, (b0, b1, p0, p1) in enumerate(mol.offset_nr_by_atom()):
        csc = reduce(np.dot, (mo_coeff.conj().T,s,c[:,p0:p1]))
        proj[i] = np.dot(csc, csc.conj().T)
    for i in range(11):
        nele = 0
        for j in range(20):
            nele += proj[j,i,i]
        print(nele)

    exit()
    '''

    natoms = mol.natm
    impAtom = np.zeros([natoms], dtype=int)
    for i in range(10):
        impAtom[i] = 1


    embed = proj_ao.proj_embed(mf, impAtom, Ne_env = 10)
    embed.pop_method = 'iao'
    embed.pm_exponent = 2
    embed.make_frozen_orbs(norb = 9)
    exit()

    mol, mo_energy, mo_coeff, mo_occ, irrep_labels, spins = molden.load("mo_lo.molden")
    s = mol.intor_symmetric('int1e_ovlp')
    csc = reduce(np.dot, (mo_coeff.T,s,mo_coeff))
    assert(np.linalg.norm(csc-np.eye(mo_coeff.shape[-1])) < 1e-8)

    embed.mo_lo = np.empty([20,5])
    embed.mo_lo[:,:3] = mo_coeff[:,:3]
    embed.mo_lo[:,3] = mo_coeff[:,4]
    embed.mo_lo[:,4] = mo_coeff[:,8]

    embed.pop_lo = np.array([.5,.5,.5,.5,.5])
    embed.embedding_potential()

    '''
    orbocc = mf.mo_coeff[:,:14]
    c = lo.iao.iao(mol, orbocc)
    s = mol.intor_symmetric('int1e_ovlp')
    #c= np.dot(c, lo.orth.lowdin(reduce(np.dot, (c.T,s,c))))
    
    #with open( 'iao.molden', 'w' ) as thefile:
    #    molden.header(mol, thefile)
    #    molden.orbital_coeff(mol, thefile, c)
    '''
