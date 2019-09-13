from pydmfet import locints, sdmfet,oep,tools
from pydmfet.qcwrap.pyscf_rks_ao import rks_ao
from pyscf import gto, scf, ao2mo
import numpy as np
from pyscf.tools import molden

t0 = tools.time0()

bas ='6-31G*'
temp = 0.005


mol = gto.Mole()
mol.atom = open('O2-C24.xyz').read()
mol.basis = bas
mol.charge = 0
mol.build(max_memory = 24000, verbose=4)

dm_guess=None
_, _, mo_coeff, mo_occ, _, _ = molden.load("MO_pbe.molden")
dm_guess = np.dot(mo_coeff*mo_occ, mo_coeff.T)

mf = rks_ao(mol,smear_sigma = temp)
mf.xc = "pbe,pbe"
mf.max_cycle = 50
mf.scf(dm0=dm_guess)

myInts = locints.LocalIntegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )

natoms = mol.natm
impAtom = np.zeros([natoms], dtype=int)
for i in range(8):
    impAtom[i] = 1


ghost_frag = 1-impAtom
ghost_env = 1-ghost_frag

mol_frag = gto.Mole()
mol_frag.atom = tools.add_ghost(mol.atom, ghost_frag)
mol_frag.basis = bas
mol_frag.charge = -6
mol_frag.build(max_memory = 24000,verbose = 4)

mol_env = gto.Mole()
mol_env.atom = tools.add_ghost(mol.atom, ghost_env)
mol_env.basis =  bas
mol_env.charge = 6
mol_env.build(max_memory = 24000,verbose = 4)

boundary_atoms = np.zeros([natoms])
boundary_atoms2 = np.zeros([natoms])
boundary_atoms[8:14] = 1
boundary_atoms2[8:14] = -1


Ne_frag = 58
aoslice = mol.aoslice_by_atom()
impurities = np.zeros([mol.nao_nr()], dtype = int)
for i in range(natoms):
    if(impAtom[i] == 1):
        impurities[aoslice[i,2]:aoslice[i,3]] = 1

params = oep.OEPparams(algorithm = 'split', opt_method = 'L-BFGS-B', diffP_tol=1e-4, outer_maxit = 40)
params.options['ftol'] = 1e-10
params.options['gtol'] = 1e-4
params.options['maxiter'] = 100
params.options['svd_thresh'] = 1e-2

umat = None
umat = np.load("umat_guess.npy")

theDMFET = sdmfet.DMFET( mf, mol_frag, mol_env,myInts,impurities, impAtom, Ne_frag, \
                         boundary_atoms=boundary_atoms, boundary_atoms2=boundary_atoms2,\
                         umat = umat, dim_imp =None, dim_bath=None, dim_big =None, smear_sigma = temp, \
                         oep_params=params,ecw_method='ccsd', mf_method = mf.xc,\
                         use_umat_ao=False, scf_max_cycle = 100, frac_occ_tol = 1e-4)

umat = theDMFET.embedding_potential()

t1 = tools.timer("total calc time", t0)
