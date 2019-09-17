from pydmfet import oep, tools
from pydmfet.dfet_ao import dfet
from pyscf import gto, scf
import numpy as np
import time

bas = 'ccpvdz'

t0 = (time.clock(), time.time())
mol = gto.Mole()
mol.atom = open('C24.xyz').read()
mol.basis = bas
mol.charge = 0
mol.symmetry=False
mol.build(max_memory = 48000, verbose=4)

mf = scf.RKS(mol)
mf.xc = 'hf'
mf.max_cycle = 100
mf.scf()

t1 = tools.timer("full scf", t0)

natoms = mol.natm
impAtom = np.zeros([natoms], dtype=int)
for i in range(6):
    impAtom[i] = 1

ghost_frag = 1-impAtom
ghost_env = 1-ghost_frag

mol_frag = gto.Mole()
mol_frag.atom = tools.add_ghost(mol.atom, ghost_frag)
mol_frag.charge = -6
mol_frag.basis = bas
mol_frag.symmetry=False
mol_frag.build(max_memory = 48000,verbose = 3)

mol_env = gto.Mole()
mol_env.atom = tools.add_ghost(mol.atom, ghost_env)
mol_env.charge = 6
mol_env.basis =  bas
mol_env.symmetry=False
mol_env.build(max_memory = 48000,verbose = 3)

#Ne_frag = 42
boundary_atoms = np.zeros([natoms])
boundary_atoms[6:12] = 1.0
boundary_atoms2 = np.zeros([natoms])
boundary_atoms2[:6] = -1.0

params = oep.OEPparams(algorithm = 'split', opt_method = 'L-BFGS-B', \
                       ftol = 1e-11, gtol = 5e-5,diffP_tol=5e-5, outer_maxit = 200, maxit = 200,l2_lambda = 0.0, oep_print = 0)

theDMFET = dfet.DFET(mf, mol_frag, mol_env,\
                     boundary_atoms=boundary_atoms, boundary_atoms2=boundary_atoms2,\
                     oep_params=params, smear_sigma=0.005, ecw_method = 'ccsd(t)',mf_method = mf.xc)


umat = theDMFET.embedding_potential()

t2 = tools.timer("dfet_ao", t1)
t3 = tools.timer("total calc", t0)
