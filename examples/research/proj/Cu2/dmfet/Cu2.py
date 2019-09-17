from pydmfet import locints, sdmfet, oep, tools, dfet_ao, proj_ao
from pydmfet.dfet_ao import dfet
from pydmfet.qcwrap.pyscf_rks_ao import rks_ao
from pyscf import gto, scf,dft, ao2mo,cc
import numpy as np
from pyscf.tools import molden, cubegen
import copy,time
from pydmfet.tools.sym import h_lattice_sym_tab
import time

t0 = (time.clock(), time.time())


bas = 'lanl2dz'
ecp = 'lanl2dz'
temp = 0.005


mol = gto.Mole()
mol.atom = open('Cu2.xyz').read()
mol.basis = bas
mol.ecp = ecp
mol.charge = -2
mol.build(max_memory = 16000, verbose=4)

mf = rks_ao(mol,smear_sigma = temp)
mf.xc = 'pbe,pbe'
mf.max_cycle = 50
mf.scf(dm0=None)


natoms = mol.natm
impAtom = np.zeros([natoms], dtype=int)
for i in range(2):
    impAtom[i] = 1

ghost_frag = 1-impAtom
ghost_env = 1-ghost_frag

mol_frag = gto.Mole()
mol_frag.atom = tools.add_ghost(mol.atom, ghost_frag)
mol_frag.basis = bas
mol_frag.ecp = ecp
mol_frag.charge = 4
mol_frag.build(max_memory = 16000,verbose = 4)

mol_env = gto.Mole()
mol_env.atom = tools.add_ghost(mol.atom, ghost_env)
mol_env.basis =  bas
mol_env.ecp = ecp
mol_env.charge = -6
mol_env.build(max_memory = 16000,verbose = 4)

Ne_frag = 34
Ne_env = 48

boundary_atoms = None
boundary_atoms2 =None

umat = None
umat = np.load("umat.npy")

params = oep.OEPparams(algorithm = 'leastsq', opt_method = 'L-BFGS-B', diffP_tol=1e-3, outer_maxit = 20)
params.options['ftol'] = 1e-10
params.options['gtol'] = 1e-3
params.options['maxiter'] = 200
params.options['svd_thresh'] = 1e-2

theDMFET = dfet.DFET(mf, mol_frag, mol_env,Ne_frag,Ne_env,\
                     boundary_atoms=boundary_atoms, boundary_atoms2=boundary_atoms2,umat = umat,\
                     oep_params=params, smear_sigma=temp, ecw_method = 'hf',mf_method = mf.xc, plot_dens=True,\
                     scf_max_cycle = 200)

umat = theDMFET.embedding_potential()

t1 = tools.timer("total calc time", t0)
