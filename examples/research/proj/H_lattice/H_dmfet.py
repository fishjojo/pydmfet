from pydmfet import proj_ao, tools, oep
from pydmfet.dfet_ao import dfet
from pydmfet.qcwrap.pyscf_rks_ao import rks_ao
from pyscf import gto,scf
import numpy as np
from pyscf.tools import molden
from pyscf import lo
from pyscf.lo import iao,orth
from functools import reduce
from read_umat import read_umat
import math
from pydmfet.tools.sym import h_lattice_sym_tab

bas ='sto-6g'
temp = 0.005

mol = gto.Mole()
mol.atom = open('H.xyz').read()
mol.basis = bas
mol.charge = 0
mol.build(max_memory = 4000, verbose=4)


#mf = scf.RKS(mol)
mf = rks_ao(mol,smear_sigma = temp)
mf.xc = "pbe,pbe"
mf.max_cycle = 50

DMguess = None
mf.scf(dm0=DMguess)


natoms = mol.natm
impAtom = np.zeros([natoms], dtype=int)
for i in range(6):
    impAtom[i] = 1


ghost_frag = 1-impAtom
ghost_env = 1-ghost_frag

mol_frag = gto.Mole()
mol_frag.atom = tools.add_ghost(mol.atom, ghost_frag)
mol_frag.basis = bas
mol_frag.charge = 0
mol_frag.build(max_memory = 4000,verbose = 4)

mol_env = gto.Mole()
mol_env.atom = tools.add_ghost(mol.atom, ghost_env)
mol_env.basis =  bas
mol_env.charge = 0
mol_env.build(max_memory = 4000,verbose = 4)

Ne_frag = 6
Ne_env = 6
#boundary_atoms = np.zeros([natoms])
#boundary_atoms2 = np.zeros([natoms])
boundary_atoms = None
boundary_atoms2 = None

atm_ind = np.zeros([3,4],dtype=int)
val = 0
for i in range(4):
    for j in range(3):
        atm_ind[j,i] = val
        val+=1
sym_tab = h_lattice_sym_tab(atm_ind)

umat = None
umat = np.load('umat.npy')

params = oep.OEPparams(algorithm = 'split-leastsq', opt_method = 'L-BFGS-B',diffP_tol=1e-3, outer_maxit = 1)
params.options['ftol'] = 1e-9
params.options['gtol'] = 1e-3
params.options['maxiter'] = 50
params.options['svd_thresh'] = 1e-4

theDMFET = dfet.DFET(mf, mol_frag, mol_env,Ne_frag,Ne_env,\
                     boundary_atoms=boundary_atoms, boundary_atoms2=boundary_atoms2,umat=umat,\
                     oep_params=params, smear_sigma=temp, ecw_method = 'hf',mf_method = mf.xc, plot_dens=True)
theDMFET.sym_tab = sym_tab 

umat = theDMFET.embedding_potential()
