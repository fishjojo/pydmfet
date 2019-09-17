from pydmfet import locints, sdmfet,oep,tools,dfet_ao,proj_ao
from pydmfet.qcwrap.pyscf_rks_ao import rks_ao
from pyscf import gto, scf, ao2mo
import numpy as np
from pyscf.tools import molden
import time
from pydmfet import dfet_ao
from pydmfet.dfet_ao import dfet

bas ='ccpvdz'
temp = 0.005


mol = gto.Mole()
mol.atom = open('C20.xyz').read()
mol.basis = bas
mol.charge = 0
mol.build(max_memory = 4000, verbose=4)


#mf = scf.RHF(mol)
mf = rks_ao(mol,smear_sigma = temp)
mf.xc = "hf"
mf.max_cycle = 50

DMguess = None
mf.scf(dm0=DMguess)

natoms = mol.natm
impAtom = np.zeros([natoms], dtype=int)
for i in range(10):
    impAtom[i] = 1

ghost_frag = 1-impAtom
ghost_env = 1-ghost_frag

mol_frag = gto.Mole()
mol_frag.atom = tools.add_ghost(mol.atom, ghost_frag)
mol_frag.basis = bas
mol_frag.build(max_memory = 4000,verbose = 4)

mol_env = gto.Mole()
mol_env.atom = tools.add_ghost(mol.atom, ghost_env)
mol_env.basis =  bas
mol_env.build(max_memory = 4000,verbose = 4)

Ne_frag = 20
Ne_env = 20

boundary_atoms = None
boundary_atoms2 = None
boundary_atoms = np.zeros([natoms])
boundary_atoms2 =np.zeros([natoms])

boundary_atoms[10] = 1
boundary_atoms[11] = 1
boundary_atoms2[0] = 1
boundary_atoms2[1] = 1

umat = None
#umat = np.load("umat.npy")

params = oep.OEPparams(algorithm = '2011', opt_method = 'L-BFGS-B', diffP_tol=1e-3, outer_maxit = 10)
params.options['ftol'] = 1e-9
params.options['gtol'] = 1e-3
params.options['maxiter'] = 50
params.options['svd_thresh'] = 1e-5

theDMFET = dfet.DFET(mf, mol_frag, mol_env,Ne_frag,Ne_env,\
                     boundary_atoms=boundary_atoms, boundary_atoms2=boundary_atoms2,umat = umat,\
                     oep_params=params, smear_sigma=temp, ecw_method = 'hf',mf_method = mf.xc, plot_dens=True)

umat = theDMFET.embedding_potential()

