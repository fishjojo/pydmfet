from pydmfet import locints, sdmfet,oep,tools
from pyscf import gto, scf, ao2mo
import numpy as np
from pyscf.tools import molden
import time

bas = 'sto-6g'

t0 = (time.clock(), time.time())
mol = gto.Mole()
mol.atom = open('h10.xyz').read()
mol.basis = bas
mol.charge = 0
mol.build(max_memory = 4000, verbose=4)

mf = scf.RHF(mol)
mf.max_cycle = 100

DMguess = None
mf.scf(dm0=DMguess)

t1 = tools.timer("full scf", t0)

myInts = locints.LocalIntegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )
myInts.loc_molden( 'ao2loc.molden' )
myInts.TI_OK = False

t2 = tools.timer("localize orbitals", t1)

natoms = mol.natm
impAtom = np.zeros([natoms], dtype=int)
for i in range(2):
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


aoslice = mol.aoslice_by_atom()
impurities = np.zeros([mol.nao_nr()], dtype = int)
for i in range(natoms):
    if(impAtom[i] == 1):
	impurities[aoslice[i,2]:aoslice[i,3]] = 1

Ne_frag = 2
#boundary_atoms = np.zeros([natoms])
#boundary_atoms[6:12]=1.0
#boundary_atoms2 = np.zeros([natoms])
#boundary_atoms2[:6] = -1.0
boundary_atoms=None
boundary_atoms2=None

nbas =  mol.nao_nr()
params = oep.OEPparams(algorithm = '2011', opt_method = 'L-BFGS-B', \
                       ftol = 1e-12, gtol = 1e-6,diffP_tol=1e-6, outer_maxit = 200, maxit = 200,l2_lambda = 0.0, oep_print = 0)
theDMFET = sdmfet.DMFET( mf, mol_frag, mol_env,myInts,impurities, impAtom, Ne_frag, boundary_atoms=boundary_atoms, boundary_atoms2=boundary_atoms2,\
                         dim_imp =2, dim_bath=8, dim_big =None, smear_sigma = 0.00, oep_params=params,ecw_method='hf', mf_method = 'hf',plot_dens=True)

umat = theDMFET.embedding_potential()

t3 = tools.timer("sdmfet", t2)
t4 = tools.timer("total calc", t0)
