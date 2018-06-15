from pydmfet import locints, sdmfet,oep,tools
from pyscf import gto, scf, ao2mo
import numpy as np
from pyscf.tools import molden
import time

t0 = (time.clock(), time.time())
mol = gto.Mole()
mol.atom = '''
H   6.29   5.63   4.0
H   6.29   6.37   4.0
H   5.92    6.0    2.0
H   6.66    6.0    2.0'''

mol.basis = 'ccpvdz'
mol.charge = 0
mol.build(max_memory = 8000, verbose=0)
mol.verbose = 5

mf = scf.RHF(mol)
mf.max_cycle = 100

DMguess = None
mf.scf(dm0=DMguess)

t1 = tools.timer("full scf", t0)

myInts = locints.LocalIntegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )
molden.from_mo(mol,'h2_mo.molden',mf.mo_coeff)
myInts.loc_molden( 'h2_boys.molden' )
myInts.TI_OK = False

t2 = tools.timer("localize orbitals", t1)
exit()
natoms = mol.natm
impAtom = np.zeros([natoms], dtype=int)
impAtom[0] = 1
impAtom[1] = 1
impAtom[2] = 1
impAtom[4] = 1
impAtom[5] = 1
impAtom[11] = 1


aoslice = mol.aoslice_by_atom()
impurities = np.zeros([mol.nao_nr()], dtype = int)
for i in range(natoms):
    if(impAtom[i] == 1):
	impurities[aoslice[i,2]:aoslice[i,3]] = 1

Ne_frag = 24
boundary_atoms = np.zeros([natoms], dtype=int)
boundary_atoms[3]=1
boundary_atoms[6]=1
boundary_atoms[7]=1
boundary_atoms[13]=1
boundary_atoms[14]=1
boundary_atoms[17]=1

params = oep.OEPparams(algorithm = 'split', ftol = 1e-10, gtol = 1e-6,diffP_tol=1e-6, outer_maxit = 100, maxit = 100,oep_print = 0)
theDMFET = sdmfet.DMFET( myInts,impurities, impAtom, Ne_frag, boundary_atoms=boundary_atoms, sub_threshold = 1e-6, oep_params=params)
umat = theDMFET.embedding_potential()

t3 = tools.timer("sdmfet", t2)
t4 = tools.timer("total calc", t0)
