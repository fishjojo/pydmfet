from pydmfet import locints, sdmfet,oep
from pyscf import gto, scf, ao2mo
import numpy as np
from pyscf.tools import molden

mol = gto.Mole()
mol.atom = open('FHF_4H2O.xyz').read()
mol.basis = 'ccpvdz'
mol.charge = -1
mol.build(verbose=0)
mol.verbose = 3

mf = scf.RHF(mol)
mf.max_cycle = 1000

DMguess = None
mf.scf(dm0=DMguess)

print mol.nao_nr()

myInts = locints.LocalIntegrals( mf, range( mol.nao_nr() ), 'iao' )
myInts.molden( 'iao.molden' )
exit()
#myInts.molden( 'hydrogen-loc.molden' )
myInts.TI_OK = False # Only s functions

natoms = mol.natm
impAtom = np.zeros([natoms], dtype=int)
for i in range(3):
    impAtom[i] = 1

aoslice = mol.aoslice_by_atom()
impurities = np.zeros([mol.nao_nr()], dtype = int)
for i in range(natoms):
    if(impAtom[i] == 1):
	impurities[aoslice[i,2]:aoslice[i,3]] = 1

Ne_frag = 20
boundary_atoms = np.zeros([natoms], dtype=int)
boundary_atoms[5] = 1
boundary_atoms[8]=1
#boundary_atoms[9]=1
#boundary_atoms[12]=1

params = oep.OEPparams(algorithm = 'split', ftol = 1e-10, gtol = 1e-6,diffP_tol=1e-6, outer_maxit = 100, maxit = 100,oep_print = 3)
theDMFET = sdmfet.DMFET( myInts,impurities, impAtom, Ne_frag, boundary_atoms=boundary_atoms, sub_threshold = 1e-3, oep_params=params)
umat = theDMFET.embedding_potential()

