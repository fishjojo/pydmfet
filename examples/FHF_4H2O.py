from pydmfet import locints, sdmfet
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



myInts = locints.LocalIntegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )
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

charge = [-1, 0]
spin   = [ 0, 0]
theDMFET = sdmfet.DMFET( myInts,impurities, impAtom, Ne_frag, charge, spin, 1e-3)
umat = theDMFET.embedding_potential()

