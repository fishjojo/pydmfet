from pydmfet import locints, sdmfet
from pyscf import gto, scf, ao2mo
import numpy as np
from pyscf.tools import molden

mol = gto.Mole()
mol.atom = open('FHF_4H2O.xyz').read()
mol.basis = 'ccpvdz'
mol.charge = -1
mol.build(verbose=0)
mol.verbose = 5

mf = scf.RHF(mol)
mf.max_cycle = 1000

DMguess = None
mf.scf(dm0=DMguess)



myInts = locints.LocalIntegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )
#myInts.molden( 'hydrogen-loc.molden' )
myInts.TI_OK = False # Only s functions

exit()

theDMFET = sdmfet.DMFET( myInts,impurities, impAtom, Ne_frag)
umat = theDMFET.embedding_potential()

