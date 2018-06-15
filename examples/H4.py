from pyscf import lib
from pyscf.pbc import gto, scf, dft
from pyscf.tools import molden
import numpy as np
from pydmfet import locints, sdmfet,oep

cell = gto.Cell()
# .a is a matrix for lattice vectors.
cell.a = '''
22.0  0       0
0       10.0  0
0       0       10.0'''
cell.atom = ''' H  9.39   5.000  5.000
		H 10.13   5.000  5.000
		H 11.87   5.000  5.000
		H 12.61   5.000  5.000'''


cell.max_memory = 16000
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.verbose = 5
cell.build()

nao = cell.nao_nr()
print nao 
b = cell.reciprocal_vectors()
print b

mf = scf.RHF(cell,exxdiv=None)
ehf = mf.kernel()
print("HF energy (per unit cell) = %.17g" % ehf)
#molden.from_mo(cell, 'diamond.molden', mf.mo_coeff)
exit()
wf = locints.wannier2(mf, [1,1,1], 'H4')
wf.kernel()
umat = wf.u_mat

ints = locints.LocInts_wf(mf,umat)


#DMFET block

mol_frag = mol_env = None   #useless
natoms = cell.natm
impAtom = np.zeros([natoms], dtype=int)
for i in range(2):
    impAtom[i] = 1

aoslice = cell.aoslice_by_atom()
impurities = np.zeros([cell.nao_nr()], dtype = int)
for i in range(natoms):
    if(impAtom[i] == 1):
        impurities[aoslice[i,2]:aoslice[i,3]] = 1

Ne_frag = 2
boundary_atoms = np.zeros([natoms], dtype=int)
#boundary_atoms[5] = 1
#boundary_atoms[8]=1
#boundary_atoms[9]=1
#boundary_atoms[12]=1
boundary_atoms =  None

mol_frag = mol_env = None

params = oep.OEPparams(algorithm = '2011', ftol = 1e-11, gtol = 1e-6,diffP_tol=1e-6, outer_maxit = 100, maxit = 100,l2_lambda = 0.0, oep_print = 0)
theDMFET = sdmfet.DMFET(mf, mol_frag, mol_env, ints,impurities, impAtom, Ne_frag, boundary_atoms=boundary_atoms, dim_bath = 2, oep_params=params, ecw_method = 'ccsd',mf_method = 'hf')

umat = theDMFET.embedding_potential()
e_corr = theDMFET.correction_energy()

e_tot = ehf + e_corr
print 'e_hf = ', ehf
print 'e_tot = ', e_tot




