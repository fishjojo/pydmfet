from pydmfet import locints, sdmfet, oep, tools, dfet_ao, proj_ao
from pydmfet.dfet_ao import dfet
from pydmfet.qcwrap.pyscf_rks_ao import rks_ao
from pyscf import gto, scf,dft, ao2mo,cc
import numpy as np
from pyscf.tools import molden, cubegen
import copy
from pydmfet.tools.sym import h_lattice_sym_tab

t0 = tools.time0()


bas = '6-31G*'
temp = 0.005


mol = gto.Mole()
mol.atom = open('trans_retinal_90_150.xyz').read()
mol.basis = bas
mol.charge = 1
mol.build(max_memory = 16000, verbose=4)

dm_guess=None
_, _, mo_coeff, mo_occ, _, _ = molden.load("PBE_MO.molden")
dm_guess = np.dot(mo_coeff*mo_occ, mo_coeff.T)

mf = rks_ao(mol,smear_sigma = temp)
mf.xc = 'pbe,pbe'
mf.max_cycle = 50
mf.scf(dm0=dm_guess)


with open( 'PBE_MO.molden', 'w' ) as thefile:
    molden.header(mf.mol, thefile)
    molden.orbital_coeff(mf.mol, thefile, mf.mo_coeff,occ = mf.mo_occ)

myInts = locints.LocalIntegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )
myInts.TI_OK = False

natoms = mol.natm
impAtom = np.zeros([natoms], dtype=int)
for i in range(4):
    impAtom[i] = 1


ghost_frag = 1-impAtom
ghost_env = 1-ghost_frag

mol_frag = gto.Mole()
mol_frag.atom = tools.add_ghost(mol.atom, ghost_frag)
mol_frag.basis = bas
mol_frag.charge = -2 
mol_frag.build(max_memory = 16000,verbose = 4)

mol_env = gto.Mole()
mol_env.atom = tools.add_ghost(mol.atom, ghost_env)
mol_env.basis =  bas
mol_env.charge = 3
mol_env.build(max_memory = 16000,verbose = 4)

boundary_atoms = np.zeros([natoms])
boundary_atoms2 = np.zeros([natoms])
boundary_atoms[5-1] = 1
boundary_atoms[6-1] = 1
boundary_atoms2[1-1] = -1
boundary_atoms2[2-1] = -1

Ne_frag = 16 
aoslice = mol.aoslice_by_atom()
impurities = np.zeros([mol.nao_nr()], dtype = int)
for i in range(natoms):
    if(impAtom[i] == 1):
        impurities[aoslice[i,2]:aoslice[i,3]] = 1


params = oep.OEPparams(algorithm = 'leastsq', opt_method = 'L-BFGS-B', diffP_tol=1e-3, outer_maxit = 20)
params.options['ftol'] = 1e-9
params.options['gtol'] = 1e-3
params.options['maxiter'] = 50
params.options['svd_thresh'] = 1e-2


theDMFET = sdmfet.DMFET( mf, mol_frag, mol_env,myInts,impurities, impAtom, Ne_frag, \
                         boundary_atoms=boundary_atoms, boundary_atoms2=boundary_atoms2,\
                         umat = None, dim_imp =None, dim_bath=None, dim_big =None, smear_sigma = temp, \
                         oep_params=params,ecw_method='ccsd', mf_method = mf.xc,\
                         use_umat_ao=False)

#umat = theDMFET.embedding_potential()
#energy = theDMFET.correction_energy()
t1 = tools.timer("total calc time",t0)
