from pydmfet import locints, sdmfet,oep,tools,dfet_ao
from pyscf import gto, scf, ao2mo
import numpy as np
from pyscf.tools import molden
import time
from pydmfet import dfet_ao
from pydmfet.dfet_ao import dfet
from read_umat import read_umat

bas ='ccpvdz'
temp = 0.01


mol = gto.Mole()
mol.atom = open('C3H6.xyz').read()
mol.basis = bas
mol.charge = 0
mol.build(max_memory = 4000, verbose=4)


#mf = scf.UKS(mol)
mf = dfet_ao.scf.EmbedSCF(mol, 0.0, smear_sigma = temp)
mf.xc = "pbe,pbe"
mf.max_cycle = 50

DMguess = None
mf.scf(dm0=DMguess)

#embedding calc
#myInts = locints.LocalIntegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )
#myInts.TI_OK = False

natoms = mol.natm
impAtom = np.zeros([natoms], dtype=int)
for i in range(5):
    impAtom[i] = 1


ghost_frag = 1-impAtom
ghost_env = 1-ghost_frag

mol_frag = gto.Mole()
mol_frag.atom = tools.add_ghost(mol.atom, ghost_frag)
mol_frag.basis = bas
mol_frag.charge = -1
mol_frag.build(max_memory = 4000,verbose = 4)

mol_env = gto.Mole()
mol_env.atom = tools.add_ghost(mol.atom, ghost_env)
mol_env.basis =  bas
mol_env.charge = 1
mol_env.build(max_memory = 4000,verbose = 4)

'''
aoslice = mol.aoslice_by_atom()
impurities = np.zeros([mol.nao_nr()], dtype = int)
for i in range(natoms):
    if(impAtom[i] == 1):
        impurities[aoslice[i,2]:aoslice[i,3]] = 1
'''

Ne_frag = 16
Ne_env = 8 
boundary_atoms = np.zeros([natoms])
boundary_atoms2 = np.zeros([natoms])

boundary_atoms[5] = 1
boundary_atoms2[5] = -1

#umat = np.loadtxt('umat.gz')

umat = read_umat(72,"C3H6.u")

params = oep.OEPparams(algorithm = '2011', opt_method = 'L-BFGS-B', \
                       ftol = 1e-8, gtol = 2e-5,diffP_tol=1e-4, outer_maxit = 20, maxit = 100,l2_lambda = 0.0, oep_print = 0)

theDMFET = dfet.DFET(mf, mol_frag, mol_env,Ne_frag,Ne_env,\
                     boundary_atoms=boundary_atoms, boundary_atoms2=boundary_atoms2,umat=umat,\
                     oep_params=params, smear_sigma=temp, ecw_method = 'hf',mf_method = mf.xc, plot_dens=True)

'''
theDMFET = sdmfet.DMFET( mf, mol_frag, mol_env,myInts,impurities, impAtom, Ne_frag, \
			 boundary_atoms=boundary_atoms, boundary_atoms2=boundary_atoms2,\
                         umat = None, dim_imp =None, dim_bath=None, dim_big =None, smear_sigma = temp, \
			 oep_params=params,ecw_method='ccsd', mf_method = mf.xc,\
                         use_umat_ao=False)
'''

umat = theDMFET.embedding_potential()
#energy = theDMFET.correction_energy()

