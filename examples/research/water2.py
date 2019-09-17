from pydmfet import locints, sdmfet,oep,tools
from pyscf import gto, scf, ao2mo
import numpy as np
from pyscf.tools import molden
import time
from pydmfet import dfet_ao
from pydmfet.dfet_ao import dfet

#bas = {"H":'ccpvdz',"O":'stuttgartdz'}
#ecp = {"O":'stuttgartdz'}

bas ='ccpvdz'

mol = gto.Mole()
mol.atom = open('water2.xyz').read()
mol.basis = bas
#mol.ecp = ecp
mol.charge = 0
mol.build(max_memory = 16000, verbose=4)

#print(mol.intor_symmetric('ECPscalar'))

mf = scf.RHF(mol)
#mf.xc = "pbe,pbe"
mf.max_cycle = 50

DMguess = None
mf.scf(dm0=DMguess)


#with open( 'water2_mo.molden', 'w' ) as thefile:
#    molden.header(mf.mol, thefile)
#    molden.orbital_coeff(mf.mol, thefile, mf.mo_coeff)


myInts = locints.LocalIntegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )
myInts.TI_OK = False


natoms = mol.natm
impAtom = np.zeros([natoms], dtype=int)
for i in range(3):
    impAtom[i] = 1


ghost_frag = 1-impAtom
ghost_env = 1-ghost_frag

mol_frag = gto.Mole()
mol_frag.atom = tools.add_ghost(mol.atom, ghost_frag)
mol_frag.basis = bas
#mol_frag.ecp = ecp
mol_frag.build(max_memory = 8000,verbose = 4)

mol_env = gto.Mole()
mol_env.atom = tools.add_ghost(mol.atom, ghost_env)
mol_env.basis =  bas
#mol_env.ecp = ecp
mol_env.build(max_memory = 8000,verbose = 4)


aoslice = mol.aoslice_by_atom()
impurities = np.zeros([mol.nao_nr()], dtype = int)
for i in range(natoms):
    if(impAtom[i] == 1):
	impurities[aoslice[i,2]:aoslice[i,3]] = 1

Ne_frag = 10
boundary_atoms = np.zeros([natoms])
boundary_atoms2 = np.zeros([natoms])

nbas =  mol.nao_nr()
params = oep.OEPparams(algorithm = 'split', opt_method = 'L-BFGS-B', \
                       ftol = 1e-12, gtol = 1.0e-6,diffP_tol= 1.0e-6, outer_maxit = 200, maxit = 200,l2_lambda = 0.0, oep_print = 0)

#theDMFET = dfet.DFET(mf, mol_frag, mol_env,\
#                     boundary_atoms=boundary_atoms, boundary_atoms2=boundary_atoms2,\
#                     oep_params=params, smear_sigma=0.00, ecw_method = 'hf',mf_method = mf.xc, plot_dens=True)

#umat = theDMFET.embedding_potential()

#params.algorithm='2011'
#params.gtol=1.e-4
#params.diffP_tol=1.e-4

theDMFET = sdmfet.DMFET( mf, mol_frag, mol_env,myInts,impurities, impAtom, Ne_frag, boundary_atoms=boundary_atoms, boundary_atoms2=boundary_atoms2,\
                         umat = None, dim_imp =10, dim_bath=10, dim_big =20, smear_sigma = 0.00, oep_params=params,ecw_method='ccsd', mf_method = 'hf',\
			 use_umat_ao=False)

umat = theDMFET.embedding_potential()

e=theDMFET.correction_energy()
print 'corr energy = ',e

#Myoep = dfet_ao.oep.OEPao(theDMFET,params)
#umat = Myoep.kernel() 

'''
params.algorithm='2011'
params.maxit=1
params.outer_maxit=1
theDMFET = dfet.DFET(mf, mol_frag, mol_env,\
                     boundary_atoms=boundary_atoms, boundary_atoms2=boundary_atoms2,\
                     umat = umat, oep_params=params, smear_sigma=0.00, ecw_method = 'hf',mf_method = mf.xc, plot_dens=True)

umat = theDMFET.embedding_potential()
'''
