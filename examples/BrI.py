from pydmfet import locints, sdmfet,oep,tools
from pyscf import gto, scf, ao2mo,lo, cc
import numpy as np
from pyscf.tools import molden
import time
from pydmfet import dfet_ao
from pydmfet.dfet_ao import dfet

bas = "lanl2dz"
ecp = {"Br":"lanl2dz","I":"lanl2dz"}

#bas = "ccpvdz"

mol = gto.Mole()
mol.atom = open('BrI.xyz').read()
mol.basis = bas
mol.ecp = ecp
mol.charge = -1
mol.build(max_memory = 4000, verbose=4)

#print(mol.intor_symmetric('ECPscalar'))

mf = scf.RKS(mol)
mf.xc = "b3lyp"
mf.max_cycle = 50

DMguess = None
mf.scf(dm0=DMguess)

'''
hf=scf.RHF(mol)
hf.scf()
mycc = cc.CCSD(hf)
mycc.kernel()
et = mycc.ccsd_t()
energy = hf.energy_tot() + mycc.e_corr + et
print "total system ccsd(t) energy: ", energy
'''

'''
nocc = mf.mol.nelectron // 2
mo_pipek = lo.pipek.PM(mol).kernel(mf.mo_coeff[:,:nocc], verbose=4)


with open( 'BrI_mo.molden', 'w' ) as thefile:
    molden.header(mf.mol, thefile)
    molden.orbital_coeff(mf.mol, thefile, mo_pipek)

exit()
'''

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
#mol_frag=gto.Mole()
#mol_frag.atom = open('BrI.xyz').read()
mol_frag.basis = bas
mol_frag.ecp = ecp
mol_frag.charge = -3
mol_frag.build(max_memory = 4000,verbose = 4)

mol_env = gto.Mole()
mol_env.atom = tools.add_ghost(mol.atom, ghost_env)
#mol_env=gto.Mole()
#mol_env.atom = open('BrI.xyz').read()
mol_env.basis =  bas
mol_env.ecp = ecp
mol_env.charge = 2
mol_env.build(max_memory = 4000,verbose = 4)


aoslice = mol.aoslice_by_atom()
impurities = np.zeros([mol.nao_nr()], dtype = int)
for i in range(natoms):
    if(impAtom[i] == 1):
	impurities[aoslice[i,2]:aoslice[i,3]] = 1

Ne_frag = 12
boundary_atoms = np.zeros([natoms])
boundary_atoms2 = np.zeros([natoms])

boundary_atoms[4] = 7
boundary_atoms[5] = 7
boundary_atoms2[0] = 6
boundary_atoms2[1:3] = 1
#boundary_atoms2[4] = -1
#boundary_atoms2[5] = -1

nbas =  mol.nao_nr()
params = oep.OEPparams(algorithm = 'split', opt_method = 'L-BFGS-B', \
                       ftol = 1e-12, gtol = 1.0e-5,diffP_tol= 1.0e-5, outer_maxit = 0, maxit = 200,l2_lambda = 0.0, oep_print = 0)

#theDMFET = dfet.DFET(mf, mol_frag, mol_env,\
#                     boundary_atoms=boundary_atoms, boundary_atoms2=boundary_atoms2,\
#                     oep_params=params, smear_sigma=0.00, ecw_method = 'hf',mf_method = mf.xc, plot_dens=True)

#umat = theDMFET.embedding_potential()

#params.algorithm='2011'
#params.gtol=1.e-4
#params.diffP_tol=1.e-4

theDMFET = sdmfet.DMFET( mf, mol_frag, mol_env,myInts,impurities, impAtom, Ne_frag, boundary_atoms=boundary_atoms, boundary_atoms2=boundary_atoms2,\
                         umat = None, dim_imp =nbas, dim_bath=nbas, dim_big =None, smear_sigma = 0.00, oep_params=params,ecw_method='ccsd(t)', mf_method = mf.xc,\
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
