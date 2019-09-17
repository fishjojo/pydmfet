from pyscf import lib
from pyscf.pbc import gto, scf, dft
from pyscf.tools import molden
import numpy
from pydmfet import locints

cell = gto.Cell()
# .a is a matrix for lattice vectors.
cell.a = '''
3.5668  0       0
0       3.5668  0
0       0       3.5668'''
cell.atom = '''C     0.      0.      0.    
              C     0.8917  0.8917  0.8917
              C     1.7834  1.7834  0.    
              C     2.6751  2.6751  0.8917
              C     1.7834  0.      1.7834
              C     2.6751  0.8917  2.6751
              C     0.      1.7834  1.7834
              C     0.8917  2.6751  2.6751'''

cell.max_memory = 16000
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.verbose = 5
cell.build()

nao = cell.nao_nr()
print nao 
b = cell.reciprocal_vectors()
print b

nnkpts = numpy.array([[[1, 1, 1, 0, 0],\
		      [1, 1, 0, 1, 0],\
		      [1, 1, 0, 0, 1]]], numpy.int32)


mf = scf.RHF(cell,exxdiv=None)
ehf = mf.kernel()
print("HF energy (per unit cell) = %.17g" % ehf)


molden.from_mo(cell, 'diamond.molden', mf.mo_coeff)


wf = locints.wannier2(mf, [1,1,1], 'diamond')
wf.kernel()



#bands = numpy.arange(0,32)
#wf = locints.wannier(mf, nnkpts, 'diamond',bands, has_proj_data=False)

#exit()
#umat =  locints.read_umat(wf,'diamond_u.mat')
umat = wf.u_mat
#print umat.shape

ints = locints.LocInts_wf(mf,umat)

#wf_coeff = numpy.dot(mf.mo_coeff,umat[0].real)
#molden.from_mo(cell, 'diamond_wf.molden', wf_coeff)

