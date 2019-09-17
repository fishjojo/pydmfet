from pyscf.pbc import gto, scf, dft
import numpy
from pydmfet import locints

cell = gto.Cell()
# .a is a matrix for lattice vectors.
cell.a = '''
30.000   0.0000000   0.000
0.000  30.0000000   0.000
0.000   0.0000000  30.000'''

cell.atom =  '''C     15.628439779     15.000000000  15.00
		C     14.309410418     17.284679795  15.00
		C     11.671351697     17.284679795  15.00
		C     10.352322336     15.000000000  15.00
		C     11.671351697     12.715320205  15.00
		C     14.309410418     12.715320205  15.00
		H     17.675013988     15.000000000  15.00
		H     15.333642386     19.057243606  15.00
		H     10.647119729     19.057243606  15.00
		H      8.305748128     15.000000000  15.00
		H     10.647119729     10.942756393  15.00
		H     15.333642386     10.942756393  15.00'''

cell.unit = 'bohr'
cell.max_memory = 16000
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.verbose = 4
cell.build()

nao = cell.nao_nr()
print nao 
b = cell.reciprocal_vectors()
print b

nnkpts = numpy.array([[1, 1, 1, 0, 0],\
		      [1, 1, 0, 1, 0],\
		      [1, 1, 0, 0, 1]], numpy.int32)


mf = scf.RHF(cell)
ehf = mf.kernel()
print("HF energy (per unit cell) = %.17g" % ehf)

wf = wannier(mf, nnkpts, 'benzene.mmn')



