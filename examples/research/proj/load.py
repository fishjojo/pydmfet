from pyscf.tools import molden


mol, mo_energy, mo_coeff, mo_occ, irrep_labels, spins = molden.load("mo_lo.molden")


print (mo_coeff.shape)
