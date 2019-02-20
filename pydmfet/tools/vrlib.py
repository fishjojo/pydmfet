import numpy as np

def fock2onedm(fock, NOcc):

    eigenvals, eigenvecs = np.linalg.eigh(fock) # Does not guarantee sorted eigenvectors!
    idx = eigenvals.argsort()
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:,idx]
    OneDM = 2 * np.dot( eigenvecs[:,:NOcc] , eigenvecs[:,:NOcc].T )

    OneDM = 0.5*(OneDM + OneDM.T)

    print "mo energy:"
    print eigenvals
    return (OneDM,eigenvecs)

def fock2mo(fock,NOcc):

    eigenvals, eigenvecs = np.linalg.eigh(fock)
    idx = eigenvals.argsort()
    mo_energy = eigenvals[idx]
    mo_coeff = eigenvecs[:,idx]
    mo_occ = np.zeros((fock.shape[0]))
    mo_occ[:NOcc] = 2.0

    return (mo_coeff, mo_energy, mo_occ)


def vec2mat(x, dim):

    mat = np.zeros((dim,dim),dtype=float)
    iu = np.triu_indices(dim)
    mat[iu] = x
    mat = np.tril(mat.T,-1) + mat

    return mat

def mat2vec(mat, dim):

    size = dim*(dim+1)/2
    x = np.zeros(size,dtype=float)
    iu = np.triu_indices(dim)
    x = mat[iu]
    return x


def dm_ao2loc(dm_ao, s, ao2loc):

    st = np.dot(s,ao2loc)
    dm_loc = reduce(np.dot, (st.T, dm_ao, st))

    return dm_loc

def dm_ao2sub(dm_ao, s, ao2sub):

    return dm_ao2loc(dm_ao, s, ao2sub)

def dm_loc2sub(dm_loc, loc2sub):

    dm_sub = reduce(np.dot, (loc2sub.T, dm_loc, loc2sub))
    return dm_sub

def dm_loc2ao(dm_loc, ao2loc):

    dm_ao = reduce(np.dot, (ao2loc, dm_loc, ao2loc.T))
    return dm_ao

def dm_sub2loc(dm_sub, loc2sub):

    return dm_loc2ao(dm_sub, loc2sub)

def dm_sub2ao(dm_sub, ao2sub):

    return dm_loc2ao(dm_sub, ao2sub)

#ao2sub = ao2loc*loc2sub
def mo_sub2ao(mo_coeff_sub, ao2sub):

    mo_coeff_ao = np.dot(ao2sub, mo_coeff_sub)
    return mo_coeff_ao


def op_loc2sub(op_loc, loc2sub):

    op_sub = reduce(np.dot, (loc2sub.T, op_loc, loc2sub))
    return op_sub

def op_ao2sub(op_ao, ao2sub):

    op_sub = reduce(np.dot, (ao2sub.T, op_ao, ao2sub))
    return op_sub

def op_sub2ao(op_sub, ao2sub):

    op_ao = reduce(np.dot, (ao2sub, op_sub, ao2sub.T))
    return op_ao

def rank(s,tol=None):

    if tol is None:
	tol = s.max()*len(s)*np.finfo(s.dtype).eps

    rank = 0
    for i in range(len(s)):

	if s[i] > tol:
	    rank += 1

    return rank

