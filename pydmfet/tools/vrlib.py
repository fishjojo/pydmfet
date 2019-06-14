import numpy as np
from functools import reduce

def fock2onedm(fock, NOcc):

    eigenvals, eigenvecs = np.linalg.eigh(fock) # Does not guarantee sorted eigenvectors!
    idx = eigenvals.argsort()
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:,idx]
    OneDM = 2 * np.dot( eigenvecs[:,:NOcc] , eigenvecs[:,:NOcc].T )

    OneDM = 0.5*(OneDM + OneDM.T)

    print ("mo energy:")
    print (eigenvals)
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

    mat = np.zeros((dim,dim),dtype=np.double)
    iu = np.triu_indices(dim)
    mat[iu] = x
    mat = np.tril(mat.T,-1) + mat

    return mat

def mat2vec(mat, dim):

    size = dim*(dim+1)//2
    x = np.zeros(size,dtype=np.double)
    iu = np.triu_indices(dim)
    x = mat[iu]
    return x


#for H chain minimal basis divided at center
def mat2vec_hchain(mat,dim):

    size = dim//2
    k=1
    while (dim-k > 0):
        size += (dim - k)
        k += 2

    x = np.zeros(size,dtype=np.double)

    k=0
    for i in range(dim//2):
        for j in range(i,dim-i):
            x[k] = mat[i,j]
            k+=1

    return x

def vec2mat_hchain(x,dim):

    mat = np.zeros((dim,dim),dtype=np.double)
    k=0
    for i in range(dim//2):
        for j in range(i,dim-i):
            mat[i,j] = x[k]
            mat[dim-j-1,dim-i-1] = x[k]
            k+=1

    for j in range(dim-1):
        for i in range(j+1,dim):
            mat[i,j] = mat[j,i]

    return mat

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


def rank(s,tol=None):

    if tol is None:
        tol = s.max()*len(s)*np.finfo(s.dtype).eps

    rank = 0
    for i in range(len(s)):

        if s[i] > tol:
            rank += 1

    return rank

