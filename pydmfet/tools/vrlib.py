import numpy as np

def fock2onedm(fock, NOcc):

    eigenvals, eigenvecs = np.linalg.eigh(fock) # Does not guarantee sorted eigenvectors!
    idx = eigenvals.argsort()
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:,idx]
    OneDM = 2 * np.dot( eigenvecs[:,:NOcc] , eigenvecs[:,:NOcc].T )

    return OneDM


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
