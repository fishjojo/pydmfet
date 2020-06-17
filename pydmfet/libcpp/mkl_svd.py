import os
import numpy as np
import time
from pydmfet import tools
import ctypes

libsvd = np.ctypeslib.load_library('liblinalg', os.path.dirname(__file__))


def mkl_svd(A, algorithm = 1):

    t0 = (time.clock(),time.time())

    m = A.shape[0]
    n = A.shape[1]

    U = np.ndarray((m,m), dtype=float, order='F')
    VT = np.ndarray((n,n), dtype=float, order='F')
    sigma = np.ndarray((min(m,n)), dtype=float, order='F')

    info = np.zeros((1),dtype = int)

    libsvd.mkl_svd(A.ctypes.data_as(ctypes.c_void_p),\
                   sigma.ctypes.data_as(ctypes.c_void_p),\
                   U.ctypes.data_as(ctypes.c_void_p),\
                   VT.ctypes.data_as(ctypes.c_void_p),\
                   ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(algorithm), \
                   info.ctypes.data_as(ctypes.c_void_p))

    t1 = tools.timer("mkl_svd", t0)
    if(info[0] != 0):
        print ('mkl_svd info = ',info[0])
        raise Exception("mkl_svd failed!")

    return (U,sigma,VT)


def invert_mat_sigular_thresh(mat,thresh):

    dim = mat.shape[0]
    u, sigma, vt = mkl_svd(mat)


    irank = 0
    D = np.zeros(dim,dtype=np.double)
    for i in range(dim):
        s = sigma[i]
        if(s > thresh):
            irank += 1
            D[i] = 1.0/s
        else:
            break
#        D[i] = s/(s*s+thresh*thresh)

    print (" rank = ", irank)
    print (" singular value | inverse singular value")
    for i in range(4):
         print (" %.3e   |   %.3e" % (sigma[i], D[i]))

    print ("     .       |       .")
    print ("     .       |       .")
    print ("     .       |       .")
    print (" %.3e   |   %.3e" %(sigma[irank-2], D[irank-2]))
    print (" %.3e   |   %.3e" %(sigma[irank-1], D[irank-1]))
    print (" -------------------------")
    if(irank < dim):
        print (" %.3e   |   %.3e" %(sigma[irank], D[irank]))
    if(irank < dim-1):
        print (" %.3e   |   %.3e" %(sigma[irank+1], D[irank+1]))

    v = vt.T
    for i in range(irank):
        v[:,i] *= D[i]

    mat_invs = np.dot(v, u.T)

    return mat_invs

