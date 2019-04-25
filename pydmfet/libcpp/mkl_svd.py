import numpy as np
import time
from pydmfet import tools

libsvd = np.ctypeslib.load_library('libsvd', os.path.dirname(__file__))


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
        print 'mkl_svd info = ',info[0]
        raise Exception("mkl_svd failed!")

    return (U,sigma,VT)

