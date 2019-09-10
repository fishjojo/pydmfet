#import ray
import copy
import numpy as np
from multiprocessing import Pool

def fd_hess(umat, v2m, scf_solver, use_suborb, nonscf, scf_args,delta=1e-5):

    dim = umat.shape[0]
    size = dim*(dim+1)//2

    hess = np.empty([size,size],dtype=float)

#    ray.init()

    '''
    index = 0
    for mu in range(dim):
        for nu in range(mu,dim):
            rdm1 = scf_calls(mu,nu,umat,delta,scf_solver, use_suborb, nonscf, scf_args)
            #rdm1 = ray.get(rdm1_id)
            hess[:,index] = v2m(rdm1, dim, False, None)
            index += 1
    '''

    munu_list = []
    for mu in range(dim):
        for nu in range(mu,dim):
            tp = (mu,nu,)
            munu_list.append(tp)

    pool = Pool()
    results = [pool.apply_async(scf_calls, (munu_list[i], umat, delta, scf_solver, use_suborb, nonscf, scf_args)) for i in range(len(munu_list))]
    pool.close()
    pool.join()
    for i in range(len(munu_list)):
        rdm1 = results[i].get()
        hess[:,i] = v2m(rdm1, dim, False, None)

    return hess


#@ray.remote
def scf_calls(munu, umat, delta, scf_solver, use_suborb, nonscf, _scf_args):

    mu = munu[0]
    nu = munu[1]

    u_p = umat.copy()
    u_m = umat.copy()
    u_p[mu,nu] += delta
    u_m[mu,nu] -= delta
    if mu != nu:
        u_p[nu,mu] += delta
        u_m[nu,mu] -= delta

    scf_args = copy.copy(_scf_args)

    scf_args.update({'vext_1e':u_p})
    mf_p = scf_solver(use_suborb, nonscf=nonscf, **scf_args)
    mf_p.verbose = 0
    mf_p.kernel()
    P_p = mf_p.rdm1

    scf_args.update({'vext_1e':u_m})
    mf_m = scf_solver(use_suborb, nonscf=nonscf, **scf_args)
    mf_m.verbose = 0
    mf_m.kernel()
    P_m = mf_m.rdm1

    P_grad = 0.5/delta*(P_p - P_m)

    return P_grad
