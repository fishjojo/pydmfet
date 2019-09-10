import numpy as np
from pydmfet import tools

def ObjFunc_WuYang(x, v2m, sym_tab, scf_solver, P_ref, dim, use_suborb, nonscf, scf_args_frag, scf_args_env, calc_hess=False):

    umat = v2m(x, dim, True, sym_tab)
    print ("|umat| = ", np.linalg.norm(umat))

    scf_args_frag.update({'vext_1e':umat})
    scf_args_env.update({'vext_1e':umat})

    mf_frag = scf_solver(use_suborb, nonscf=nonscf, **scf_args_frag)
    mf_frag.kernel()
    #P_frag = mf_frag.make_rdm1()
    #E_frag = mf_frag.energy_elec(P_frag)[0]
    P_frag = mf_frag.rdm1
    E_frag = mf_frag.elec_energy

    mf_env  = scf_solver(use_suborb, nonscf=nonscf, **scf_args_env)
    mf_env.kernel()
    #P_env = mf_env.make_rdm1()
    #E_env = mf_env.energy_elec(P_env)[0]
    P_env = mf_env.rdm1
    E_env = mf_env.elec_energy


    P_diff = P_frag + P_env - P_ref
    W = E_frag + E_env - np.trace(np.dot(P_ref,umat))

    grad = v2m(P_diff, dim, False, sym_tab)
    grad *= -1.0

    #l2_f, l2_g = l2_reg(x, dim, v2m, sym_tab, l2_lambda)

    f = -W #+ l2_f
    #grad = grad + l2_g

    print ('-W = ', f)
    print ("2-norm (grad),       max(grad):" )
    print (np.linalg.norm(grad), ", ", np.amax(np.absolute(grad)))

    '''
    #test grad
    eps = 1e-5
    size = dim*(dim+1)//2
    step = np.zeros([size])
    step[0] = eps
    x_p = x + step
    x_m = x - step

    umat_p = v2m(x_p, dim, True, None)
    scf_args_frag.update({'vext_1e':umat_p})
    mf_frag_p = scf_solver(use_suborb, nonscf=nonscf, **scf_args_frag)
    mf_frag_p.kernel()
    E_frag_p = mf_frag_p.elec_energy

    umat_m = v2m(x_m, dim, True, None)
    scf_args_frag.update({'vext_1e':umat_m})
    mf_frag_m = scf_solver(use_suborb, nonscf=nonscf, **scf_args_frag)
    mf_frag_m.kernel()
    E_frag_m = mf_frag_m.elec_energy

    print ("grad_finite: ", (E_frag_p-E_frag_m)/2.0/eps)
    print ("grad_anal: ", P_frag[0,0])
    exit()
    #end test
    '''

    if calc_hess:
        from pydmfet.libcpp import oep_hess, symmtrize_hess, fd_hess
        size = dim*(dim+1)//2
        smear_sigma = getattr(mf_frag, 'smear_sigma', 0.0)
        Ne = int(np.sum(mf_frag.mo_occ))
        hess_frag = oep_hess(mf_frag.mo_coeff, mf_frag.mo_energy, size, dim, Ne//2, mf_frag.mo_occ, smear_sigma, sym_tab)
        #hess_frag = fd_hess(umat,v2m, scf_solver, use_suborb, nonscf, scf_args_frag)

        smear_sigma = getattr(mf_env, 'smear_sigma', 0.0)
        Ne = int(np.sum(mf_env.mo_occ))
        hess_env = oep_hess(mf_env.mo_coeff, mf_env.mo_energy, size, dim, Ne//2, mf_env.mo_occ, smear_sigma, sym_tab)
        #hess_env = fd_hess(umat,v2m, scf_solver, use_suborb, nonscf, scf_args_env)

        hess = -hess_frag - hess_env

        if sym_tab is not None:
            hess = symmtrize_hess(hess,sym_tab,size)

        return f,grad,hess


    return f, grad


def l2_reg(x, dim, v2m, sym_tab, l2_lambda):

    '''
    L2 regularization
    seems not working well
    '''

    target = 0.0

    u = v2m(x, dim, True, sym_tab)
    u_norm = np.linalg.norm(u)
    f = l2_lambda * ((u_norm - target)**2)

    fac = 2.0*l2_lambda*(u_norm-target)/u_norm
    g_mat = fac*u
    g = v2m(g_mat, dim, False, sym_tab)

    return f, g

