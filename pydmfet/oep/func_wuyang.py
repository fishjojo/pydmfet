import numpy as np

def ObjFunc_WuYang(x, v2m, sym_tab, scf_solver, P_ref, dim, use_suborb, scf_args_frag, scf_args_env):

    umat = v2m(x, dim, True, sym_tab)
    print ("|umat| = ", np.linalg.norm(umat))

    scf_args_frag.update({'vext_1e':umat})
    scf_args_env.update({'vext_1e':umat})

    mf_frag = scf_solver(use_suborb, **scf_args_frag)
    mf_frag.kernel()
    P_frag = mf_frag.make_rdm1()
    E_frag = mf_frag.energy_elec(P_frag)[0]

    mf_env  = scf_solver(use_suborb, **scf_args_env)
    mf_env.kernel()
    P_env = mf_env.make_rdm1()
    E_env = mf_env.energy_elec(P_env)[0]


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

