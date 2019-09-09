import numpy as np
from pydmfet.libcpp import oep_hess, symmtrize_hess
from pydmfet import tools

def ObjFunc_LeastSq(x, v2m, sym_tab, scf_solver, P_ref, dim, use_suborb, nonscf, scf_args_frag, scf_args_env):

    umat = v2m(x, dim, True, sym_tab)
    print ("|umat| = ", np.linalg.norm(umat))
    #tools.MatPrint(umat, "umat")

    scf_args_frag.update({'vext_1e':umat})
    scf_args_env.update({'vext_1e':umat})

    mf_frag = scf_solver(use_suborb, nonscf=nonscf, **scf_args_frag)
    #mf_frag.init_guess = 'hcore'
    mf_frag.kernel()
    #P_frag = mf_frag.make_rdm1()
    #E_frag = mf_frag.energy_elec(P_frag)[0]
    P_frag = mf_frag.rdm1
    E_frag = mf_frag.elec_energy

    mf_env  = scf_solver(use_suborb, nonscf=nonscf, **scf_args_env)
    #mf_env.init_guess = 'hcore'
    mf_env.kernel()
    #P_env = mf_env.make_rdm1()
    #E_env = mf_env.energy_elec(P_env)[0]
    P_env = mf_env.rdm1
    E_env = mf_env.elec_energy


    P_diff = P_frag + P_env - P_ref
    f = v2m(P_diff, dim, False, sym_tab)

    print ("2-norm (grad),       max(grad):" )
    print (np.linalg.norm(f), ", ", np.amax(np.absolute(f)))

    size = dim*(dim+1)//2

    smear_sigma = getattr(mf_frag, 'smear_sigma', 0.0)
    Ne = int(np.sum(mf_frag.mo_occ))
    hess_frag = oep_hess(mf_frag.mo_coeff, mf_frag.mo_energy, size, dim, Ne//2, mf_frag.mo_occ, smear_sigma, sym_tab)

    smear_sigma = getattr(mf_env, 'smear_sigma', 0.0)
    Ne = int(np.sum(mf_env.mo_occ))
    hess_env = oep_hess(mf_env.mo_coeff, mf_env.mo_energy, size, dim, Ne//2, mf_env.mo_occ, smear_sigma, sym_tab)

    hess = hess_frag + hess_env

    '''
    #test hess
    eps = 1e-5
    step = np.zeros([size])
    step[0] = eps
    x_p = x + step
    x_m = x - step

    umat_p = v2m(x_p, dim, True, None)
    scf_args_frag.update({'vext_1e':umat_p})
    mf_frag_p = scf_solver(use_suborb, nonscf=nonscf, **scf_args_frag)
    mf_frag_p.init_guess = 'hcore'
    mf_frag_p.kernel()
    P_frag_p = mf_frag_p.rdm1

    umat_m = v2m(x_m, dim, True, None)
    scf_args_frag.update({'vext_1e':umat_m})
    mf_frag_m = scf_solver(use_suborb, nonscf=nonscf, **scf_args_frag)
    mf_frag_m.init_guess = 'hcore'
    mf_frag_m.kernel()
    P_frag_m = mf_frag_m.rdm1

    P_grad = 0.5/eps*(P_frag_p - P_frag_m)
    P_grad_anl = v2m(hess_frag[:,0], dim, True, None)
    tools.MatPrint(P_grad,"P_grad finite")
    tools.MatPrint(P_grad_anl,"P_grad anl")
    tools.MatPrint((P_grad_anl-P_grad)/P_grad,"P_grad anl-P_grad / P_grad")
    tools.MatPrint(P_grad*P_grad_anl,"P_grad * P_grad_anl")
    exit()
    #end test hess
    '''

    if sym_tab is not None:
        hess = symmtrize_hess(hess,sym_tab,size)

    grad = np.dot(f,hess)
    #print('delP*X')
    #print(grad)

    f = v2m(P_diff, dim, False, None)
    f = 0.5*np.dot(f,f)
    print("f = ", f)
    return f, grad
