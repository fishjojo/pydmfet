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
    mf_frag.init_guess = 'hcore'
    mf_frag.kernel()
    #P_frag = mf_frag.make_rdm1()
    #E_frag = mf_frag.energy_elec(P_frag)[0]
    P_frag = mf_frag.rdm1
    E_frag = mf_frag.elec_energy

    mf_env  = scf_solver(use_suborb, nonscf=nonscf, **scf_args_env)
    mf_env.init_guess = 'hcore'
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

    grad = hess_frag + hess_env

    if sym_tab is not None:
        grad = symmtrize_hess(grad,sym_tab,size)

    grad = np.dot(f,grad)
    #print('delP*X')
    #print(grad)

    f = v2m(P_diff, dim, False, None)
    f = 0.5*np.dot(f,f)

    return f, grad

