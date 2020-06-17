from pydmfet import qcwrap

def init_dens_par(dfet, dim, use_suborb, mol1=None, mol2=None, umat=None, mf_method=None):

    if(umat is None): umat = dfet.umat
    if(mol1 is None): mol1 = dfet.mol_frag
    if(mol2 is None): mol2 = dfet.mol_env
    if(mf_method is None): mf_method = dfet.mf_method

    if use_suborb:
        ao2sub = dfet.ao2sub[:,:dim]
        ops = dfet.ops
        kin = ops["subKin"]
        vnuc_frag = ops["subVnuc1"]
        vnuc_env  = ops["subVnuc2"]
        vnuc_bound_frag = ops["subVnuc_bound1"]
        vnuc_bound_env  = ops["subVnuc_bound2"]
        coreJK = ops["subCoreJK"]
        tei = ops["subTEI"]
        oei_frag = kin + vnuc_frag + vnuc_bound_frag
        oei_env  = kin + vnuc_env  + vnuc_bound_env + coreJK

        scf_args_frag = {'mol':mol1, 'Ne':dfet.Ne_frag, 'Norb':dim, 'method':dfet.mf_method,
                         'vext_1e':umat, 'oei':oei_frag, 'tei':tei, 'dm0':None, 'coredm':0.0,
                         'ao2sub':ao2sub, 'smear_sigma':dfet.smear_sigma, 'max_cycle':dfet.scf_max_cycle}
        mf_frag = qcwrap.qc_scf(use_suborb, **scf_args_frag)
        mf_frag.kernel()
        FRAG_1RDM = mf_frag.rdm1

        scf_args_env  = {'mol':mol2, 'Ne':dfet.Ne_env, 'Norb':dim, 'method':dfet.mf_method,
                         'vext_1e':umat, 'oei':oei_env, 'tei':tei, 'dm0':None, 'coredm':0.0,
                         'ao2sub':ao2sub, 'smear_sigma':dfet.smear_sigma, 'max_cycle':dfet.scf_max_cycle}
        mf_env = qcwrap.qc_scf(use_suborb, **scf_args_env)
        mf_env.kernel()
        ENV_1RDM = mf_env.rdm1
    else:
        scf_args_frag = {'mol':mol1, 'xc_func':mf_method, 'dm0':None,
                         'vext_1e':umat, 'extra_oei':dfet.vnuc_bound_frag,
                         'smear_sigma':dfet.smear_sigma, 'max_cycle':dfet.scf_max_cycle}
        mf_frag = qcwrap.qc_scf(use_suborb, **scf_args_frag)
        mf_frag.kernel()
        FRAG_1RDM = mf_frag.rdm1

        scf_args_env  = {'mol':mol2, 'xc_func':mf_method, 'dm0':None,
                         'vext_1e':umat, 'extra_oei':dfet.vnuc_bound_env,
                         'smear_sigma':dfet.smear_sigma, 'max_cycle':dfet.scf_max_cycle}
        mf_env = qcwrap.qc_scf(use_suborb, **scf_args_env)
        mf_env.kernel()
        ENV_1RDM = mf_env.rdm1

    return FRAG_1RDM, ENV_1RDM
