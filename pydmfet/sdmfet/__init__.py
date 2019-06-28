from pydmfet.sdmfet import dmfet

def DMFET(mf,mol_frag,mol_env, ints, cluster, impAtom, Ne_frag,**args):

    return dmfet.DMFET(mf,mol_frag,mol_env, ints, cluster, impAtom, Ne_frag, **args) 
