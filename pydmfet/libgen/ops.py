import numpy as np
from pydmfet import tools


def build_locops(mol_frag, mol_env, ints, core1PDM_loc, Kcoeff = 1.0, Ne_core=0):

    t0 = tools.time0()

    locKin = ints.frag_kin_loc()

    locVnuc1 = ints.frag_vnuc_loc(mol_frag)
    locVnuc2 = ints.frag_vnuc_loc(mol_env)

#    locVnuc_bound1 = 0.0
#    locVnuc_bound2 = 0.0

    locCoreJK = 0.0
    if(Ne_core>0):
        locCoreJK = ints.coreJK_loc(core1PDM_loc, Kcoeff)

    locTEI = ints.tei_loc()

    ops = {"locKin":locKin}
    ops["locVnuc1"] = locVnuc1
    ops["locVnuc2"] = locVnuc2
#    ops["locVnuc_bound1"] = locVnuc_bound1
#    ops["locVnuc_bound2"] = locVnuc_bound2
    ops["locCoreJK"] = locCoreJK
    ops["locTEI"] = locTEI

    tools.timer("libgen.ops.build_locops",t0)

    return ops


def build_subops(impAtom, mol_frag, mol_env, boundary_atoms1, boundary_atoms2, ints, loc2sub, core1PDM_loc, dim, Kcoeff = 1.0, Ne_core=0):

    t0 = tools.time0()

    subKin = frag_kin_sub( ints, loc2sub, dim )
    #subVnuc1 = frag_vnuc_sub( ints, impAtom, loc2sub, dim)
    #subVnuc2 = frag_vnuc_sub( ints, 1-impAtom, loc2sub, dim )

    subVnuc1 = frag_vnuc_sub(mol_frag, ints, loc2sub, dim)
    subVnuc2 = frag_vnuc_sub(mol_env, ints, loc2sub, dim)

    subVnuc_bound1 = 0.0
    subVnuc_bound2 = 0.0
    if(boundary_atoms1 is not None):
        subVnuc_bound1 += ints.bound_vnuc_sub(boundary_atoms1, loc2sub, dim )
    if(boundary_atoms2 is not None):
        subVnuc_bound2 += ints.bound_vnuc_sub(boundary_atoms2, loc2sub, dim )


    #tools.MatPrint(subVnuc_bound1,"subVnuc_bound1")
    #tools.MatPrint(subVnuc_bound2,"subVnuc_bound2")

    subCoreJK = coreJK_sub( ints, loc2sub, dim, core1PDM_loc, Ne_core, Kcoeff )
    subTEI = ints.tei_sub( loc2sub, dim )

    ops = {"subKin":subKin}
    ops["subVnuc1"] = subVnuc1
    ops["subVnuc2"] = subVnuc2
    ops["subVnuc_bound1"] = subVnuc_bound1
    ops["subVnuc_bound2"] = subVnuc_bound2
    ops["subCoreJK"] = subCoreJK
    ops["subTEI"] = subTEI

    tools.timer("libgen.ops.build_subops",t0)
    return ops


def frag_kin_sub( ints, loc2sub, numActive ):

    kin_loc = ints.frag_kin_loc()
    kin_sub = tools.op_loc2sub(kin_loc, loc2sub[:,:numActive])
    return kin_sub

'''
def frag_vnuc_sub(ints, impAtom, loc2sub, numActive):

    vnuc_loc = ints.frag_vnuc_loc(impAtom)
    vnuc_sub = tools.op_loc2sub(vnuc_loc, loc2sub[:,:numActive])
    return vnuc_sub
'''

def frag_vnuc_sub(mol, ints, loc2sub, numActive):

    vnuc_loc = ints.frag_vnuc_loc(mol)
    vnuc_sub = tools.op_loc2sub(vnuc_loc, loc2sub[:,:numActive])
    return vnuc_sub


def coreJK_sub(ints, loc2sub, numActive, coreDMloc, Ne_core, Kcoeff = 1.0):

    t0 = tools.time0()
    sub_coreJK = None
    if(Ne_core == 0):
        sub_coreJK = 0.0        
    else:
        loc_coreJK = ints.coreJK_loc(coreDMloc, Kcoeff)
        sub_coreJK = tools.op_loc2sub(loc_coreJK, loc2sub[:,:numActive])

    t1 = tools.timer("coreJK_sub",t0)
    return sub_coreJK

