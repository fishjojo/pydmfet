import time
import numpy as np
from pydmfet import tools

def build_subops(impAtom, boundary_atoms,boundary_atoms2, ints, loc2sub, core1PDM_loc, dim):

    t0 = (time.clock(), time.time())

    subKin = ints.frag_kin_sub( impAtom, loc2sub, dim )
    subVnuc1 = ints.frag_vnuc_sub( impAtom, loc2sub, dim)
    subVnuc2 = ints.frag_vnuc_sub( 1-impAtom, loc2sub, dim )

    subVnuc_bound1 = ints.bound_vnuc_sub(boundary_atoms, loc2sub, dim )
    subVnuc_bound2 = -subVnuc_bound1
    if(boundary_atoms2 is not None):
	subVnuc_bound2 = -ints.bound_vnuc_sub(boundary_atoms2, loc2sub, dim )

    #tools.MatPrint(subVnuc_bound1,"subVnuc_bound1")
    #tools.MatPrint(subVnuc_bound2,"subVnuc_bound2")

    subCoreJK = ints.coreJK_sub( loc2sub, dim, core1PDM_loc )
    subTEI = ints.dmet_tei( loc2sub, dim )

    ops = {"subKin":subKin}
    ops["subVnuc1"] = subVnuc1
    ops["subVnuc2"] = subVnuc2
    ops["subVnuc_bound1"] = subVnuc_bound1
    ops["subVnuc_bound2"] = subVnuc_bound2
    ops["subCoreJK"] = subCoreJK
    ops["subTEI"] = subTEI

    tools.timer("libgen.build_subops",t0)
    return ops

