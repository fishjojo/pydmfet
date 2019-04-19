from pydmfet.qcwrap import pyscf_rks
from pydmfet.qcwrap import pyscf_rhf
#from pydmfet.qcwrap import pyscf_rks_ao

def qc_scf(Ne, Norb, method, software = 'pyscf', **args):

    mf_method = method.lower()

    if(software == 'pyscf'):
	if(mf_method == 'hf'):
	    return pyscf_rks.rhf_pyscf(Ne, Norb, **args)
	else:
	    return pyscf_rks.rks_pyscf(Ne, Norb, mf_method, **args)
    else:
	raise Exception("software NYI")

