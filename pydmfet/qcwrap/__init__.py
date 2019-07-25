from . import pyscf_rks
from . import pyscf_rhf
from . import pyscf_rks_nonscf
from . import pyscf_rks_ao
from . import pyscf_rks_nonscf_ao

def qc_scf(use_suborb, software = 'pyscf', nonscf=False, mol=None, Ne=None, Norb=None, method=None, **kwargs):

    if software != 'pyscf':
        raise NotImplementedError("software %s is not implemented" % software)

    #will need to merge thses scf calls eventually
    if use_suborb:
        if Ne is None or Norb is None or method is None:
            raise ValueError("Ne, Norb, and method have to be set")

        if nonscf:
            return pyscf_rks_nonscf.rks_nonscf(Ne, Norb, method.lower(), mol=mol, **kwargs)

        if(method.lower() == 'hf'):
            return pyscf_rks.rhf_pyscf(Ne, Norb, mol=mol, **kwargs)
        else:
            return pyscf_rks.rks_pyscf(Ne, Norb, method.lower(), mol=mol, **kwargs)

    else:
        if mol is None:
            raise ValueError("mol can't be None")

        if nonscf:
            return pyscf_rks_nonscf_ao.rks_nonscf_ao(mol, **kwargs)

        return pyscf_rks_ao.rks_ao(mol, **kwargs)
