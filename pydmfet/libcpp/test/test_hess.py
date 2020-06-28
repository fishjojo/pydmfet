import unittest
import numpy as np
from pydmfet.libcpp import oep_hess
from pyscf.lib import fingerprint

class KnownValues(unittest.TestCase):

    def test_hess(self):
        np.random.seed(0)
        NOrb = 10
        size = NOrb * (NOrb + 1) // 2
        jCa = np.random.rand(NOrb,NOrb)
        orb_Ea = np.random.rand(NOrb,1)
        NAlpha = 4
        hess = oep_hess(jCa, orb_Ea, size, NOrb, NAlpha=NAlpha, mo_occ=None, smear=0.0, sym_tab=None)
        self.assertAlmostEqual(fingerprint(hess), 1231.816324996269, 8)

    def test_hess_smear(self):
        np.random.seed(0)
        NOrb = 10
        size = NOrb * (NOrb + 1) // 2
        jCa = np.random.rand(NOrb,NOrb)
        orb_Ea = np.random.rand(NOrb,1)
        mo_occ = np.asarray([2., 2., 2., 1.85, 0.12, 0.03, 0., 0., 0., 0.,])
        hess = oep_hess(jCa, orb_Ea, size, NOrb, NAlpha=None, mo_occ=mo_occ, smear=0.05, sym_tab=None)
        self.assertAlmostEqual(fingerprint(hess), 1128.2042486908058, 8)

if __name__ == '__main__':
    unittest.main()
