'''
This is a python wrapper for s-DMFET and s-PFET
Author: Xing Zhang
'''

__version__ = '0.1'  #test version

import os
from distutils.version import LooseVersion
import numpy
if LooseVersion(numpy.__version__) <= LooseVersion('1.8.0'):
    raise SystemError("You're using an old version of Numpy (%s). "
                      "It is recommended to upgrad numpy to 1.8.0 or newer to support PySCF." %
                      numpy.__version__)

__path__.append(os.path.join(os.path.dirname(__file__), 'tools'))

del(os, LooseVersion, numpy)
