'''
This is a python wrapper for s-DMFET and s-PFET
Author: Xing Zhang
'''

__version__ = '0.1'  #test version

import os
from distutils.version import LooseVersion
import scipy
if LooseVersion(scipy.__version__) < LooseVersion('1.0.0'):
    raise SystemError("You're using an old version of Scipy (%s). "
                      "It is recommended to upgrad scipy to 1.0.0 or newer to support pydmfet." %
                      scipy.__version__)

__path__.append(os.path.join(os.path.dirname(__file__), 'tools'))

del(os, LooseVersion, scipy)
