from __future__ import division, print_function, absolute_import


try:
    __PYDMFET_SETUP__
except NameError:
    __PYDMFET_SETUP__ = False


if __PYDMFET_SETUP__:
    import sys as _sys
    _sys.stderr.write('Running from pydmfet source directory.\n')
    del _sys
else:
    '''
    try:
        from pydmfet.__config__ import show as show_config
    except ImportError:
        msg = """Error importing pydmfet: you cannot import pydmfet while
        being in pydmfet source director."""
        raise ImportError(msg)
    '''
    try:
        from pydmfet.version import version as __version__
    except ImportError:
        msg = """Error importing pydmfet: you cannot import pydmfet while
        being in pydmfet source director."""
        raise ImportError(msg)

    from distutils.version import LooseVersion
    import scipy
    if LooseVersion(scipy.__version__) < LooseVersion('1.2.1'):
        import warnings
        warnings.warn("Scipy 1.2.1 or above is required for this version of "
                      "pydmfet (detected version %s)" % scipy.__version__,
                      UserWarning)

    del (LooseVersion,scipy)
