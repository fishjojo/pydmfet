#!/usr/bin/env python

"""PyDMFET: Python wrapper for DMFET

"""

DOCLINES = (__doc__ or '').split("\n")


import os
import sys
import subprocess
import textwrap
import warnings
import sysconfig
from distutils.version import LooseVersion

if sys.version_info[:2] < (3, 5):
    raise RuntimeError("Python version >= 3.5 required.")

import builtins


CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Programming Language :: C
Programming Language :: Python
Programming Language :: Python :: 2.7
Topic :: Scientific/Engineering
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

MAJOR = 0
MINOR = 1
MICRO = 0
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION

# BEFORE importing setuptools, remove MANIFEST. Otherwise it may not be
# properly updated when the contents of directories change (true for distutils,
# not sure about setuptools).
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')


builtins.__PYDMFET_SETUP__ = True



def get_version_info():
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists('pydmfet/version.py'):
        import imp
        version = imp.load_source('pydmfet.version', 'pydmfet/version.py')
        GIT_REVISION = version.git_revision
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev0+' + GIT_REVISION[:7]

    return FULLVERSION, GIT_REVISION


def write_version_py(filename='pydmfet/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s
if not release:
    version = full_version
"""
    FULLVERSION, GIT_REVISION = get_version_info()

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()


#placeholder for building doc

###


def check_submodules():
    """ verify that the submodules are checked out and clean
        use `git submodule update --init`; on failure
    """
    if not os.path.exists('.git'):
        return
    with open('.gitmodules') as f:
        for l in f:
            if 'path' in l:
                p = l.split('=')[-1].strip()
                if not os.path.exists(p):
                    raise ValueError('Submodule %s missing' % p)

    proc = subprocess.Popen(['git', 'submodule', 'status'],
                            stdout=subprocess.PIPE)
    status, _ = proc.communicate()
    status = status.decode("ascii", "replace")
    for line in status.splitlines():
        if line.startswith('-') or line.startswith('+'):
            raise ValueError('Submodule not clean: %s' % line)


class license_files():
    """LICENSE.txt for sdist creation
    """
    def __init__(self):
        self.f1 = 'LICENSE.txt'

    def __enter__(self):
        with open(self.f1, 'r') as f1:
            self.bsd_text = f1.read()

    def __exit__(self, exception_type, exception_value, traceback):
        with open(self.f1, 'w') as f:
            f.write(self.bsd_text)


from distutils.command.sdist import sdist
class sdist_checked(sdist):
    """ check submodules on sdist to prevent incomplete tarballs """
    def run(self):
        check_submodules()
        with license_files():
            sdist.run(self)



def parse_setuppy_commands():

    args = sys.argv[1:]

    if not args:
        # User forgot to give an argument probably, let setuptools handle that.
        return True

    info_commands = ['--help-commands', '--name', '--version', '-V',
                     '--fullname', '--author', '--author-email',
                     '--maintainer', '--maintainer-email', '--contact',
                     '--contact-email', '--url', '--license', '--description',
                     '--long-description', '--platforms', '--classifiers',
                     '--keywords', '--provides', '--requires', '--obsoletes']

    for command in info_commands:
        if command in args:
            return False


    if 'install' in args:
        print(textwrap.dedent("""
            Note: if you need reliable uninstall behavior, then install
            with pip instead of using `setup.py install`:
              - `pip install .`       (from a git repo or downloaded source
                                       release)
            """))
        return True


    # If we got here, we didn't detect what setup.py command was given
    warnings.warn("Unrecognized setuptools command ('{}'), proceeding with "
                  "generating Cython sources and expanding templates".format(
                  ' '.join(sys.argv[1:])))
    return True


def configuration(parent_package='', top_path=None):
    from scipy._build_utils.system_info import get_info, NotFoundError
    from numpy.distutils.misc_util import Configuration

    lapack_opt = get_info('lapack_opt')

    if not lapack_opt:
        msg = 'No lapack/blas resources found.'
        if sys.platform == "darwin":
            msg = ('No lapack/blas resources found. '
                   'Note: Accelerate is no longer supported.')
        raise NotFoundError(msg)

    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('pydmfet')
    config.add_data_files(('pydmfet', '*.txt'))

    config.get_version('pydmfet/version.py')

    return config



def setup_package():

    write_version_py()

    try:
        import numpy
	import scipy
    except ImportError:  # We do not have numpy installed
        build_requires = ['numpy>=1.13.3', 'scipy>=1.2.1']
    else:
        # If we're building a wheel, assume there already exist numpy wheels
        # for this platform, so it is safe to add numpy to build requirements.
        # See gh-5184.
        build_requires = (['numpy>=1.13.3', 'scipy>=1.2.1'] if 'bdist_wheel' in sys.argv[1:]
                          else [])


    metadata = dict(
        name = "pydmfet",
        maintainer = "Xing Zhang",
        maintainer_email = "xingz@princeton.edu",
        description = DOCLINES[0],
        long_description = "\n".join(DOCLINES[2:]),
        #url = "",
        author = "Xing Zhang",
        #download_url = "",
        project_urls={
            "Bug Tracker": "https://github.com/fishjojo/pydmfet/issues",
            "Source Code": "https://github.com/fishjojo/pydmfet/pydmfet",
        },
        license = 'BSD',
        classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
        platforms = ["Linux", "Mac OS-X", "Unix"],
        test_suite='nose.collector',
        cmdclass={"sdist": sdist_checked},
        python_requires='>=2.7',
        zip_safe=False,
    )

    if "--force" in sys.argv:
        run_build = True
        sys.argv.remove('--force')
    else:
        # Raise errors for unsupported commands, improve help output, etc.
        run_build = parse_setuppy_commands()

    os.environ['ACCELERATE'] = 'None'


    from setuptools import setup

    if run_build:
        from numpy.distutils.core import setup

        # Customize extension building
        cmdclass['build_ext'] = get_build_ext_override()

        cwd = os.path.abspath(os.path.dirname(__file__))
        if not os.path.exists(os.path.join(cwd, 'PKG-INFO')):
            # Generate Cython sources, unless building from source release
            generate_cython()

	metadata['configuration'] = configuration
    else:
        metadata['version'] = get_version_info()[0]




    setup(**metadata)


if __name__ == "__main__":

    setup_package()
