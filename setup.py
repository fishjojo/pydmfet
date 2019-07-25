import os
import re
import sys
import platform
import subprocess

from setuptools import find_packages, Extension
from setuptools.command.build_ext import build_ext
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
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
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


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-DCMAKE_C_COMPILER=' + 'icc',
                      '-DCMAKE_CXX_COMPILER=' + 'icpc',
                      '-DVERSION_INFO=' + self.distribution.get_version()]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j4']

        env = os.environ.copy()
        #env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
        #                                                      self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


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
        description = "sDMFET",
        long_description = "",
        author = "Xing Zhang",
        project_urls={
            "Bug Tracker": "https://github.com/fishjojo/pydmfet/issues",
            "Source Code": "https://github.com/fishjojo/pydmfet/pydmfet",
        },
        license = 'BSD',
        classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
        packages=find_packages(exclude=['*test*', '*example*',
                                        '*setup.py']),
        platforms = ["Linux", "Mac OS-X", "Unix"],
        #test_suite='nose.collector',
        ext_modules=[CMakeExtension('pydmfet.libcpp.libhess'),
                     CMakeExtension('pydmfet.libcpp.libsvd')],
        cmdclass={"sdist": sdist_checked,
                  "build_ext": CMakeBuild},
        python_requires='>=3.5',
        zip_safe=False,
    )

    run_build = False

    from setuptools import setup
    if run_build:
    #    from numpy.distutils.core import setup
        metadata['configuration'] = configuration
    else:
        metadata['version'] = get_version_info()[0]

    setup(**metadata)


if __name__ == "__main__":

    setup_package()

