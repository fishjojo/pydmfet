import os
import re
import sys
import sysconfig
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
MICRO = 1
ISRELEASED = True
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

topdir = os.path.abspath(os.path.join(__file__, '..'))

try:
    import numpy
except ImportError as e:
    print('**************************************************')
    print('* numpy was not installed in your system.  Please run')
    print('*     pip install numpy')
    print('* before installing pydmfet.')
    print('**************************************************')
    raise e

if (sys.platform.startswith('linux') or
    sys.platform.startswith('cygwin') or
    sys.platform.startswith('gnukfreebsd')):
    ostype = 'linux'
    so_ext = '.so'
    LD_LIBRARY_PATH = 'LD_LIBRARY_PATH'
elif sys.platform.startswith('darwin'):
    ostype = 'mac'
    so_ext = '.dylib'
    LD_LIBRARY_PATH = 'DYLD_LIBRARY_PATH'
    from distutils.sysconfig import get_config_vars
    conf_vars = get_config_vars()
    conf_vars['LDSHARED'] = conf_vars['LDSHARED'].replace('-bundle', '-dynamiclib')
    conf_vars['CCSHARED'] = " -dynamiclib"
    if sys.version_info[0] >= 3:  # python3
        conf_vars['EXT_SUFFIX'] = '.dylib'
    else:
        conf_vars['SO'] = '.dylib'
elif sys.platform.startswith('win'):
    ostype = 'windows'
    so_ext = '.dll'
elif sys.platform.startswith('aix') or sys.platform.startswith('os400'):
    ostype = 'aix'
    so_ext = '.so'
    LD_LIBRARY_PATH = 'LIBPATH'
    if(os.environ.get('PYSCF_INC_DIR') is None):
        os.environ['PYSCF_INC_DIR'] = '/QOpenSys/pkgs:/QOpenSys/usr:/usr:/usr/local'
else:
    raise OSError('Unknown platform')
    ostype = None


def search_lib_path(libname, extra_paths=None, version=None):
    paths = os.environ.get(LD_LIBRARY_PATH, '').split(os.pathsep)
    if 'PYDMFET_INC_DIR' in os.environ:
        PYDMFET_INC_DIR = os.environ['PYDMFET_INC_DIR'].split(os.pathsep)
        for p in PYDMFET_INC_DIR:
            paths = [p, os.path.join(p, 'lib'), os.path.join(p, '..', 'lib')] + paths
    if extra_paths is not None:
        paths += extra_paths

    len_libname = len(libname)
    for path in paths:
        full_libname = os.path.join(path, libname)
        if os.path.isfile(full_libname):
            if version is None or ostype == 'mac':
                return os.path.abspath(path)
            else:
                for f in os.listdir(path):
                    f_name = f[:len_libname]
                    f_version = f[len_libname+1:]
                    if (f_name == libname and f_version and
                        check_version(f_version, version)):
                        return os.path.abspath(path)

def search_inc_path(incname, extra_paths=None):
    paths = os.environ.get(LD_LIBRARY_PATH, '').split(os.pathsep)
    if 'PYDMFET_INC_DIR' in os.environ:
        PYDMFET_INC_DIR = os.environ['PYDMFET_INC_DIR'].split(os.pathsep)
        for p in PYSCF_INC_DIR:
            paths = [p, os.path.join(p, 'include'), os.path.join(p, '..', 'include')] + paths
    if extra_paths is not None:
        paths += extra_paths
    for path in paths:
        full_incname = os.path.join(path, incname)
        if os.path.exists(full_incname):
            return os.path.abspath(path)

if 'LDFLAGS' in os.environ:
    blas_found = any(x in os.environ['LDFLAGS']
                     for x in ('blas', 'atlas', 'openblas', 'mkl', 'Accelerate'))
else:
    blas_found = False

blas_include = []
blas_lib_dir = []
blas_libraries = []
blas_extra_link_flags = []
blas_extra_compile_flags = []
if not blas_found:
    np_blas = numpy.__config__.get_info('blas_opt')
    blas_include = np_blas.get('include_dirs', [])
    blas_lib_dir = np_blas.get('library_dirs', [])
    blas_libraries = np_blas.get('libraries', [])
    blas_path_guess = [search_lib_path('lib'+x+so_ext, blas_lib_dir)
                       for x in blas_libraries]
    blas_extra_link_flags = np_blas.get('extra_link_args', [])
    blas_extra_compile_flags = np_blas.get('extra_compile_args', [])
    if ostype == 'mac':
        if blas_extra_link_flags:
            blas_found = True
    else:
        if None not in blas_path_guess:
            blas_found = True
            blas_lib_dir = list(set(blas_path_guess))

if not blas_found:  # for MKL
    mkl_path_guess = search_lib_path('libmkl_rt'+so_ext, blas_lib_dir)
    if mkl_path_guess is not None:
        blas_libraries = ['mkl_rt']
        blas_lib_dir = [mkl_path_guess]
        blas_found = True
        print("Using MKL library in %s" % mkl_path_guess)

if not blas_found:
    possible_blas = ('blas', 'atlas', 'openblas')
    for x in possible_blas:
        blas_path_guess = search_lib_path('lib'+x+so_ext, blas_lib_dir)
        if blas_path_guess is not None:
            blas_libraries = [x]
            blas_lib_dir = [blas_path_guess]
            blas_found = True
            print("Using BLAS library %s in %s" % (x, blas_path_guess))
            break

if not blas_found:
    print("****************************************************************")
    print("*** WARNING: BLAS library not found.")
    print("* You can include the BLAS library in the global environment LDFLAGS, eg")
    print("*   export LDFLAGS='-L/path/to/blas/lib -lblas'")
    print("* or specify the BLAS library path in  PYDMFET_INC_DIR")
    print("*   export PYDMFET_INC_DIR=/path/to/blas/lib:/path/to/other/lib")
    print("****************************************************************")
    raise RuntimeError

distutils_lib_dir = 'lib.{platform}-{version[0]}.{version[1]}'.format(
    platform=sysconfig.get_platform(),
    version=sys.version_info)

pydmfet_lib_dir = os.path.join(topdir, 'pydmfet', 'libcpp')
build_lib_dir = os.path.join('build', distutils_lib_dir, 'pydmfet', 'libcpp')
default_lib_dir = [build_lib_dir] + blas_lib_dir
default_include = ['.', 'build', pydmfet_lib_dir] + blas_include

if not os.path.exists(os.path.join(topdir, 'build')):
    os.mkdir(os.path.join(topdir, 'build'))
with open(os.path.join(topdir, 'build', 'config.h'), 'w') as f:
    f.write('''
#if defined _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif
''')


def make_ext(pkg_name, relpath, srcs, libraries=[], library_dirs=default_lib_dir,
             include_dirs=default_include, extra_compile_flags=[],
             extra_link_flags=[], **kwargs):
    if '/' in relpath:
        relpath = os.path.join(*relpath.split('/'))
    if (os.path.isfile(os.path.join(pydmfet_lib_dir, 'build', 'CMakeCache.txt')) and
        os.path.isfile(os.path.join(pydmfet_lib_dir, *pkg_name.split('.')) + so_ext)):
        return None
    else:
        if sys.platform.startswith('darwin'):
            soname = pkg_name.split('.')[-1]
            extra_link_flags = extra_link_flags + ['-install_name', '@loader_path/'+soname+so_ext]
            runtime_library_dirs = []
        elif sys.platform.startswith('aix') or sys.platform.startswith('os400'):
            extra_compile_flags = extra_compile_flags + ['-fopenmp']
            extra_link_flags = extra_link_flags + ['-lblas', '-lgomp', '-Wl,-brtl']
            runtime_library_dirs = ['$ORIGIN', '.']
        else:
            extra_compile_flags = extra_compile_flags + ['-fopenmp']
            extra_link_flags = extra_link_flags + ['-fopenmp']
            runtime_library_dirs = ['$ORIGIN', '.']
        srcs = make_src(relpath, srcs)
        return Extension(pkg_name, srcs,
                         libraries = libraries,
                         library_dirs = library_dirs,
                         include_dirs = include_dirs + [os.path.join(pydmfet_lib_dir,relpath)],
                         extra_compile_args = extra_compile_flags,
                         extra_link_args = extra_link_flags,
# Be careful with the ld flag "-Wl,-R$ORIGIN" in the shell.
# When numpy.distutils is imported, the default CCompiler of distutils will be
# overwritten. Compilation is executed in shell and $ORIGIN will be converted to ''
                         runtime_library_dirs = runtime_library_dirs,
                         **kwargs)

def make_src(relpath, srcs):
    srcpath = os.path.join(pydmfet_lib_dir, relpath)
    abs_srcs = []
    for src in srcs.split():
        if '/' in src:
            abs_srcs.append(os.path.relpath(os.path.join(srcpath, *src.split('/'))))
        else:
            abs_srcs.append(os.path.relpath(os.path.join(srcpath, src)))
    return abs_srcs

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

extensions = []
extensions += [
    make_ext('pydmfet.libcpp.liblinalg', 'linalg',
             'svd.cpp',
             blas_libraries,
             extra_compile_flags=blas_extra_compile_flags,
             extra_link_flags=blas_extra_link_flags),
    make_ext('pydmfet.libcpp.libhess', 'hess',
             'hess.cpp',
             blas_libraries,
             extra_compile_flags=blas_extra_compile_flags,
             extra_link_flags=blas_extra_link_flags)]
extensions = [x for x in extensions if x is not None]

# Python ABI updates since 3.5
# https://www.python.org/dev/peps/pep-3149/
class BuildExtWithoutPlatformSuffix(build_ext):
    def get_ext_filename(self, ext_name):
        from distutils.sysconfig import get_config_var
        ext_path = ext_name.split('.')
        filename = build_ext.get_ext_filename(self, ext_name)
        name, ext_suffix = os.path.splitext(filename)
        return os.path.join(*ext_path) + ext_suffix


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
        maintainer_email = "xzhang8@caltech.edu",
        description = "subspace DMFET",
        long_description = "",
        author = "Xing Zhang",
        author_email = "xzhang8@caltech.edu",
        url = "https://github.com/fishjojo/pydmfet/pydmfet",
        project_urls={
            "Bug Tracker": "https://github.com/fishjojo/pydmfet/issues",
            "Source Code": "https://github.com/fishjojo/pydmfet/pydmfet",
        },
        license = 'BSD',
        classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
        include_package_data=True,
        packages=find_packages(exclude=['*test*', '*example*',
                                        '*setup.py']),
        platforms = ["Linux", "Mac OS-X", "Unix"],
        #test_suite='nose.collector',
        ext_modules=extensions,
        cmdclass={#"sdist": sdist_checked,
                  "build_ext": BuildExtWithoutPlatformSuffix},
        python_requires='>=3.5',
        zip_safe=False,
    )

    run_build = True

    from setuptools import setup
    if run_build:
    #    from numpy.distutils.core import setup
        metadata['configuration'] = configuration
        metadata['version'] = get_version_info()[0]
        setup(**metadata)
    else:
        metadata['version'] = get_version_info()[1]


if __name__ == "__main__":

    setup_package()

