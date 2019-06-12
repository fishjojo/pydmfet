#!/usr/bin/env python

"""PyDMFET: Python wrapper for DMFET

"""

DOCLINES = (__doc__ or '').split("\n")


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
        with concat_license_files():
            sdist.run(self)



def setup_package():

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




if __name__ == "__main__":

    setup_package()
