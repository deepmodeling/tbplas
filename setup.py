#!/usr/bin/env python
import os


# Shared package information by FORTRAN and C extensions
pkg_info = {
    'name': 'tbplas',
    'version': '1.0',
    'description': 'Tight-binding Package for Large-scale Simulation',
    'long_description': 'TBPLaS is a tight-binding package for large scale '
                        'simulation, which implements featuring exact '
                        'diagonalization, kernel polynomial and propagation '
                        'methods.',
    'author': 'the TBPLaS development team',
    'author_email': 'liyunhai1016@whu.edu.cn',
    'url': 'www.tbplas.net',
    'license': 'BSD 3-clause',
    'platforms': 'Unix-like operating systems',
}
full_packages = ['tbplas', 'tbplas.adapter', 'tbplas.base', 'tbplas.builder',
                 'tbplas.cython', 'tbplas.diagonal',  'tbplas.fortran',
                 'tbplas.materials', 'tbplas.tbpm']
c_packages = ['tbplas.cython']
f_packages = set(full_packages).difference(c_packages)


# FORTRAN extensions
def f_setup():
    from numpy.distutils.core import setup, Extension

    # Generate the interface
    f90_dir = 'tbplas/fortran'
    f90_export = ['tbpm.f90', 'analysis.f90', 'lindhard.f90']
    f2py_cmd = f'f2py -h {f90_dir}/f2py.pyf -m f2py --overwrite-signature'
    for src in f90_export:
        f2py_cmd += f' {f90_dir}/{src}'
    os.system(f2py_cmd)

    # Define the extensions
    # DO NOT change the ordering of f90 files. Otherwise, the dependencies will
    # be violated the compilation will fail.
    f_sources = ['f2py.pyf', 'const.f90', 'math.F90', 'csr.F90', 'fft.F90',
                 'random.f90', 'propagation.f90', 'kpm.f90', 'funcs.f90',
                 'tbpm.f90', 'analysis.f90', 'lindhard.f90']
    f_sources = [f'{f90_dir}/{file}' for file in f_sources]
    f_extensions = [
        Extension(name='tbplas.fortran.f2py', sources=f_sources)
    ]

    # Run setup
    setup(**pkg_info, packages=f_packages, ext_modules=f_extensions)


# C Extensions
def c_setup():
    import configparser
    from setuptools import Extension, setup
    from Cython.Build import cythonize
    import numpy as np

    # Detect compiler from setup.cfg
    config = configparser.ConfigParser()
    config.read('setup.cfg')
    if 'config_cc' in config.sections():
        cc = config.get('config_cc', 'compiler')
    else:
        cc = 'unix'
    if cc == 'intelem':
        os.environ['CC'] = 'icc'
        os.environ['LDSHARED'] = 'icc -shared'

    # Define the extensions
    ext_names = ['primitive', 'super', 'sample', 'lindhard']
    c_extensions = [
        Extension(name=f"tbplas.cython.{name}",
                  sources=[f"tbplas/cython/{name}.pyx"],
                  include_dirs=[np.get_include()])
        for name in ext_names
    ]

    # Run setup
    setup(**pkg_info, packages=c_packages, ext_modules=cythonize(c_extensions))


if __name__ == "__main__":
    # FORTRAN extensions should go first. Otherwise, the compilation will fail.
    f_setup()
    c_setup()
