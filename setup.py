#!/usr/bin/env python
import os
import configparser


# Shared package information by FORTRAN and C extensions
PKG_INFO = {
    'name': 'tbplas',
    'version': '1.0',
    'description': 'Tight-binding Package for Large-scale Simulation',
}
full_packages = ['tbplas', 'tbplas.adapter', 'tbplas.builder',
                 'tbplas.diagonal',  'tbplas.fortran', 'tbplas.materials',
                 'tbplas.tbpm']
c_packages = ['tbplas.builder', 'tbplas.diagonal']
f_packages = set(full_packages).difference(c_packages)

# FORTRAN extension
try:
    import numpy as np
    from numpy.distutils.core import setup, Extension
except ImportError:
    print('Error: numpy not found. Installation failed.')
    exit()

# Generate the interface
f90_dir = 'tbplas/fortran'
os.system(f'f2py -h {f90_dir}/f2py.pyf -m f2py --overwrite-signature '
          f'{f90_dir}/tbpm.f90 {f90_dir}/analysis.f90 {f90_dir}/lindhard.f90')

# NOTE: DO NOT change the ordering of f90 files. Otherwise, the
# dependencies will be violated the compilation will fail.
f90_files = ['const.f90', 'math.F90', 'csr.F90', 'fft.F90', 'random.f90',
             'propagation.f90', 'kpm.f90', 'funcs.f90',
             'tbpm.f90', 'analysis.f90', 'lindhard.f90']
f_sources = [f'{f90_dir}/{file}' for file in f90_files]
f_sources.insert(0, f'{f90_dir}/f2py.pyf')

f_extensions = [
    Extension(
        name='tbplas.fortran.f2py',
        sources=f_sources
    )
]

setup(
    name=PKG_INFO['name'],
    version=PKG_INFO['version'],
    description=PKG_INFO['description'],
    packages=f_packages,
    ext_modules=f_extensions,
)


# C Extension
try:
    from setuptools import Extension, setup
except ImportError:
    print('Error: setuptools not found. Installation failed.')
    exit()
try:
    from Cython.Build import cythonize
except ImportError:
    print('Error: Cython not found. Installation failed.')
    exit()

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

ext_names = ['tbplas.builder.core', 'tbplas.diagonal.core']
c_extensions = [
    Extension(name=name,
              sources=[f"{name.replace('.', '/')}.pyx"],
              include_dirs=[np.get_include()])
    for name in ext_names
]

setup(
    name=PKG_INFO['name'],
    version=PKG_INFO['version'],
    description=PKG_INFO['description'],
    packages=c_packages,
    ext_modules=cythonize(c_extensions),
)
