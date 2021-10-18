#!/usr/bin/env python


# Shared package information by FORTRAN and C extensions
PKG_INFO = {
    'name': 'tipsi',
    'version': '0.9.8',
    'description': 'TIght-binding Propagation SImulator',
}

# FORTRAN extension
try:
    import numpy as np
    from numpy.distutils.core import setup, Extension
except ImportError:
    print('Error: numpy not found. Installation failed.')
    exit()

f90_dir = 'tipsi/fortran'
# import os
# os.system('f2py -h %s/f2py.pyf -m f2py %s/{analysis,tbpm}.f90 \
#           --overwrite-signature' % (f90_dir, f90_dir))

# NOTE: DO NOT change the ordering of f90 files. Otherwise the
# dependencies will be violated the compilation will fail.
f90files = ['const.f90', 'math.F90', 'csr.F90', 'fft.F90', 'random.f90',
            'propagation.f90', 'kpm.f90', 'funcs.f90',
            'tbpm.f90', 'analysis.f90']
sourcefiles = [f'{f90_dir}/{file}' for file in f90files]
sourcefiles.insert(0, f'{f90_dir}/f2py.pyf')

setup(
    name=PKG_INFO['name'],
    version=PKG_INFO['version'],
    description=PKG_INFO['description'],
    packages=['tipsi', 'tipsi.fortran', 'tipsi.materials'],
    ext_modules=[Extension(name='tipsi.fortran.f2py',
                           sources=sourcefiles)]
)


# # C Extension
# try:
#     from setuptools import Extension, setup
# except ImportError:
#     print('Error: setuptools not found. Installation failed.')
#     exit()
# try:
#     from Cython.Build import cythonize
# except ImportError:
#     print('Error: Cython not found. Installation failed.')
#     exit()

# extensions = [
#     Extension(
#         "tipsi.builder.core",
#         ["tipsi/builder/core.pyx"],
#         include_dirs=[np.get_include()],
#     )
# ]

# setup(
#     name=PKG_INFO['name'],
#     version=PKG_INFO['version'],
#     description=PKG_INFO['description'],
#     packages=['tipsi.builder'],
#     ext_modules=cythonize(extensions),
# )
