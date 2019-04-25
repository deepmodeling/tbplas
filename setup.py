#!/usr/bin/env python

import os

try:
    import numpy as np
    from numpy.distutils.core import setup, Extension
except ImportError:
    print('Error: numpy not found. Installation failed.')
    exit()

fpath = 'tipsi/fortran/'

os.system("f2py -h f2py.pyf -m f2py " + fpath + "{analysis,tbpm}.f90 \
          --overwrite-signature")

f90files = ['const.f90', 'math.f90', 'csr.f90', 'fft.f90', 'random.f90',
            'propagation.f90', 'kpm.f90', 'funcs.f90',
            'tbpm.f90', 'analysis.f90']

sourcefiles = ['f2py.pyf']
for file in f90files:
    sourcefiles.append(fpath + file)

setup(
    name='tipsi',
    version='0.1',
    description='TIght-binding Propagation SImulator',
    packages=['tipsi', 'tipsi.fortran', 'tipsi.materials'],
    ext_modules=[Extension(name='tipsi.fortran.f2py',
                           sources=sourcefiles)]
)

os.system("rm -f f2py.pyf f2pymodule.c")
