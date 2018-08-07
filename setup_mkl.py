#!/usr/bin/env python

from numpy.distutils.core import setup, Extension


setup(name = 'tipsi',
      version = '0.1',
      description = 'TIght-binding Propagation SImulator',
      packages = ['tipsi', 'tipsi.fortran', 'tipsi.materials'],
      ext_modules = [Extension(name='tipsi.fortran.tbpm_f2py',
      sources=['tipsi/fortran/tbpm_f2py.pyf'\
               'tipsi/fortran/tbpm_module_mkl.f90',\
               'tipsi/fortran/tbpm_f2py_mkl.f90'])],
     )
