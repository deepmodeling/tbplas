# cython: language_level=3
# cython: warn.undeclared=True
# cython: warn.unreachable=True
# cython: warn.maybe_uninitialized=True
# cython: warn.unused=True
# cython: warn.unused_arg=True
# cython: warn.unused_result=True
# cython: warn.multiple_declarators=True

import cython
from libc.math cimport pi, sqrt, exp
import numpy as np


cdef double _factorial(int n):
    """Evaluate the factorial of n."""
    cdef double y = 1.0
    cdef int i = 1
    for i in range(1, n+1):
        y *= i
    return y


cdef double _rnl_factor(int z, int n, int l):
    """Evaluate the renormalization factor of Rnl(r)."""
    cdef double a, b, c
    a = 2.0 / (n**2 * _factorial(2*l+1))
    b = sqrt(_factorial(n+l) / _factorial(n-l-1))
    c = sqrt(z**3)
    return a * b * c


cdef double _kummer(int alpha, int gamma, double ita):
    """Evaluate the Kummer function in Rnl(r)."""
    cdef double _sum = 1.0
    cdef double ck = 1.0
    cdef int k = 1
    for k in range(1, -alpha+1):
        ck = ck * (alpha+k-1) / (gamma+k-1) / k
        _sum = _sum + ck * ita**k
    return _sum


cdef double _radial(int z, int n, int l, double r):
    """Evaluate the un-normalized Rnl(r)."""
    cdef double ita, a, b, c
    ita = 2.0 * z / n * r
    a = exp(-0.5 * ita)
    b = ita**l
    c = _kummer(-n+l+1, 2*l+2, ita)
    return a * b * c


cdef double _coeff(int a, int b, int c, int d):
    """Evaluate the coefficient term in ylm(x, y, z)."""
    return a / b * sqrt(c / (d * pi))


cdef double _ylm_factor(int l, int m):
    """Evaluate the renormalization factor of ylm(x, y, z)."""
    cdef double factor
    if l == 0:
        factor = _coeff( 1, 2, 1, 1 )
    elif l == 1:
        if m == -1:  # py
            factor = _coeff( 1, 1, 3, 4 )
        elif m == 0:  # pz
            factor = _coeff( 1, 1, 3, 4 )
        elif m == 1:  # px
            factor = _coeff( 1, 1, 3, 4 )
        else:
            factor = 0.0
    elif l == 2:
        if m == -2:  # dxy
            factor = _coeff( 1, 2, 15, 1 )
        elif m == -1:  # dyz
            factor = _coeff( 1, 2, 15, 1 )
        elif m == 0:  # dz^2
            factor = _coeff( 1, 4,  5, 1 )
        elif m == 1:  # dxz
            factor = _coeff( 1, 2, 15, 1 )
        elif m == 2:  # d(x^2-y^2)
            factor = _coeff( 1, 4, 15, 1 )
        else:
            factor = 0.0
    elif l == 3:
        if m == -3:  # fy(3x^2-y^2)
            factor = _coeff( 1, 4,  35, 2 )
        elif m == -2:  # fxyz
            factor = _coeff( 1, 2, 105, 1 )
        elif m == -1:  # fyz^2
            factor = _coeff( 1, 4,  21, 2 )
        elif m ==  0:  # fz^3
            factor = _coeff( 1, 4,   7, 1 )
        elif m ==  1:  # fxz^2
            factor = _coeff( 1, 4,  21, 2 )
        elif m ==  2:  # fz(x^2-y^2)
            factor = _coeff( 1, 4, 105, 1 )
        elif m ==  3:  # fx(x^2-3y^2)
            factor = _coeff( 1, 4,  35, 2 )
        else: 
            factor = 0.0
    else:
        factor = 0.0
    return factor


cdef double _ylm(int l, int m, double x, double y, double z, double r):
    """
    Evaluate the un-normalized real spherical harmonics ylm(x, y, z).

    Reference:
    https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
    """
    cdef double denorm, ylm
    if l == 0:
        ylm = 1.0
    elif l == 1:
        denorm = r
        if m == -1:  # py
            ylm = 1.0 / denorm * y
        elif m == 0:  # pz
            ylm = 1.0 / denorm * z
        elif m == 1:  # px
            ylm = 1.0 / denorm * x
        else:
            ylm = 0.0
    elif l == 2:
        denorm = r**2
        if m == -2:  # dxy
            ylm = 1.0 / denorm * ( x*y )
        elif m == -1:  # dyz
            ylm = 1.0 / denorm * ( y*z )
        elif m == 0:  # dz^2
            ylm = 1.0 / denorm * ( 2*z*z - x*x - y*y )
        elif m == 1:  # dxz
            ylm = 1.0 / denorm * ( z*x )
        elif m == 2:  # d(x^2-y^2)
            ylm = 1.0 / denorm * ( x*x - y*y )
        else:
            ylm = 0.0
    elif l == 3:
        denorm = r**3
        if m == -3:  # fy(3x^2-y^2)
            ylm = 1.0 / denorm * y * ( 3*x*x -y*y )
        elif m == -2:  # fxyz
            ylm = 1.0 / denorm * ( x*y*z )
        elif m == -1:  # fyz^2
            ylm = 1.0 / denorm * y * ( 4*z*z - x*x - y*y )
        elif m ==  0:  # fz^3
            ylm = 1.0 / denorm * z * ( 2*z*z - 3*x*x - 3*y*y )
        elif m ==  1:  # fxz^2
            ylm = 1.0 / denorm * x * ( 4*z*z - x*x - y*y )
        elif m ==  2:  # fz(x^2-y^2)
            ylm = 1.0 / denorm * z * ( x*x - y*y )
        elif m ==  3:  # fx(x^2-3y^2)
            ylm = 1.0 / denorm * x * ( x*x - 3*y*y )
        else: 
            ylm = 0.0
    else:
        ylm = 0.0
    return ylm


@cython.boundscheck(False)
@cython.wraparound(False)
def set_cube(double [::1] c0, int[::1] qn, double[::1] cube_origin,
             int [::1] num_grid, double res, double [:,:,::1] cube):
    """
    Write atomic wavefunction to cube.

    Parameters
    ----------
    c0: (3,) float64 array
        Cartesian coordinate of the nuclei in bohr
    qn: (4,) int32 array
        z, n, l, m of the atom
    cube_origin: (3,) float64 array
        Cartesian coordinate of origin of cube in bohr
    num_grid: (3,) int32 array
        number of grid points along x, y, and z directions
    res: float64
        resolution of cube
    cube: (nx, ny, nz) float64 array
        cube to store the atomic wave function
    """
    cdef double xmin, ymin, zmin
    cdef double dx, dy, dz, dx2, dx2y2
    cdef double r, gridval, factor
    cdef int z, n, l, m
    cdef int i, j, k

    xmin, ymin, zmin = cube_origin[0], cube_origin[1], cube_origin[2]
    z, n, l, m = qn[0], qn[1], qn[2], qn[3]
    factor = _rnl_factor(z, n, l) * _ylm_factor(l, m)

    for i in range(num_grid[0]):
        dx = xmin + res * i - c0[0]
        dx2 = dx**2
        for j in range(num_grid[1]):
            dy = ymin + res * j - c0[1]
            dx2y2 = dx2 + dy**2
            for k in range(num_grid[2]):
                dz = zmin + res * k - c0[2]
                r = sqrt(dx2y2 + dz**2)
                if r < 1.0e-7:
                    gridval = 0.0
                else:
                    gridval = _radial(z, n, l, r) * _ylm(l, m, dx, dy, dz, r) * factor
                cube[i, j, k] = gridval
