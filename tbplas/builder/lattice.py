"""
Functions for lattice operations.

Functions
---------
    gen_lattice_vectors: user function
        generate lattice vectors from lattice parameters
    gen_lattice_vectors2: user function
        generate lattice vectors from given vectors
    gen_reciprocal_vectors: user function
        generate reciprocal lattice vectors from real-space lattice
        vectors
    cart2frac: user function
        convert Cartesian coordinates to fractional coordinates
    frac2cart: user function
        convert fractional coordinates to Cartesian coordinates
    rotate_coord: user function
        rotate Cartesian coordinates according to Euler angles
    get_lattice_area: user function
        calculate the area formed by lattice vectors along given direction
    get_lattice_volume: user function
        calculate the volume formed by lattice vectors
"""

from math import sin, cos, sqrt, pi

import numpy as np


def gen_lattice_vectors(a=1.0, b=1.0, c=1.0, alpha=90.0, beta=90.0, gamma=90.0):
    """
    Generate lattice vectors from given lattice parameters.

    Reference:
    http://www.quantum-espresso.org/Doc/INPUT_PW.html

    :param a: float
        lattice constant a in angstrom or nm
    :param b: float
        lattice constant b in angstrom or nm
    :param c: float
        lattice constant c in angstrom or nm
    :param alpha: float
        angle between a2 and a3 in DEGREE
    :param beta: float
        angle between a3 and a1 in DEGREE
    :param gamma: float
        angle between a1 and a2 in DEGREE
    :return: lattice_vectors: (3, 3) float64 array
        Cartesian coordinates of lattice vectors in the same unit as a/b/c
    """
    alpha = alpha / 180 * pi
    beta = beta / 180 * pi
    gamma = gamma / 180 * pi
    lattice_vectors = np.zeros((3, 3))
    lattice_vectors[0, :] = [a, 0, 0]
    lattice_vectors[1, :] = [b*cos(gamma), b*sin(gamma), 0]
    lattice_vectors[2, 0] = c * cos(beta)
    lattice_vectors[2, 1] = c * (cos(alpha) - cos(beta)*cos(gamma)) / sin(gamma)
    lattice_vectors[2, 2] = c * sqrt(1 + 2*cos(alpha)*cos(beta)*cos(gamma)
                                     - cos(alpha)**2 - cos(beta)**2
                                     - cos(gamma)**2) / sin(gamma)
    # Remove small fluctuations due to sqrt/sin/cos
    for i in range(3):
        for j in range(3):
            if abs(lattice_vectors[i, j]) < 1.0e-15:
                lattice_vectors[i, j] = 0.0
    return lattice_vectors


def gen_lattice_vectors2(a1, a2, a3=None):
    """
    Generate lattice vectors from given a1, a2 and a3.

    :param a1: tuple or list of floats
        Cartesian coordinates of lattice vector a1
    :param a2: tuple or list of floats
        Cartesian coordinates of lattice vector a2
    :param a3: tuple or list of floats
        Cartesian coordinates of lattice vector a3
    :return: lattice_vectors: (3, 3) float64 array
        Cartesian coordinates of lattice vectors in the same unit as
        a1, a2 and a3
    """
    assert len(a1) in (2, 3)
    a1 = tuple(a1)
    if len(a1) == 2:
        a1 += (0.0,)
    assert len(a2) in (2, 3)
    a2 = tuple(a2)
    if len(a2) == 2:
        a2 += (0.0,)
    if a3 is None:
        a3 = [0.0, 0.0, 1.0]
    assert len(a3) == 3
    lattice_vectors = np.zeros((3, 3))
    lattice_vectors[0] = a1
    lattice_vectors[1] = a2
    lattice_vectors[2] = a3
    return lattice_vectors


def gen_reciprocal_vectors(lattice_vectors):
    """
    Generate reciprocal lattice vectors from real-space lattice vectors.

    NOTE: Here we evaluate reciprocal lattice vectors via
        dot_product(a_i, b_j) = 2 * pi * delta_{ij}
    The formulae based on cross-products are not robust in some cases.

    :param lattice_vectors: (3, 3) float64 array
        Cartesian coordinates of real-space lattice vectors
    :return: reciprocal_vectors: (3, 3) float64 array
        Cartesian coordinates of reciprocal lattice vectors
        Unit is inverse to the one of real-space lattice vectors.
    """
    reciprocal_vectors = np.zeros((3, 3))
    product = 2 * pi * np.eye(3)
    for i in range(3):
        reciprocal_vectors[i] = np.linalg.solve(lattice_vectors, product[i])
    return reciprocal_vectors


def cart2frac(lattice_vectors, cartesian_coordinates):
    """
    Convert Cartesian coordinates to fractional coordinates.

    :param lattice_vectors: (3, 3) float64 array
        Cartesian coordinates of lattice vectors
    :param cartesian_coordinates: (num_coord, 3) float64 array
        Cartesian coordinates to convert
    :return: fractional_coordinates: (num_coord, 3) float64 array
        fractional coordinates in basis of lattice vectors
    """
    fractional_coordinates = np.zeros(cartesian_coordinates.shape)
    conversion_matrix = np.linalg.inv(lattice_vectors.T)
    for i, row in enumerate(cartesian_coordinates):
        fractional_coordinates[i] = np.matmul(conversion_matrix, row.T)
    return fractional_coordinates


def frac2cart(lattice_vectors, fractional_coordinates):
    """
    Convert fractional coordinates to Cartesian coordinates.

    :param lattice_vectors: (3, 3) float64 array
        Cartesian coordinates of lattice vectors
    :param fractional_coordinates: (num_coord, 3) float64 array
        fractional coordinates to convert in basis of lattice vectors
    :return: cartesian coordinates: (num_coord, 3) float64 array
        Cartesian coordinates converted from fractional coordinates
    """
    cartesian_coordinates = np.zeros(fractional_coordinates.shape)
    conversion_matrix = lattice_vectors.T
    for i, row in enumerate(fractional_coordinates):
        cartesian_coordinates[i] = np.matmul(conversion_matrix, row.T)
    return cartesian_coordinates


def rotate_coord(coord, angle=0.0, axis="z"):
    """
    Rotate Cartesian coordinates according to Euler angles.

    Reference:
    https://mathworld.wolfram.com/RotationMatrix.html
    https://mathworld.wolfram.com/EulerAngles.html

    Note that in the reference the axes are rotated, not the vectors.
    So here you will see a minus sign before the angles.

    :param coord: (num_coord, 3) float64 array
        Cartesian coordinates to rotate
    :param angle: float
        rotation angle in RADIAN, not degrees
    :param axis: string
        axis around which the rotation is performed
        x - pitch, y - roll, z - yawn
    :return: coord_rot: (num_coord, 3) float64 array
        rotated Cartesian coordinates
    :raises ValueError: if axis is not "x", "y" or "z"
    """
    cos_ang, sin_ang = cos(-angle), sin(-angle)
    rot_mat = np.eye(3)
    if axis == "x":
        rot_mat[1, 1] = cos_ang
        rot_mat[1, 2] = sin_ang
        rot_mat[2, 1] = -sin_ang
        rot_mat[2, 2] = cos_ang
    elif axis == "y":
        rot_mat[0, 0] = cos_ang
        rot_mat[0, 2] = -sin_ang
        rot_mat[2, 0] = sin_ang
        rot_mat[2, 2] = cos_ang
    elif axis == "z":
        rot_mat[0, 0] = cos_ang
        rot_mat[0, 1] = sin_ang
        rot_mat[1, 0] = -sin_ang
        rot_mat[1, 1] = cos_ang
    else:
        raise ValueError("Axis should be in 'x', 'y', 'z'")
    coord_rot = np.zeros(shape=coord.shape, dtype=coord.dtype)
    for i in range(coord.shape[0]):
        coord_rot[i] = np.matmul(rot_mat, coord[i])
    return coord_rot


def get_lattice_area(lattice_vectors, direction="c"):
    """
    Calculate the area along given direction.

    :param lattice_vectors: (3, 3) float64 array
        Cartesian coordinates of lattice vectors
    :param direction: string
        direction along which the area is evaluated
        should be in ("a", "b", "c")
    :return: float
        area along given direction in squared unit of lattice vectors
    :raises ValueError: if direction is not in ("a", "b", "c")
    """
    if direction == "a":
        a0 = lattice_vectors[1]
        a1 = lattice_vectors[2]
    elif direction == "b":
        a0 = lattice_vectors[2]
        a1 = lattice_vectors[0]
    elif direction == "c":
        a0 = lattice_vectors[0]
        a1 = lattice_vectors[1]
    else:
        raise ValueError("Direction should be in 'a', 'b', 'c'")
    return np.linalg.norm(np.cross(a0, a1)).item()


def get_lattice_volume(lattice_vectors):
    """
    Calculate the volume formed by lattice vectors.

    :param lattice_vectors: (3, 3) float64 array
        Cartesian coordinates of lattice vectors
    :return: float
        lattice volume in cubic unit of lattice vectors
    """
    a0 = lattice_vectors[0]
    a1 = lattice_vectors[1]
    a2 = lattice_vectors[2]
    return np.abs(np.dot(np.cross(a0, a1), a2)).item()
