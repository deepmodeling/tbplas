"""Functions for lattice operations."""

from math import sin, cos, sqrt, pi

import numpy as np


__all__ = ["gen_lattice_vectors", "gen_reciprocal_vectors",
           "frac2cart", "cart2frac", "rotate_coord",
           "get_lattice_area", "get_lattice_volume"]


def gen_lattice_vectors(a: float = 1.0,
                        b: float = 1.0,
                        c: float = 1.0,
                        alpha: float = 90.0,
                        beta: float = 90.0,
                        gamma: float = 90.0) -> np.ndarray:
    """
    Generate lattice vectors from given lattice parameters.

    Reference:
    http://www.quantum-espresso.org/Doc/INPUT_PW.html

    :param a: lattice constant 'a' in angstrom or nm
    :param b: lattice constant 'b' in angstrom or nm
    :param c: lattice constant 'c' in angstrom or nm
    :param alpha: angle between a2 and a3 in DEGREE
    :param beta: angle between a3 and a1 in DEGREE
    :param gamma: angle between a1 and a2 in DEGREE
    :return: (3, 3) float64 array
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


def gen_reciprocal_vectors(lattice_vectors: np.ndarray) -> np.ndarray:
    """
    Generate reciprocal lattice vectors from real-space lattice vectors.

    Here we evaluate reciprocal lattice vectors via
        dot_product(a_i, b_j) = 2 * pi * delta_{ij}
    The formulae based on cross-products are not robust in some cases.

    :param lattice_vectors: (3, 3) float64 array
        Cartesian coordinates of real-space lattice vectors
    :return: (3, 3) float64 array
        Cartesian coordinates of reciprocal lattice vectors
        Unit is inverse to the one of real-space lattice vectors.
    """
    reciprocal_vectors = np.zeros((3, 3))
    product = 2 * pi * np.eye(3)
    for i in range(3):
        reciprocal_vectors[i] = np.linalg.solve(lattice_vectors, product[i])
    return reciprocal_vectors


def cart2frac(lattice_vectors: np.ndarray,
              cartesian_coordinates: np.ndarray,
              origin: np.ndarray = np.zeros(3)) -> np.ndarray:
    """
    Convert Cartesian coordinates to fractional coordinates.

    :param lattice_vectors: (3, 3) float64 array
        Cartesian coordinates of lattice vectors
    :param cartesian_coordinates: (num_coord, 3) float64 array
        Cartesian coordinates to convert
    :param origin: (3,) float64 array
        Cartesian coordinate of lattice origin
    :return: (num_coord, 3) float64 array
        fractional coordinates in basis of lattice vectors
    """
    if not isinstance(lattice_vectors, np.ndarray):
        lattice_vectors = np.array(lattice_vectors)
    if not isinstance(cartesian_coordinates, np.ndarray):
        cartesian_coordinates = np.array(cartesian_coordinates)
    fractional_coordinates = np.zeros(cartesian_coordinates.shape)
    conversion_matrix = np.linalg.inv(lattice_vectors.T)
    for i, row in enumerate(cartesian_coordinates):
        fractional_coordinates[i] = np.matmul(conversion_matrix,
                                              (row - origin).T)
    return fractional_coordinates


def frac2cart(lattice_vectors: np.ndarray,
              fractional_coordinates: np.ndarray,
              origin: np.ndarray = np.zeros(3)) -> np.ndarray:
    """
    Convert fractional coordinates to Cartesian coordinates.

    :param lattice_vectors: (3, 3) float64 array
        Cartesian coordinates of lattice vectors
    :param fractional_coordinates: (num_coord, 3) float64 array
        fractional coordinates to convert in basis of lattice vectors
    :param origin: (3,) float64 array
        Cartesian coordinate of lattice origin
    :return: (num_coord, 3) float64 array
        Cartesian coordinates converted from fractional coordinates
    """
    if not isinstance(lattice_vectors, np.ndarray):
        lattice_vectors = np.array(lattice_vectors)
    if not isinstance(fractional_coordinates, np.ndarray):
        fractional_coordinates = np.ndarray(fractional_coordinates)
    cartesian_coordinates = np.zeros(fractional_coordinates.shape)
    conversion_matrix = lattice_vectors.T
    for i, row in enumerate(fractional_coordinates):
        cartesian_coordinates[i] = np.matmul(conversion_matrix, row.T) + origin
    return cartesian_coordinates


def rotate_coord(coord: np.ndarray,
                 angle: float = 0.0,
                 axis: str = "z",
                 center: np.ndarray = np.zeros(3)) -> np.ndarray:
    """
    Rotate Cartesian coordinates according to Euler angles.

    Reference:
    https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle

    Note that the right-hand rule is used, i.e., a positive value means the
    angle is counter-clockwise.

    :param coord: (num_coord, 3) float64 array
        Cartesian coordinates to rotate
    :param angle: float
        rotation angle in RADIAN, not degrees
    :param axis: string
        axis around which the rotation is performed
    :param center: (3,) float64 array
        Cartesian coordinate of rotation center
    :return: (num_coord, 3) float64 array
        rotated Cartesian coordinates
    :raises ValueError: if axis is not "x", "y" or "z"
    """
    if not isinstance(coord, np.ndarray):
        coord = np.array(coord)
    if not isinstance(center, np.ndarray):
        center = np.array(center)
    if len(center) != 3:
        raise ValueError(f"Length of rotation center should be 3")
    if axis not in ("x", "y", "z"):
        raise ValueError("Axis should be in 'x', 'y', 'z'")

    # Determine rotation matrix
    if axis == "x":
        u = np.array([1, 0, 0])
    elif axis == "y":
        u = np.array([0, 1, 0])
    else:
        u = np.array([0, 0, 1])
    ux, uy, uz = u
    u_prod = np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]])
    u_tens = np.tensordot(u, u, axes=0)
    cos_ang, sin_ang = cos(angle), sin(angle)
    rot_mat = cos_ang * np.eye(3) + sin_ang * u_prod + (1 - cos_ang) * u_tens

    # Rotate coordinates
    coord_rot = np.zeros(shape=coord.shape, dtype=coord.dtype)
    for i in range(coord.shape[0]):
        coord_rot[i] = np.matmul(rot_mat, coord[i] - center) + center
    return coord_rot


def get_lattice_area(lattice_vectors: np.ndarray,
                     direction: str = "c") -> float:
    """
    Calculate the area along given direction.

    :param lattice_vectors: (3, 3) float64 array
        Cartesian coordinates of lattice vectors
    :param direction: direction along which the area is evaluated, should be in
        ("a", "b", "c")
    :return: area along given direction in squared unit of lattice vectors
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


def get_lattice_volume(lattice_vectors: np.ndarray) -> float:
    """
    Calculate the volume formed by lattice vectors.

    :param lattice_vectors: (3, 3) float64 array
        Cartesian coordinates of lattice vectors
    :return: lattice volume in cubic unit of lattice vectors
    """
    a0 = lattice_vectors[0]
    a1 = lattice_vectors[1]
    a2 = lattice_vectors[2]
    return np.abs(np.dot(np.cross(a0, a1), a2)).item()
