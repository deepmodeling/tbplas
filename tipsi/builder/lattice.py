"""Functions for lattice operations."""

from math import sin, cos, sqrt, pi
import numpy as np


def gen_lattice_vectors(a=1.0, b=1.0, c=1.0, alpha=90.0, beta=90.0, gamma=90.0):
    """
    Generate lattice vectors from given lattice parameters.

    :param a: float, lattice constant a in angstrom or nm
    :param b: float, lattice constant b in angstrom or nm
    :param c: float, lattice constant c in angstrom or nm
    :param alpha: float, angle between a2 and a3 in DEGREE
    :param beta: float, angle between a3 and a1 in DEGREE
    :param gamma: float, angle between a1 and a2 in DEGREE
    :return lattice_vectors: a 3*3 array containing the lattice vectors, in the
        same unit as a/b/c.
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

    :param a1: tuple or list of floats, lattice vector a1
    :param a2: tuple or list of floats, lattice vector a2
    :param a3: tuple or list of floats, lattice vector a3
    :return lattice_vectors: a 3*3 array containing the lattice vectors, in the
        same unit as a/b/c.
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

    :param lattice_vectors: 3*3 array
    :return: reciprocal_vectors: 3*3 array
    """
    reciprocal_vectors = np.zeros((3, 3))
    product = 2 * pi * np.eye(3)
    for i in range(3):
        reciprocal_vectors[i] = np.linalg.solve(lattice_vectors, product[i])
    return reciprocal_vectors


def cart2frac(lattice_vectors, cartesian_coordinates):
    """
    Convert Cartesian coordinates to fractional coordinates.

    :param lattice_vectors: 3*3 array containing the coordinates of lattice
            vectors
    :param cartesian_coordinates: N*3 array containing the Cartesian coordinates
            of atoms
    :return: fractional_coordinates: N*3 array
    """
    fractional_coordinates = np.zeros(cartesian_coordinates.shape)
    conversion_matrix = np.linalg.inv(lattice_vectors.T)
    for i, row in enumerate(cartesian_coordinates):
        fractional_coordinates[i] = np.matmul(conversion_matrix, row.T)
    return fractional_coordinates


def frac2cart(lattice_vectors, fractional_coordinates):
    """
    Convert fractional coordinates to Cartesian coordinates.

    :param lattice_vectors: 3*3 array
    :param fractional_coordinates: N*3 array
    :return: cartesian coordinates: N*3 array
    """
    cartesian_coordinates = np.zeros(fractional_coordinates.shape)
    conversion_matrix = lattice_vectors.T
    for i, row in enumerate(fractional_coordinates):
        cartesian_coordinates[i] = np.matmul(conversion_matrix, row.T)
    return cartesian_coordinates


def get_lattice_area(lattice_vectors, direction="c"):
    """
    Calculate the area along given direction.

    :param lattice_vectors: 3*3 array
    :param direction: character, direction along which the area is evaluated
    :return: float, area along given direction
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

    :param lattice_vectors: 3*3 array
    :return: float, lattice volume
    """
    a0 = lattice_vectors[0]
    a1 = lattice_vectors[1]
    a2 = lattice_vectors[2]
    return np.abs(np.dot(np.cross(a0, a1), a2)).item()
