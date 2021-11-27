"""
Helper classes and functions used among builder package.

Classes
-------
    None.

Functions
---------
    proj_coord: developer function
        project 3-dimensional coordinates onto given view
"""

import numpy as np


def proj_coord(coord: np.ndarray, view="ab"):
    """
    Project 3-dimensional coordinates onto given view.

    :param coord: (num_row, 3) float64 array
        coordinates for projection
    :param view: string
        kind of view, should be in ('ab', 'bc', 'ca', 'ba', 'cb', 'ac')
    :return: projected_coord (num_row, 2) or (2,) float64 array
        projected coordinates
    :raises ValueError: if view is illegal
    """
    if view not in ('ab', 'bc', 'ca', 'ba', 'cb', 'ac'):
        raise ValueError(f"Illegal view {view}")
    idx_dict = {'a': 0, 'b': 1, 'c': 2}
    idx_x, idx_y = idx_dict[view[0]], idx_dict[view[1]]
    if len(coord.shape) == 1:
        projected_coord = coord[[idx_x, idx_y]]
    else:
        projected_coord = coord[:, [idx_x, idx_y]]
    return projected_coord
