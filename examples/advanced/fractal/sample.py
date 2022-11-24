#! /usr/bin/env python
"""
Example for constructing Sierpinski carpet at 'Sample' level.

Equivalent to the example at 'PrimitiveCell' level, but much faster.
"""

import numpy as np

import tbplas as tb
from mask import Box, Mask


def top_down(prim_cell: tb.PrimitiveCell, start_width: int,
             iteration: int, extension: int) -> tb.Sample:
    """
    Build fractal in top-down approach.

    :param prim_cell: primitive cell of square lattice
    :param start_width: starting width of the sample
    :param iteration: iteration number of sample
    :param extension: extension of the sample width
    :return: fractal
    """
    # Create the supercell
    final_width = start_width * extension**iteration
    super_cell = tb.SuperCell(prim_cell, dim=(final_width, final_width, 1),
                              pbc=(False, False, False))

    # Create the mask
    start_box = Box(0, 0, final_width - 1, final_width - 1)
    mask = Mask(start_box, num_grid=extension, num_iter=iteration)

    # Etch the supercell
    mask.etch_super_cell(super_cell)

    # Make the sample
    sample = tb.Sample(super_cell)
    return sample


def bottom_up(prim_cell: tb.PrimitiveCell, start_width: int,
              iteration: int, extension: int) -> tb.Sample:
    """
    Build fractal in bottom-up approach.

    :param prim_cell: primitive cell of square lattice
    :param start_width: starting width of the sample
    :param iteration: iteration number of sample
    :param extension: extension of the sample width
    :return: fractal
    """
    final_width = start_width * extension**iteration

    # Build 0-th order fractal
    fractal = [(ia, ib)
               for ia in range(start_width)
               for ib in range(start_width)]

    # Build pattern for replication.
    pattern = [(ia, ib)
               for ia in range(extension)
               for ib in range(extension)
               if not (1 <= ia < extension-1 and 1 <= ib < extension-1)]

    # Build n-th order fractal by replicating (n-1)-th order according to
    # pattern, which is a direct product mathematically.
    for i in range(iteration):
        fractal_new = []
        width = start_width * extension**i
        for entry in pattern:
            di = width * entry[0]
            dj = width * entry[1]
            replica = [(grid[0] + di, grid[1] + dj) for grid in fractal]
            fractal_new.extend(replica)
        fractal = fractal_new

    # Get grid coordinates of vacancies.
    full_sites = [(ia, ib)
                  for ia in range(final_width)
                  for ib in range(final_width)]
    vacancies = list(set(full_sites).difference(set(fractal)))
    vacancies = [(grid[0], grid[1], 0, 0) for grid in vacancies]

    # Create the sample
    super_cell = tb.SuperCell(prim_cell, dim=(final_width, final_width, 1),
                              pbc=(False, False, False), vacancies=vacancies)
    sample = tb.Sample(super_cell)
    return sample


def main():
    # Create a square lattice
    lattice = np.eye(3, dtype=np.float64)
    prim_cell = tb.PrimitiveCell(lattice)
    prim_cell.add_orbital((0, 0))
    prim_cell.add_hopping((1, 0), 0, 0, 1.0)
    prim_cell.add_hopping((0, 1), 0, 0, 1.0)
    prim_cell.add_hopping((1, 1), 0, 0, 1.0)
    prim_cell.add_hopping((1, -1), 0, 0, 1.0)

    # Create fractal using top-down approach
    timer = tb.Timer()
    timer.tic("top_down")
    fractal = top_down(prim_cell, 2, 3, 3)
    timer.toc("top_down")
    fractal.plot(with_cells=False, with_orbitals=False, hop_as_arrows=False)

    # Create fractal using bottom-up approach
    timer.tic("bottom_up")
    fractal = bottom_up(prim_cell, 2, 3, 3)
    timer.toc("bottom_up")
    fractal.plot(with_cells=False, with_orbitals=False, hop_as_arrows=False)
    timer.report_total_time()


if __name__ == "__main__":
    main()
