#! /usr/bin/env python
"""
In this tutorial we show how to build a fractal, Sierpinski carpet. We will try
two approaches: top-down and bottom-up. The first approach is implemented by
etching orbitals falling in void regions of a mask, while the second approach
is implemented by replication of given pattern. Top-down approach is simpler in
principle, but slower. Bottom-up approach is complicated in principle, but
faster.
"""

import numpy as np

import tbplas as tb


class Box:
    """
    Class representing a rectangular area.

    Attributes
    ----------
    i0: integer
        x-component of grid coordinate of bottom left corner
    j0: integer
        y-component of grid coordinate of bottom left corner
    i1: integer
        x-component of grid coordinate of top right corner
    j1: integer
        y-component of grid coordinate of top right corner
    void: boolean
        whether the box is void or not
        Orbitals falling in a void box will be removed.
    """
    def __init__(self, i0, j0, i1, j1, void=False):
        """
        :param i0: integer
            x-component of grid coordinate of bottom left corner
        :param j0: integer
            y-component of grid coordinate of bottom left corner
        :param i1: integer
            x-component of grid coordinate of top right corner
        :param j1: integer
            y-component of grid coordinate of top right corner
        :param void: boolean
            whether the box is void or not
        """
        self.i0 = i0
        self.j0 = j0
        self.i1 = i1
        self.j1 = j1
        self.void = void


class Mask:
    """
    Class for partitioning and masking a rectangular area.

    Attributes
    ----------
    boxes: list of 'Box' instances
        partition of the rectangular area
    num_grid: integer
        number of grid points when splitting boxes
    """
    def __init__(self, starting_box: Box, num_grid: int, num_iter=0):
        """
        :param starting_box: list of 'Box' instances
            starting partition of the area
        :param num_grid: integer
           number of grid points when splitting boxes
        :param num_iter: integer
            number of fractal iteration
        """
        self.boxes = [starting_box]
        self.num_grid = num_grid
        for i in range(num_iter):
            new_boxes = []
            for box in self.boxes:
                new_boxes.extend(self.partition_box(box))
            self.boxes = new_boxes

    def partition_box(self, box: Box):
        """
        Partition given box into smaller boxes.

        :param box: instance of 'Box' class
            box to split
        :return: sub_boxes: list of 'Box' instances
            smaller boxes split from given box
        """
        # Void box will be kept as-is
        if box.void:
            sub_boxes = [box]
        # Other box will be partitioned into num_grid*num_grid smaller
        # boxes with the center box marked as void.
        else:
            sub_boxes = []
            di = (box.i1 - box.i0 + 1) // self.num_grid
            dj = (box.j1 - box.j0 + 1) // self.num_grid
            for ii in range(self.num_grid):
                i0 = box.i0 + ii * di
                i1 = i0 + di
                for jj in range(self.num_grid):
                    j0 = box.j0 + jj * dj
                    j1 = j0 + dj
                    if (1 <= ii < self.num_grid - 1 and
                            1 <= jj < self.num_grid - 1):
                        void = True
                    else:
                        void = False
                    sub_boxes.append(Box(i0, j0, i1, j1, void))
        return sub_boxes

    def etch_super_cell(self, super_cell: tb.SuperCell):
        """
        Remove orbitals from supercell by checking if they fall in void boxes.

        :param super_cell: instance of 'SuperCell' class
            supercell to mask
        :return: None
            The incoming supercell is modified.
        """
        super_cell.sync_array()
        masked_id_pc = []
        for id_pc in super_cell.orb_id_pc:
            for box in self.boxes:
                if (box.void
                        and box.i0 <= id_pc.item(0) < box.i1
                        and box.j0 <= id_pc.item(1) < box.j1):
                    masked_id_pc.append(tuple(id_pc))
        super_cell.unlock()
        super_cell.vacancy_list = masked_id_pc
        super_cell.sync_array()


def top_down(prim_cell: tb.PrimitiveCell,
             start_width: int,
             iteration: int,
             extension: int):
    """
    Build fractal in top-down approach.

    :param prim_cell: instance of 'PrimitiveCell'
        primitive cell of square lattice
    :param start_width: integer
        starting width of the sample
    :param iteration: integer
        iteration number of sample
    :param extension: integer
        extension of the sample width
    :return: sample: instance of 'Sample' class
        fractal sample
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


def bottom_up(prim_cell: tb.PrimitiveCell,
              start_width: int,
              iteration: int,
              extension: int):
    """
    Build fractal in bottom-up approach.

    :param prim_cell: instance of 'PrimitiveCell'
        primitive cell of square lattice
    :param start_width: integer
        starting width of the sample
    :param iteration: integer
        iteration number of sample
    :param extension: integer
        extension of the sample width
    :return: sample: instance of 'Sample' class
        fractal sample
    """
    final_width = start_width * extension**iteration

    # Index_good is the starting fractal (0-th order) and pattern
    # replication.
    index_good = []
    for i in range(extension):
        for j in range(extension):
            if not (1 <= i < extension-1 and 1 <= j < extension-1):
                index_good.append([i, j])

    # Get grid coordinates of reserved orbitals
    frac_index = []
    frac_site = []
    for i in range(iteration):
        frac_index.append([0, 0])
    for i in range((extension*extension)**len(frac_index)):
        for j in range(len(frac_index)):
            if frac_index[j] != [extension-1, extension-1]:
                if frac_index[j][1] != extension-1:
                    frac_index[j][1] = frac_index[j][1] + 1
                else:
                    frac_index[j][0] = frac_index[j][0] + 1
                    frac_index[j][1] = 0
                break
            else:
                frac_index[j] = [0, 0]

        # Determine whether to replicate the fractal
        add_point = True
        for j in range(len(frac_index)):
            if frac_index[j] not in index_good:
                add_point = False
                break

        # Replicate the fractal
        if add_point:
            x0 = 0
            y0 = 0
            for j in range(len(frac_index)):
                x0 = x0 + frac_index[j][0] * start_width * extension**j
                y0 = y0 + frac_index[j][1] * start_width * extension**j
            for x in range(x0, x0+start_width):
                for y in range(y0, y0+start_width):
                    frac_site.append((x, y, 0, 0))
        i += 1

    # Get grid coordinates of vacancies.
    full_sites = [(ia, ib, 0, 0)
                  for ia in range(final_width)
                  for ib in range(final_width)]
    vacancies = list(set(full_sites).difference(set(frac_site)))

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

    # Create sample using top-down approach
    sample = top_down(prim_cell, 2, 3, 3)
    sample.plot(with_cells=False, with_orbitals=False, hop_as_arrows=False)

    # Create sample using bottom-up approach
    sample = bottom_up(prim_cell, 2, 3, 3)
    sample.plot(with_cells=False, with_orbitals=False, hop_as_arrows=False)


if __name__ == "__main__":
    main()
