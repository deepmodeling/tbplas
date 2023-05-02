from typing import List

import tbplas as tb


class Box:
    """
    Class representing a rectangular area spanning from [i0, j0] to (i1, j1).

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
    def __init__(self, i0: int, j0: int, i1: int, j1: int, void: bool = False):
        """
        :param i0: x-component of grid coordinate of bottom left corner
        :param j0: y-component of grid coordinate of bottom left corner
        :param i1: x-component of grid coordinate of top right corner
        :param j1: y-component of grid coordinate of top right corner
        :param void: whether the box is void or not
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
    def __init__(self, starting_box: Box, num_grid: int, num_iter: int = 0):
        """
        :param starting_box: starting partition of the area
        :param num_grid: umber of grid points when splitting boxes
        :param num_iter: number of fractal iteration
        """
        self.boxes = [starting_box]
        self.num_grid = num_grid
        for i in range(num_iter):
            new_boxes = []
            for box in self.boxes:
                new_boxes.extend(self.partition_box(box))
            self.boxes = new_boxes

    def partition_box(self, box: Box) -> List[Box]:
        """
        Partition given box into smaller boxes.

        :param box: box to split
        :return: smaller boxes split from given box
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

    def etch_super_cell(self, super_cell: tb.SuperCell) -> None:
        """
        Remove orbitals from supercell by checking if they fall in void boxes.

        :param super_cell: supercell to mask
        :return: None. The incoming supercell is modified.
        """
        masked_id_pc = []
        for box in self.boxes:
            if box.void:
                id_pc = [(ia, ib, 0, 0)
                         for ia in range(box.i0, box.i1)
                         for ib in range(box.j0, box.j1)]
                masked_id_pc.extend(id_pc)
        super_cell.unlock()
        super_cell.set_vacancies(masked_id_pc)

    def etch_prim_cell(self, prim_cell: tb.PrimitiveCell, width: int) -> None:
        """
        Remove orbitals from primitive cell by checking if they fall in void
        boxes.

        :param prim_cell: primitive cell to mask
        :param width: total width of the primitive cell
        :return: None. The incoming primitive cell is modified.
        """
        masked_id_pc = []
        for box in self.boxes:
            if box.void:
                id_pc = [(ia, ib)
                         for ia in range(box.i0, box.i1)
                         for ib in range(box.j0, box.j1)]
                masked_id_pc.extend(id_pc)
        masked_id_pc = [i[0]*width + i[1] for i in masked_id_pc]
        prim_cell.unlock()
        prim_cell.remove_orbitals(masked_id_pc)
