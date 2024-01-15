#! /usr/bin/env python
import tbplas as tb
import numpy as np


def h2_molecule():
    """Plot the bonding and anti-boding states of H2 molecule."""
    # 1nm * 1nm * 1nm cubic cell
    lattice = np.eye(3, dtype=np.float64)
    h_bond = 0.074  # H-H bond length in nm
    prim_cell = tb.PrimitiveCell(lattice, unit=tb.NM)
    prim_cell.add_orbital((0.5-0.5*h_bond, 0.5, 0.5))
    prim_cell.add_orbital((0.5+0.5*h_bond, 0.5, 0.5))
    prim_cell.add_hopping((0, 0, 0), 0, 1, -1.0)
    qn = np.array([(1, 1, 0, 0) for _ in range(prim_cell.num_orb)])

    # Calculate wave function
    k_points = np.array([[0.0, 0.0, 0.0]])
    solver = tb.DiagSolver(prim_cell)
    bands, states = solver.calc_states(k_points, convention=1)

    # Define plotting range
    # Cube volume: [0.25, 0.75] * [0.25, 0.75] * [0.25, 0.75] in nm
    cube_origin = np.array([0.25, 0.25, 0.25])
    cube_size = np.array([0.5, 0.5, 0.5])
    rn_max = np.array([0, 0, 0])

    # Plot wave function
    vis = tb.Visualizer()
    vis.plot_wfc3d(prim_cell, wfc=states[0, 0], quantum_numbers=qn,
                   convention=1, k_point=k_points[0], rn_max=rn_max,
                   cube_name="h2.bond.cube", cube_origin=cube_origin,
                   cube_size=cube_size, kind="abs2")
    vis.plot_wfc3d(prim_cell, wfc=states[0, 1], quantum_numbers=qn,
                   convention=1, k_point=k_points[0], rn_max=rn_max,
                   cube_name="h2.anti-bond.cube", cube_origin=cube_origin,
                   cube_size=cube_size, kind="abs2")


def h2_chain():
    """Plot the wave function of a hydrogen chain."""
    # 0.074nm * 0.074nm * 0.074nm cubic cell
    lattice = 0.074 * np.eye(3, dtype=np.float64)
    prim_cell = tb.PrimitiveCell(lattice, unit=tb.NM)
    prim_cell.add_orbital((0.0, 0.0, 0.0))
    prim_cell.add_hopping((1, 0, 0), 0, 0, -1.0)
    qn = np.array([(1, 1, 0, 0) for _ in range(prim_cell.num_orb)])

    # Calculate wave function
    k_points = np.array([[0.0, 0.0, 0.0]])
    solver = tb.DiagSolver(prim_cell)
    bands, states = solver.calc_states(k_points, convention=1)

    # Define plotting range
    # Cube volume: [-0.75, 0.75] * [-0.25, 0.25] * [-0.25, 0.25] in nm
    cube_origin = np.array([-0.75, -0.25, -0.25])
    cube_size = np.array([1.5, 0.5, 0.5])
    rn_max = np.array([15, 0, 0])

    # Plot wave function
    vis = tb.Visualizer()
    vis.plot_wfc3d(prim_cell, wfc=states[0, 0], quantum_numbers=qn,
                   convention=1, k_point=k_points[0], rn_max=rn_max,
                   cube_origin=cube_origin, cube_size=cube_size, kind="real")


def graphene():
    prim_cell = tb.make_graphene_diamond()
    qn = np.array([(6, 2, 1, 0) for _ in range(prim_cell.num_orb)])

    # Calculate wave function
    k_points = np.array([[0.0, 0.0, 0.0]])
    solver = tb.DiagSolver(prim_cell)
    bands, states = solver.calc_states(k_points, convention=1)

    # Define plotting range
    # Cube volume: [-0.75, 0.75] * [-0.75, 0.75] * [-0.25, 0.25] in nm
    cube_origin = np.array([-0.75, -0.75, -0.25])
    cube_size = np.array([1.5, 1.5, 0.5])
    rn_max = np.array([3, 3, 0])

    # Plot wave function
    vis = tb.Visualizer()
    vis.plot_wfc3d(prim_cell, wfc=states[0, 0], quantum_numbers=qn,
                   convention=1, k_point=k_points[0], rn_max=rn_max,
                   cube_origin=cube_origin, cube_size=cube_size, kind="real")


if __name__ == "__main__":
    h2_molecule()
    h2_chain()
    graphene()
