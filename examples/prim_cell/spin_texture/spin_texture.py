#! /usr/bin/env python
"""
Example for calculating spin texture for bismuth and graphene.
"""

import tbplas as tb

import bismuth as bi


def test(cell: tb.PrimitiveCell, ib: int = 0, spin_major: bool = True) -> None:
    # Evaluate expectation of sigma_z.
    k_grid = 2 * tb.gen_kmesh((240, 240, 1)) - 1
    spin_texture = tb.SpinTexture(cell, k_grid, spin_major)
    k_cart = spin_texture.k_cart
    sz = spin_texture.eval("z")
    vis = tb.Visualizer()
    vis.plot_scalar(x=k_cart[:, 0], y=k_cart[:, 1], z=sz[:, ib],
                    num_grid=(480, 480), cmap="jet")

    # Evaluate spin_texture
    k_grid = 2 * tb.gen_kmesh((48, 48, 1)) - 1
    spin_texture.k_grid = k_grid
    k_cart = spin_texture.k_cart
    sx = spin_texture.eval("x")
    sy = spin_texture.eval("y")
    vis.plot_vector(x=k_cart[:, 0], y=k_cart[:, 1], u=sx[:, ib], v=sy[:, ib])


def main():
    cell = bi.add_soc(bi.make_cell())
    test(cell, ib=7)
    cell = tb.make_graphene_soc(is_qsh=True)
    test(cell, ib=2, spin_major=False)


if __name__ == "__main__":
    main()
