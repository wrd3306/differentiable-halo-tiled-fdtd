"""Force MEEP to use the canonical hard voxel mask and verify epsilon.

The geometry is built from the shared voxel index array using MaterialGrid with
do_averaging=False. MEEP's MaterialGrid grid_size is the number of grid points
(not a physical length), so each material gets a bbox-restricted block whose
weights are the exact voxel array. sim.get_epsilon() is then compared against
the float er map reconstructed from the same mask, with no time marching.
"""
import os
import sys

import meep as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VOX = os.path.join(HERE, "..", "..", "results", "external", "voxels")
A, B, L = 24e-3, 12e-3, 100e-3


def load(tag):
    z = np.load(os.path.join(VOX, tag + ".npz"))
    return z["idx"], z["db_er"].astype(np.float64), float(z["grid_mm"])


def build_geometry(idx, db, grid):
    dx = grid * 1e-3
    blocks = []
    wall_t = 1.0e-3
    for mi in range(1, db.shape[0]):
        occ = (idx == mi)
        if not occ.any():
            continue
        ix, iy, iz = np.where(occ)
        imin, imax = ix.min(), ix.max() + 1
        jmin, jmax = iy.min(), iy.max() + 1
        kmin, kmax = iz.min(), iz.max() + 1
        x0, x1 = imin * dx - A / 2, imax * dx - A / 2
        y0, y1 = jmin * dx - B / 2, jmax * dx - B / 2
        z0, z1 = kmin * dx - L / 2, kmax * dx - L / 2
        sub = np.ascontiguousarray(
            occ[imin:imax, jmin:jmax, kmin:kmax], dtype=np.float64)
        # grid_size is the point count along each axis of the bbox.
        mg = mp.MaterialGrid(
            mp.Vector3(imax - imin, jmax - jmin, kmax - kmin),
            mp.Medium(epsilon=1.0), mp.Medium(epsilon=float(db[mi])),
            weights=sub.flatten(), do_averaging=False)
        blocks.append(mp.Block(
            material=mg,
            center=mp.Vector3((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2),
            size=mp.Vector3(x1 - x0, y1 - y0, z1 - z0)))
    # PEC walls appended last so they override at the wall cells.
    for xc in (-A / 2, A / 2):
        blocks.append(mp.Block(material=mp.perfect_electric_conductor,
                               center=mp.Vector3(xc, 0, 0),
                               size=mp.Vector3(wall_t, B + 2 * wall_t,
                                               L + 2 * wall_t)))
    for yc in (-B / 2, B / 2):
        blocks.append(mp.Block(material=mp.perfect_electric_conductor,
                               center=mp.Vector3(0, yc, 0),
                               size=mp.Vector3(A + 2 * wall_t, wall_t,
                                               L + 2 * wall_t)))
    return blocks


def main():
    tags = sys.argv[1:] or ["full_slab_0.5mm", "two_material_0.5mm",
                            "full_slab_0.25mm", "centered_post_0.5mm",
                            "grating_0.5mm"]
    for tag in tags:
        idx, db, grid = load(tag)
        dx = grid * 1e-3
        n_x, n_y, n_z = idx.shape
        geom = build_geometry(idx, db, grid)
        sim = mp.Simulation(
            cell_size=mp.Vector3(n_x * dx, n_y * dx, n_z * dx),
            geometry=geom, resolution=1.0 / dx, eps_averaging=False)
        sim.init_sim()
        eps = sim.get_epsilon()
        expect = db[idx]
        match = np.allclose(eps, expect, atol=1e-6)
        print(tag, "exact epsilon match:", match)
        if not match:
            bad = ~np.isclose(eps, expect, atol=1e-6)
            print("  n mismatched:", int(bad.sum()),
                  "values:", np.unique(eps[bad])[:8])


if __name__ == "__main__":
    main()
