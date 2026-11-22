"""Library of genuinely 3-D waveguide devices and their rasterization.

Each device is a list of boxes in physical millimetres, (x0, x1, y0, y1, z0, z1,
er), inside the a=24 mm, b=12 mm guide. The boxes are rasterized onto a scaled
grid to produce a full 3-D relative-permittivity map. Unlike the
full-cross-section slab, these devices are non-uniform over the cross-section,
so the internal fields are genuinely 3-D and excite evanescent higher modes;
the ports remain single-mode (TE10) because the next modes are cut off at
10 GHz.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "engine"))
import fdtd_gpu as fg  # noqa: E402


def config3d(scale):
    """Base waveguide configuration at a given refinement scale."""
    s = scale
    dx = 1e-3 / s
    return dict(Nx=24 * s, Ny=12 * s, Nz=100 * s, dx=dx, dy=dx, dz=dx,
                dt=fg.cfl_dt(dx, dx, dx), f0=10e9, src_z=15 * s,
                npml=10 * s,
                pulse=dict(t0=1.0e-9, tw=0.3e-9, fc=10e9))


def planes(cfg):
    """Monitor-plane cell indices (physical z = 25 and 70 mm)."""
    s = int(round(1e-3 / cfg["dx"]))
    return 25 * s, 70 * s


# Device library: name -> list of physical boxes (x0,x1,y0,y1,z0,z1, er).
DEVICES = {
    # full-cross-section slab: limiting case that must match the analytic slab
    "full_slab": [
        (0, 24, 0, 12, 40, 48, 4.0),
    ],
    # asymmetric side-loaded partial-width slab (full height)
    "side_slab_left": [
        (0, 12, 0, 12, 40, 48, 4.0),
    ],
    # mirror of the above, used for the symmetry check
    "side_slab_right": [
        (12, 24, 0, 12, 40, 48, 4.0),
    ],
    # symmetric centered partial-width slab (full height)
    "centered_slab": [
        (6, 18, 0, 12, 40, 48, 4.0),
    ],
    # isolated dielectric post, finite in all three dimensions
    "centered_post": [
        (8, 16, 3, 9, 40, 48, 4.0),
    ],
    # asymmetric corner block touching two walls
    "corner_block": [
        (0, 10, 0, 6, 40, 48, 4.0),
    ],
    # two-material composite slab
    "two_material": [
        (0, 12, 0, 12, 40, 48, 4.0),
        (12, 24, 0, 12, 40, 48, 2.0),
    ],
    # three centered partial-width teeth (grating)
    "grating": [
        (6, 18, 0, 12, 37, 40, 4.0),
        (6, 18, 0, 12, 43, 46, 4.0),
        (6, 18, 0, 12, 49, 52, 4.0),
    ],
    # two isolated posts with independently tunable er (2-parameter design)
    "two_post": [
        (4, 10, 3, 9, 40, 48, 4.0),
        (14, 20, 3, 9, 40, 48, 4.0),
    ],
    # anti-reflection matching structure: a fixed high-index obstacle (box 2)
    # sits between two tunable matching posts (boxes 0, 1). With the posts
    # absent (er=1) the obstacle reflects strongly; tuning the two post
    # permittivities cancels that reflection through quarter-wave interference,
    # giving a genuine interior optimum rather than a boundary solution.
    "ar_match": [
        (8, 16, 3, 9, 32, 38, 1.0),
        (8, 16, 3, 9, 62, 68, 1.0),
        (6, 18, 2, 10, 44, 56, 6.0),
    ],
}


def rasterize(boxes, cfg, er_override=None):
    """Rasterize physical boxes to a full 3-D er map.

    er_override: optional dict mapping the box order index to a replacement er
    (used to drive the multi-parameter differentiable design).
    """
    Nx, Ny, Nz = cfg["Nx"], cfg["Ny"], cfg["Nz"]
    dx = cfg["dx"]
    er = np.ones((Nx, Ny, Nz), dtype=np.float32)
    for bi, (x0, x1, y0, y1, z0, z1, erv) in enumerate(boxes):
        if er_override is not None and bi in er_override:
            erv = er_override[bi]
        # Clamp to interior cells: the PEC boundary is enforced in the wall
        # cells (0 and Nx-1 / Ny-1), so material is placed in interior cells
        # only. This is the discretization that converges to the analytic
        # dielectric-section model; writing er into a wall cell corrupts the
        # normal-field boundary update.
        i0 = max(1, int(round(x0 * 1e-3 / dx)))
        i1 = min(Nx - 1, int(round(x1 * 1e-3 / dx)))
        j0 = max(1, int(round(y0 * 1e-3 / dx)))
        j1 = min(Ny - 1, int(round(y1 * 1e-3 / dx)))
        k0, k1 = int(round(z0 * 1e-3 / dx)), int(round(z1 * 1e-3 / dx))
        er[i0:i1, j0:j1, k0:k1] = erv
    return er


def er_map(name, cfg, er_override=None):
    return rasterize(DEVICES[name], cfg, er_override)
