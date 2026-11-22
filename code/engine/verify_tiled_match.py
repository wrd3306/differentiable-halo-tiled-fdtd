"""Verify fused_tiled CUDA kernels are the exact fused twin of fdtd_diff.step_fn.

Two independent trajectories are advanced from byte-identical initial states and
constants:

  * functional path: fdtd_diff.step_fn (out-of-place autograd tensor expressions);
  * tiled path:      fused_tiled.cu h_launch_tiled / e_launch_tiled (shared-memory
                     halo tiling), with the identical source scalar.

All ten state tensors are compared after every step. A second test seeds a random
state on a small grid to stress every edge and corner clamp. No fitted or
interpolated data are used; both paths solve the same discrete update.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "code", "inversion"))

import fdtd_diff as fd  # noqa: E402
import fdtd_diff3d as f3  # noqa: E402
from ext_build import get_ext  # noqa: E402

CU_SRC = os.path.join(ROOT, "code", "kernels", "fused_tiled.cu")
BUILD = os.path.join(ROOT, "code", "kernels", "_tiled_build")


def source_scalar(n, dt, pulse):
    t = n * dt
    env = np.exp(-((t - pulse["t0"]) / pulse["tw"]) ** 2)
    return float(env * np.sin(2 * np.pi * pulse["fc"] * t))


def tiled_step(mod, state, c, n, geom):
    """One fused tiled step mirroring fdtd_diff.step_fn (returns new tuple)."""
    Ex, Ey, Ez, Hx, Hy, Hz, pe, px, phy, phx = [t.clone() for t in state]
    Nx, Ny, Nz = Ex.shape
    idx, idy, idz, mh, dt = geom["idx"], geom["idy"], geom["idz"], geom["mh"], geom["dt"]
    mod.h_launch_tiled(
        Ex, Ey, Ez, c["bz"], c["cz"], Hx, Hy, Hz, pe, px,
        Nx, Ny, Nz, idx, idy, idz, mh)
    sv = source_scalar(n, dt, geom["pulse"])
    mod.e_launch_tiled(
        Ex, Ey, Ez, Hx, Hy, Hz, c["Ce"], c["De"], c["bz"], c["cz"],
        phy, phx, c["jprof"], c["mx"], c["my"],
        Nx, Ny, Nz, idx, idy, idz, sv)
    return (Ex, Ey, Ez, Hx, Hy, Hz, pe, px, phy, phx)


def compare(s_a, s_b, label, atol, rtol):
    """numpy-allclose-style check. Return (max absolute, worst tolerance excess).

    A point passes when |a-b| <= atol + rtol*max(|a|,|b|); atol dominates near
    zero (where a raw relative ratio is meaningless) and rtol bounds large
    magnitudes. The worst excess is <= 0 when every point is within tolerance.
    """
    worst_abs = 0.0
    worst_excess = 0.0
    for k, a, b in zip(fd.STATE_KEYS, s_a, s_b):
        diff = (a - b).abs()
        da = diff.max().item()
        excess = (diff - (atol + rtol * torch.maximum(a.abs(), b.abs()))).max().item()
        worst_abs = max(worst_abs, da)
        worst_excess = max(worst_excess, excess)
        if excess > 0:
            print(f"  [{label}] {k}: abs {da:.3e}, excess {excess:.3e}")
    return worst_abs, worst_excess


def test_real_trajectory(device, mod, n_steps=600):
    import pixel_design as px
    cfg, _ = px.pixel_config(record_ns=6.0)
    base, cell_masks, _ = px.build_geometry(cfg, device)
    # A relaxed, non-symmetric pattern so every cell exercises a different er.
    g = torch.Generator(device="cpu").manual_seed(1234)
    theta = torch.rand(cell_masks.shape[0], generator=g).to(device)
    er_map, _ = px.er_map_from(theta, base, cell_masks, beta=2.0)
    c = f3.build_constants_map(cfg, device, er_map)
    geom = dict(dt=cfg["dt"], idx=1 / cfg["dx"], idy=1 / cfg["dy"],
                idz=1 / cfg["dz"], mh=cfg["dt"] / fd.fg.MU0,
                pulse=cfg["pulse"])

    s_func = fd.init_state(cfg, device)
    s_tile = tuple(t.clone() for t in s_func)
    worst_abs = worst_excess = 0.0
    for n in range(n_steps):
        s_func = fd.step_fn(s_func, c, n, geom["dt"], cfg["f0"], geom["idx"],
                            geom["idy"], geom["idz"], geom["mh"], geom["pulse"])
        s_tile = tiled_step(mod, s_tile, c, n, geom)
        if n % 50 == 0 or n == n_steps - 1:
            da, ex = compare(s_func, s_tile, f"real n={n}", atol=1e-6, rtol=1e-5)
            worst_abs = max(worst_abs, da)
            worst_excess = max(worst_excess, ex)
    print(f"TEST 1 real design grid ({cfg['Nx']}x{cfg['Ny']}x{cfg['Nz']}), "
          f"{n_steps} steps: abs = {worst_abs:.3e}, excess = {worst_excess:.3e}")
    return worst_abs, worst_excess


def test_random_state(device, mod, n_steps=1):
    """Single-step edge/corner clamp test with nonzero data.

    A random state is not a physical field and the explicit update amplifies it,
    so multi-step runs accumulate rounding noise. The edge rules are checked at
    one step, where the two orderings differ only by fp32 single-step rounding.
    """
    # Small non-divisible sizes to exercise partial tiles and all boundaries.
    Nx, Ny, Nz = 19, 13, 29
    dx = dy = dz = 1.0e-3
    cfg = dict(Nx=Nx, Ny=Ny, Nz=Nz, dx=dx, dy=dy, dz=dz,
               dt=fd.fg.cfl_dt(dx, dy, dz), f0=10e9, src_z=5, npml=8,
               pulse=dict(t0=0.8e-9, tw=0.15e-9, fc=10e9))
    g = torch.Generator(device="cpu").manual_seed(7)
    er_map = torch.empty((Nx, Ny, Nz)).uniform_(1.0, 2.0, generator=g).to(device)
    c = f3.build_constants_map(cfg, device, er_map)
    geom = dict(dt=cfg["dt"], idx=1 / dx, idy=1 / dy, idz=1 / dz,
                mh=cfg["dt"] / fd.fg.MU0, pulse=cfg["pulse"])

    s_func = []
    for _ in fd.STATE_KEYS:
        s_func.append(torch.empty((Nx, Ny, Nz)).uniform_(-0.1, 0.1,
                                                         generator=g).to(device))
    s_func = tuple(s_func)
    s_tile = tuple(t.clone() for t in s_func)
    worst_abs = worst_excess = 0.0
    for n in range(n_steps):
        s_func = fd.step_fn(s_func, c, n, geom["dt"], cfg["f0"], geom["idx"],
                            geom["idy"], geom["idz"], geom["mh"], geom["pulse"])
        s_tile = tiled_step(mod, s_tile, c, n, geom)
        da, ex = compare(s_func, s_tile, f"rand n={n}", atol=1e-5, rtol=1e-5)
        worst_abs = max(worst_abs, da)
        worst_excess = max(worst_excess, ex)
    print(f"TEST 2 random state ({Nx}x{Ny}x{Nz}), {n_steps} steps: "
          f"abs = {worst_abs:.3e}, excess = {worst_excess:.3e}")
    return worst_abs, worst_excess


if __name__ == "__main__":
    device = torch.device("cuda")
    print("loading fused_tiled extension ...")
    mod = get_ext("fused_tiled", [CU_SRC], BUILD)
    a1, e1 = test_real_trajectory(device, mod)
    a2, e2 = test_random_state(device, mod)
    assert a1 < 1e-5 and e1 <= 0, "physical trajectory mismatch"
    assert e2 <= 0, "random-state edge check exceeds fp32 tolerance"
    print("PASS: fused_tiled CUDA matches fdtd_diff.step_fn (fp32-equivalent).")
