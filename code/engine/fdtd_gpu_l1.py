"""L1 optimized GPU FDTD: preallocated reusable buffers + in-place updates.

Same physics as fdtd_gpu.py (L0), but:
  - all 12 derivative buffers are allocated once and reused (no per-step
    allocation / no full-grid zeros_like);
  - interior stencil is written directly, only the thin boundary row is reset;
  - field updates are in-place (add_ / mul_ / addcmul_) so no output tensors
    and no temporary curl tensors are materialised.
No new software required; pure PyTorch.
"""
import numpy as np
import torch

import fdtd_gpu as fg


def build_state_l1(cfg, device, dtype=torch.float32, block=None):
    fields, const = fg.build_state(cfg, device, dtype=dtype, block=block)
    n = (cfg["Nx"], cfg["Ny"], cfg["Nz"])
    # Reusable derivative buffers (6 E-derivatives + 6 H-derivatives).
    bufs = {k: torch.zeros(n, device=device, dtype=dtype) for k in (
        "dEydz", "dExdz", "dEzdy", "dEzdx", "dEydx", "dExdy",
        "dHzdy", "dHydz", "dHxdz", "dHzdx", "dHydx", "dHxdy")}
    return fields, const, bufs


def step_l1(f, c, b, n):
    dt = c["dt"]
    dx, dy, dz = c["dx"], c["dy"], c["dz"]
    mh = dt / fg.MU0
    Ex, Ey, Ez = f["Ex"], f["Ey"], f["Ez"]
    Hx, Hy, Hz = f["Hx"], f["Hy"], f["Hz"]
    pe, px, phy, phx = f["pe"], f["px"], f["phy"], f["phx"]
    bz, cz = c["bz"], c["cz"]

    # ---- E derivatives (interior direct + thin boundary reset) ----
    b["dEydz"][:, :, :-1] = (Ey[:, :, 1:] - Ey[:, :, :-1]) / dz; b["dEydz"][:, :, -1] = 0
    b["dExdz"][:, :, :-1] = (Ex[:, :, 1:] - Ex[:, :, :-1]) / dz; b["dExdz"][:, :, -1] = 0
    b["dEzdy"][:, :-1, :] = (Ez[:, 1:, :] - Ez[:, :-1, :]) / dy; b["dEzdy"][:, -1, :] = 0
    b["dEzdx"][:-1, :, :] = (Ez[1:, :, :] - Ez[:-1, :, :]) / dx; b["dEzdx"][-1, :, :] = 0
    b["dEydx"][:-1, :, :] = (Ey[1:, :, :] - Ey[:-1, :, :]) / dx; b["dEydx"][-1, :, :] = 0
    b["dExdy"][:, :-1, :] = (Ex[:, 1:, :] - Ex[:, :-1, :]) / dy; b["dExdy"][:, -1, :] = 0

    # ---- CPML psi (in-place, no temp): p = bz*p + cz*d ----
    pe.mul_(bz).addcmul_(cz, b["dEydz"], value=1.0)
    px.mul_(bz).addcmul_(cz, b["dExdz"], value=1.0)

    # ---- H in-place update: H -= mh*curl, psi sign + ----
    Hx.add_(b["dEzdy"], alpha=-mh).add_(b["dEydz"], alpha=mh).add_(pe, alpha=mh)
    Hy.add_(b["dExdz"], alpha=-mh).add_(b["dEzdx"], alpha=mh).add_(px, alpha=-mh)
    Hz.add_(b["dEydx"], alpha=-mh).add_(b["dExdy"], alpha=mh)

    # ---- H derivatives ----
    b["dHzdy"][:, 1:, :] = (Hz[:, 1:, :] - Hz[:, :-1, :]) / dy; b["dHzdy"][:, 0, :] = 0
    b["dHydz"][:, :, 1:] = (Hy[:, :, 1:] - Hy[:, :, :-1]) / dz; b["dHydz"][:, :, 0] = 0
    b["dHxdz"][:, :, 1:] = (Hx[:, :, 1:] - Hx[:, :, :-1]) / dz; b["dHxdz"][:, :, 0] = 0
    b["dHzdx"][1:, :, :] = (Hz[1:, :, :] - Hz[:-1, :, :]) / dx; b["dHzdx"][0, :, :] = 0
    b["dHydx"][1:, :, :] = (Hy[1:, :, :] - Hy[:-1, :, :]) / dx; b["dHydx"][0, :, :] = 0
    b["dHxdy"][:, 1:, :] = (Hx[:, 1:, :] - Hx[:, :-1, :]) / dy; b["dHxdy"][:, 0, :] = 0

    # ---- E-side CPML psi (in-place) ----
    phy.mul_(bz).addcmul_(cz, b["dHydz"], value=1.0)
    phx.mul_(bz).addcmul_(cz, b["dHxdz"], value=1.0)

    # ---- E in-place update: Ex = Ce*Ex + De*curl (elementwise De via addcmul) ----
    De, Ce = c["De"], c["Ce"]
    Ex.mul_(Ce)
    Ex.addcmul_(De, b["dHzdy"], value=1.0).addcmul_(De, b["dHydz"], value=-1.0).addcmul_(De, phy, value=-1.0)
    Ey.mul_(Ce)
    Ey.addcmul_(De, b["dHxdz"], value=1.0).addcmul_(De, b["dHzdx"], value=-1.0).addcmul_(De, phx, value=1.0)
    Ez.mul_(Ce)
    Ez.addcmul_(De, b["dHydx"], value=1.0).addcmul_(De, b["dHxdy"], value=-1.0)

    # ---- source + PEC (in-place) ----
    sv = np.sin(2 * np.pi * c["f0"] * n * dt)
    Ey.addcmul_(De, c["jprof"], value=-sv * c["j_amp"])
    Ex.mul_(c["my"]); Ey.mul_(c["mx"]); Ez.mul_(c["mx"]).mul_(c["my"])


def run_l1(cfg, device, n_steps, dtype=torch.float32, block=None, warmup=5):
    import time
    fields, const, bufs = build_state_l1(cfg, device, dtype=dtype, block=block)
    with torch.no_grad():
        for n in range(warmup):
            step_l1(fields, const, bufs, n)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for n in range(n_steps):
            step_l1(fields, const, bufs, n + warmup)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t0, fields
