"""Differentiable functional 3-D FDTD (L5).

All updates are out-of-place functional tensor expressions (no in-place ops),
which sidesteps the historical in-place autograd failure. Time is processed in
chunks wrapped with gradient checkpointing so memory stays bounded. The design
parameter is the relative permittivity er of a fixed material block; Ce/De are
built differentiably from er.
"""
import os
import sys

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fdtd_gpu as fg  # noqa: E402

STATE_KEYS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "pe", "px", "phy", "phx")


# ---- functional differences (fp32), pads reproduce L0 edge rules ----
def fwd_z(E, s):
    return torch.cat([(E[:, :, 1:] - E[:, :, :-1]) * s,
                      torch.zeros_like(E[:, :, :1])], dim=2)


def fwd_y(E, s):
    return torch.cat([(E[:, 1:, :] - E[:, :-1, :]) * s,
                      torch.zeros_like(E[:, :1, :])], dim=1)


def fwd_x(E, s):
    return torch.cat([(E[1:, :, :] - E[:-1, :, :]) * s,
                      torch.zeros_like(E[:1, :, :])], dim=0)


def bwd_z(E, s):
    return torch.cat([torch.zeros_like(E[:, :, :1]),
                      (E[:, :, 1:] - E[:, :, :-1]) * s], dim=2)


def bwd_y(E, s):
    return torch.cat([torch.zeros_like(E[:, :1, :]),
                      (E[:, 1:, :] - E[:, :-1, :]) * s], dim=1)


def bwd_x(E, s):
    return torch.cat([torch.zeros_like(E[:1, :, :]),
                      (E[1:, :, :] - E[:-1, :, :]) * s], dim=0)


def build_constants(cfg, device, er, block_geom):
    """fp32 constants; Ce/De depend differentiably on er."""
    Nx, Ny, Nz = cfg["Nx"], cfg["Ny"], cfg["Nz"]
    n_pml = cfg["npml"]
    shape = (Nx, Ny, Nz)
    eps_abs = fg.EPS0 * er
    # block mask (fixed geometry)
    x0, x1, y0, y1, z0, z1 = block_geom
    mask = torch.zeros(shape, device=device)
    mask[x0:x1, y0:y1, z0:z1] = 1.0

    Ce_vac = torch.ones(shape, device=device)
    De_vac = torch.full(shape, cfg["dt"] / fg.EPS0, device=device)
    # lossless block (sigma=0): Ce_b=1, De_b = dt/(EPS0*er)
    De_b = cfg["dt"] / eps_abs
    Ce = Ce_vac  # lossless => Ce=1 everywhere
    De = De_vac * (1 - mask) + De_b * mask

    # CPML grading (same as L0): Taflove optimal cubic sigma, kappa=1, alpha=0.
    m = 3
    sig_peak = 0.8 * (m + 1) / (fg.ETA0 * cfg["dz"])
    sz = torch.zeros(Nz)
    for k in range(n_pml):
        d = (n_pml - k) / n_pml
        sz[k] = sig_peak * d ** m
        sz[Nz - 1 - k] = sig_peak * d ** m
    sz2 = sz.view(1, 1, Nz).expand(Nx, Ny, Nz).to(device)
    bz = torch.exp(-sz2 / fg.EPS0 * cfg["dt"])
    cz = bz - 1.0  # kappa=1, alpha=0 => pure-conductance limit

    mx = torch.ones(shape, device=device)
    mx[0, :, :] = 0
    mx[-1, :, :] = 0
    my = torch.ones(shape, device=device)
    my[:, 0, :] = 0
    my[:, -1, :] = 0

    xi = torch.arange(1, Nx - 1, dtype=torch.float32)
    jprof = torch.zeros(shape, device=device)
    jprof[1:Nx - 1, 1:Ny - 1, cfg["src_z"]] = torch.sin(
        np.pi * xi * cfg["dx"] / (Nx * cfg["dx"])).unsqueeze(1)
    return dict(Ce=Ce, De=De, bz=bz, cz=cz, mx=mx, my=my, jprof=jprof)


def init_state(cfg, device):
    shape = (cfg["Nx"], cfg["Ny"], cfg["Nz"])
    return tuple(torch.zeros(shape, device=device) for _ in STATE_KEYS)


def step_fn(state, c, n, dt, f0, idx, idy, idz, mh, pulse=None):
    Ex, Ey, Ez, Hx, Hy, Hz, pe, px, phy, phx = state
    dEydz = fwd_z(Ey, idz); dExdz = fwd_z(Ex, idz)
    dEzdy = fwd_y(Ez, idy); dEzdx = fwd_x(Ez, idx)
    dEydx = fwd_x(Ey, idx); dExdy = fwd_y(Ex, idy)

    pe_n = c["bz"] * pe + c["cz"] * dEydz
    px_n = c["bz"] * px + c["cz"] * dExdz
    Hx_n = Hx - mh * (dEzdy - dEydz - pe_n)
    Hy_n = Hy - mh * (dExdz + px_n - dEzdx)
    Hz_n = Hz - mh * (dEydx - dExdy)

    dHzdy = bwd_y(Hz_n, idy); dHxdy = bwd_y(Hx_n, idy)
    dHydz = bwd_z(Hy_n, idz); dHxdz = bwd_z(Hx_n, idz)
    dHzdx = bwd_x(Hz_n, idx); dHydx = bwd_x(Hy_n, idx)

    phy_n = c["bz"] * phy + c["cz"] * dHydz
    phx_n = c["bz"] * phx + c["cz"] * dHxdz
    Ex_n = c["Ce"] * Ex + c["De"] * (dHzdy - dHydz - phy_n)
    Ey_n = c["Ce"] * Ey + c["De"] * (dHxdz + phx_n - dHzdx)
    Ez_n = c["Ce"] * Ez + c["De"] * (dHydx - dHxdy)

    t = n * dt
    if pulse is not None:
        env = np.exp(-((t - pulse["t0"]) / pulse["tw"]) ** 2)
        sv = env * np.sin(2 * np.pi * pulse["fc"] * t)
    else:
        sv = np.sin(2 * np.pi * f0 * t)
    Ey_n = Ey_n - c["De"] * float(sv) * c["jprof"]
    Ex_n = Ex_n * c["my"]
    Ey_n = Ey_n * c["mx"]
    Ez_n = Ez_n * c["mx"] * c["my"]
    return (Ex_n, Ey_n, Ez_n, Hx_n, Hy_n, Hz_n, pe_n, px_n, phy_n, phx_n)


def _modal_amp(Ey, cfg, z):
    """TE10 modal amplitude on plane z (matches sparam.modal_amp)."""
    Nx, dx = cfg["Nx"], cfg["dx"]
    a = Nx * dx
    xi = torch.arange(1, Nx - 1, device=Ey.device)
    p = torch.sin(np.pi * xi * dx / a)
    prof = Ey[1:Nx - 1, :, z].mean(dim=1)
    return (2.0 / a) * torch.sum(prof * p) * dx


def run_chunk(state, c, n0, n1, geom, record=None, acc=None):
    """Pure function: never mutates dict inputs.

    acc is either None or a tuple of complex scalar phasors aligned with the
    sorted gate planes in geom['_planes']. Returns (state,) or (state, acc).
    """
    dt, f0 = geom["dt"], geom["f0"]
    idx, idy, idz = 1 / geom["dx"], 1 / geom["dy"], 1 / geom["dz"]
    mh = dt / fg.MU0
    pulse = geom.get("pulse")
    gates = geom.get("gates")
    if acc is not None:
        planes = geom["_planes"]
        new_acc = list(acc)
    for n in range(n0, n1):
        state = step_fn(state, c, n, dt, f0, idx, idy, idz, mh, pulse)
        if record is not None:
            for z, buf in record.items():
                buf.append(_modal_amp(state[1], geom["cfg"], z))
        if acc is not None and gates is not None:
            for i, z in enumerate(planes):
                s0, s1, win, ker = gates[z]
                if s0 <= n <= s1:
                    a = _modal_amp(state[1], geom["cfg"], z)
                    new_acc[i] = new_acc[i] + a * win[n - s0] * ker[n - s0] * dt
    if acc is None:
        return state
    return state, tuple(new_acc)


def forward(cfg, device, er, block_geom, n_steps, chunk=20, use_ckpt=True,
            record_planes=None, gates=None):
    """Run the differentiable FDTD.

    record_planes: retain the modal time series (graph tensors) for diagnostics.
    gates: {plane: (s0, s1, window, kernel)} to accumulate gated phasors online
    with bounded memory (no per-step graph retention). The phasor accumulator is
    a pure tuple threaded through checkpoint, so no mutable container is
    aliased across checkpoint boundaries.
    """
    er_t = er.to(device) if torch.is_tensor(er) else torch.tensor(
        er, device=device, dtype=torch.float32)
    c = build_constants(cfg, device, er_t, block_geom)
    state = init_state(cfg, device)
    plane_list = sorted(gates) if gates else None
    geom = dict(dt=cfg["dt"], f0=cfg["f0"], dx=cfg["dx"],
                dy=cfg["dy"], dz=cfg["dz"], pulse=cfg.get("pulse"), cfg=cfg,
                gates=gates, _planes=plane_list)
    record = {z: [] for z in record_planes} if record_planes else None
    acc = (tuple(torch.tensor(0j, device=device) for _ in plane_list)
           if gates else None)
    n0 = 0
    while n0 < n_steps:
        n1 = min(n0 + chunk, n_steps)
        if gates is not None:
            if use_ckpt:
                state, acc = checkpoint(run_chunk, state, c, n0, n1, geom,
                                        record, acc, use_reentrant=False)
            else:
                state, acc = run_chunk(state, c, n0, n1, geom, record, acc)
        elif use_ckpt:
            state = checkpoint(run_chunk, state, c, n0, n1, geom, record,
                               use_reentrant=False)
        else:
            state = run_chunk(state, c, n0, n1, geom, record)
        n0 = n1
    if gates is None:
        return state, c, record
    acc_dict = {z: acc[i] for i, z in enumerate(plane_list)}
    return state, c, acc_dict
