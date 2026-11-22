"""Differentiable functional 3-D FDTD with a general permittivity map.

The only extension over fdtd_diff is that Ce/De are built from a full,
differentiable 3-D relative-permittivity map rather than a single scalar er and
a block mask. The map is constructed elsewhere from fixed geometry masks and
several design parameters (e.g. two independent post permittivities), so each
parameter receives a gradient. The functional step, chunking and gradient
checkpointing of fdtd_diff are reused unchanged.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fdtd_gpu as fg  # noqa: E402
import fdtd_diff as fd  # noqa: E402


def build_constants_map(cfg, device, er_map):
    """fp32 constants; Ce/De depend differentiably on the full er map."""
    Nx, Ny, Nz = cfg["Nx"], cfg["Ny"], cfg["Nz"]
    n_pml = cfg["npml"]
    shape = (Nx, Ny, Nz)

    Ce = torch.ones(shape, device=device)            # lossless Ce = 1
    De = cfg["dt"] / (fg.EPS0 * er_map)             # cellwise, differentiable

    # CPML (identical to fdtd_diff / fdtd_gpu).
    m = 3
    sig_peak = 0.8 * (m + 1) / (fg.ETA0 * cfg["dz"])
    sz = torch.zeros(Nz)
    for k in range(n_pml):
        d = (n_pml - k) / n_pml
        sz[k] = sig_peak * d ** m
        sz[Nz - 1 - k] = sig_peak * d ** m
    sz2 = sz.view(1, 1, Nz).expand(Nx, Ny, Nz).to(device)
    bz = torch.exp(-sz2 / fg.EPS0 * cfg["dt"])
    cz = bz - 1.0

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


def forward_map(cfg, device, er_map, n_steps, chunk=20, use_ckpt=True,
                record_planes=None, gates=None):
    """Run the differentiable FDTD on a general er map (mirrors fd.forward)."""
    c = build_constants_map(cfg, device, er_map)
    state = fd.init_state(cfg, device)
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
                state, acc = torch.utils.checkpoint.checkpoint(
                    fd.run_chunk, state, c, n0, n1, geom, record, acc,
                    use_reentrant=False)
            else:
                state, acc = fd.run_chunk(state, c, n0, n1, geom, record, acc)
        elif use_ckpt:
            state = torch.utils.checkpoint.checkpoint(
                fd.run_chunk, state, c, n0, n1, geom, record,
                use_reentrant=False)
        else:
            state = fd.run_chunk(state, c, n0, n1, geom, record)
        n0 = n1
    if gates is None:
        return state, c, record
    acc_dict = {z: acc[i] for i, z in enumerate(plane_list)}
    return state, c, acc_dict
