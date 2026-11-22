"""Tiled differentiable FDTD engine.

Drop-in counterpart of the broadband differentiable path in which every time
step is the differentiable halo-tiled kernel (halo_tiled_step.HaloTiledStep):
the forward runs the shared-memory CUDA kernel and the reverse pass is its
discrete adjoint. Functional tuple state, gradient checkpointing and the online
gated DFT are otherwise identical to the production path, so the same design
geometry and objective can be reused.
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "code", "inversion"))

import fdtd_diff as fd  # noqa: E402
import fdtd_diff3d as f3  # noqa: E402
import devices3d as d3  # noqa: E402
import halo_tiled_step as hts  # noqa: E402


def _geom(cfg, device, freqs_hz):
    planes = list(d3.planes(cfg))
    return dict(dt=cfg["dt"], f0=cfg["f0"], idx=1 / cfg["dx"],
                idy=1 / cfg["dy"], idz=1 / cfg["dz"],
                mh=cfg["dt"] / fd.fg.MU0, pulse=cfg.get("pulse"), cfg=cfg,
                planes=planes,
                freqs_t=torch.tensor(freqs_hz, device=device)), planes


def run_chunk_tiled(state, c, n0, n1, geom, acc):
    """Pure chunk: halo-tiled steps + multi-frequency online DFT at two planes."""
    dt = geom["dt"]
    freqs = geom["freqs_t"]
    planes = geom["planes"]
    for n in range(n0, n1):
        state = hts.step(state, c, n, geom)
        amps = torch.stack([fd._modal_amp(state[1], geom["cfg"], z)
                            for z in planes])
        ker = torch.exp(1j * 2 * np.pi * freqs * (n * dt))
        acc = acc + amps[:, None].to(torch.complex64) * ker[None, :] * dt
    return state, acc


def forward_broadband_tiled(cfg, device, er_map, n_steps, freqs_hz, chunk=20,
                            use_ckpt=True):
    """Run the tiled differentiable FDTD; return final state and DFT phasors."""
    c = f3.build_constants_map(cfg, device, er_map)
    state = fd.init_state(cfg, device)
    geom, _ = _geom(cfg, device, freqs_hz)
    acc = torch.zeros((len(geom["planes"]), len(freqs_hz)),
                      dtype=torch.complex64, device=device)
    n0 = 0
    while n0 < n_steps:
        n1 = min(n0 + chunk, n_steps)
        if use_ckpt:
            state, acc = torch.utils.checkpoint.checkpoint(
                run_chunk_tiled, state, c, n0, n1, geom, acc,
                use_reentrant=False)
        else:
            state, acc = run_chunk_tiled(state, c, n0, n1, geom, acc)
        n0 = n1
    return state, acc


def benchmark(cfg, device, er_map, n_steps, freqs_hz, chunk=20):
    """Measure forward-only and forward+backward throughput on the design grid."""
    n_cells = cfg["Nx"] * cfg["Ny"] * cfg["Nz"]
    c = f3.build_constants_map(cfg, device, er_map)
    geom, _ = _geom(cfg, device, freqs_hz)

    # ---- forward-only (compiled tiled kernels, no graph) ----
    state = fd.init_state(cfg, device)
    for n in range(5):
        hts.step(state, c, n, geom)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    state = fd.init_state(cfg, device)
    for n in range(n_steps):
        hts.step(state, c, n, geom)
    torch.cuda.synchronize()
    fwd_time = time.perf_counter() - t0
    fwd_rate = n_steps * n_cells / fwd_time / 1e6

    # ---- forward + backward (one differentiable solve with a scalar loss) ----
    torch.cuda.reset_peak_memory_stats()
    er_d = er_map.detach().clone().requires_grad_(True)
    t0 = time.perf_counter()
    state = fd.init_state(cfg, device)
    cc = f3.build_constants_map(cfg, device, er_d)
    acc = torch.zeros((len(geom["planes"]), len(freqs_hz)),
                      dtype=torch.complex64, device=device)
    n0 = 0
    while n0 < n_steps:
        n1 = min(n0 + chunk, n_steps)
        state, acc = torch.utils.checkpoint.checkpoint(
            run_chunk_tiled, state, cc, n0, n1, geom, acc, use_reentrant=False)
        n0 = n1
    loss = (acc.abs() ** 2).sum()
    loss.backward()
    torch.cuda.synchronize()
    fb_time = time.perf_counter() - t0
    fb_rate = n_steps * n_cells / fb_time / 1e6
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    return dict(n_cells=n_cells, n_steps=n_steps,
                fwd_time=fwd_time, fwd_mcells=fwd_rate,
                fb_time=fb_time, fb_mcells=fb_rate,
                step_ms=fwd_time / n_steps * 1e3, peak_gb=peak_mem)
