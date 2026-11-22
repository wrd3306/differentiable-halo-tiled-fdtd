"""Finer-grid (0.25 mm) re-optimization of the 28-var GRIN on the real FDTD.

The 0.5-mm design lands at max |S11| ~ 0.11 at 0.25 mm with mean 0.085, so there
is headroom to re-balance the ripple. This optimises the same 28 x-invariant
1-mm layers directly on the 0.25-mm grid with discrete-adjoint L16 minimax,
using three starts. All evaluations are real full-wave solves; no surrogate,
fitted, single-mode, or interpolated model is used.
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "engine"))
sys.path.insert(0, HERE)
import grin_design as gd  # noqa: E402
import grin_lp as glp  # noqa: E402
import devices3d as d3  # noqa: E402
import fdtd_diff3d as f3  # noqa: E402
import fdtd_diff as fd  # noqa: E402
import pixel_design as px  # noqa: E402


def build(device):
    cfg = d3.config3d(4)
    cfg["pulse"] = dict(t0=0.8e-9, tw=0.15e-9, fc=10e9)
    n_steps = int(np.ceil(6.0 * 1e-9 / cfg["dt"]))
    Nx, Ny, Nz = cfg["Nx"], cfg["Ny"], cfg["Nz"]
    base = np.ones((Nx, Ny, Nz), dtype=np.float32)
    sk0 = max(1, int(round(glp.Z_REG1 * 1e-3 / cfg["dx"])))
    sk1 = min(Nz - 1, int(round((glp.Z_REG1 + gd.SLAB_D) * 1e-3 / cfg["dx"])))
    base[1:Nx - 1, 1:Ny - 1, sk0:sk1] = gd.SLAB_ER
    reg_k0 = max(1, int(round(glp.Z_REG0 * 1e-3 / cfg["dx"])))
    reg_k1 = min(Nz - 1, int(round(glp.Z_REG1 * 1e-3 / cfg["dx"])))
    row_id = np.zeros(Nz, dtype=np.int64)
    for k in range(reg_k0, reg_k1):
        z_mm = k * cfg["dx"] * 1e3
        rid = int((z_mm - glp.Z_REG0) / glp.DZ_MM)
        row_id[k] = min(glp.NZ - 1, max(0, rid))
    region = np.zeros(Nz, dtype=bool)
    region[reg_k0:reg_k1] = True
    return (cfg, n_steps, torch.tensor(base, device=device),
            torch.tensor(row_id, device=device),
            torch.tensor(region, device=device))


def er_from_rows(rows, base, row_id, region):
    delta = rows[row_id] - 1.0
    delta = torch.where(region, delta, torch.zeros_like(delta))
    return base + delta[None, None, :]


def run(cfg, device, er, n_steps, chunk=80):
    c = f3.build_constants_map(cfg, device, er)
    state = fd.init_state(cfg, device)
    planes = list(d3.planes(cfg))
    geom = dict(dt=cfg["dt"], f0=cfg["f0"], idx=1 / cfg["dx"],
                idy=1 / cfg["dy"], idz=1 / cfg["dz"],
                mh=cfg["dt"] / fd.fg.MU0, pulse=cfg.get("pulse"), cfg=cfg,
                planes=planes, freqs_t=torch.tensor(gd.FREQS_HZ, device=device))
    acc = torch.zeros((len(planes), len(gd.FREQS_HZ)),
                      dtype=torch.complex64, device=device)
    n0 = 0
    while n0 < n_steps:
        n1 = min(n0 + chunk, n_steps)
        state, acc = torch.utils.checkpoint.checkpoint(
            px.run_chunk_bb, state, c, n0, n1, geom, acc,
            use_reentrant=False)
        n0 = n1
    return acc


def main():
    t = time.time()
    device = torch.device("cuda")
    cfg, n_steps, base, row_id, region = build(device)
    with torch.no_grad():
        empty = run(cfg, device, torch.ones_like(base), n_steps)

    best = None
    for si, s0 in enumerate((1.1, 1.3)):
        rows = torch.full((glp.NZ,), s0, device=device, requires_grad=True)
        opt = torch.optim.Adam([rows], lr=0.06)
        for it in range(45):
            er = er_from_rows(rows, base, row_id, region)
            acc = run(cfg, device, er, n_steps)
            r = ((acc[0] - empty[0]) / empty[0]).abs()
            loss = torch.mean(r ** 16) ** (1.0 / 16.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                rows.clamp_(1.0, 2.0)
            if it % 9 == 0 or it == 44:
                print("start %d it %2d max %.4f mean %.4f" % (
                    si, it, r.max().item(), r.mean().item()), flush=True)
        with torch.no_grad():
            er = er_from_rows(rows, base, row_id, region)
            r = ((run(cfg, device, er, n_steps)[0] - empty[0])
                 / empty[0]).abs()
            mx = r.max().item()
        if best is None or mx < best[0]:
            best = (mx, rows.detach().cpu().numpy().copy())
            print("  -> new best max %.4f" % mx, flush=True)
    print("best max %.4f" % best[0], flush=True)
    print("best rows", np.round(best[1], 3), flush=True)
    np.save(os.path.join(HERE, "_grinfine_rows.npy"), best[1])
    print("elapsed %.0f s" % (time.time() - t), flush=True)


if __name__ == "__main__":
    main()
