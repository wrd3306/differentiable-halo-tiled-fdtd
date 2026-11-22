"""Long graded-index (GRIN) full-width AR, optimised directly on the real FDTD.

Each of the 14 x-invariant 2-mm layers is a real homogeneous er variable
(14 design variables, in the 10-50 range). The fixed obstacle is a full-width
er = 1.45 slab. All objective evaluations are real full-wave solves; no
surrogate, fitted, or single-mode transfer-matrix model is used.
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "engine"))
sys.path.insert(0, HERE)
import devices3d as d3  # noqa: E402
import fdtd_diff3d as f3  # noqa: E402
import fdtd_diff as fd  # noqa: E402
import pixel_design as px  # noqa: E402

Z_REG0, Z_REG1 = 28.0, 56.0
SLAB_ER, SLAB_D = 1.45, 4.0
NZ = 14
DZ_MM = (Z_REG1 - Z_REG0) / NZ
FREQS_HZ = np.array(px.FREQS_GHZ) * 1e9


def build(device):
    cfg, n_steps = px.pixel_config(record_ns=6.0)
    Nx, Ny, Nz = cfg["Nx"], cfg["Ny"], cfg["Nz"]
    base = np.ones((Nx, Ny, Nz), dtype=np.float32)
    k0 = max(1, int(round(Z_REG1 * 1e-3 / cfg["dx"])))
    k1 = min(Nz - 1, int(round((Z_REG1 + SLAB_D) * 1e-3 / cfg["dx"])))
    base[1:Nx - 1, 1:Ny - 1, k0:k1] = SLAB_ER

    # integer row id for every design-region voxel (differentiable gather key)
    reg_k0 = max(1, int(round(Z_REG0 * 1e-3 / cfg["dx"])))
    reg_k1 = min(Nz - 1, int(round(Z_REG1 * 1e-3 / cfg["dx"])))
    row_id = np.zeros(Nz, dtype=np.int64)
    for k in range(reg_k0, reg_k1):
        z_mm = k * cfg["dx"] * 1e3
        rid = int((z_mm - Z_REG0) / DZ_MM)
        row_id[k] = min(NZ - 1, max(0, rid))
    region = np.zeros(Nz, dtype=bool)
    region[reg_k0:reg_k1] = True
    return (cfg, n_steps, torch.tensor(base, device=device),
            torch.tensor(row_id, device=device),
            torch.tensor(region, device=device), reg_k0, reg_k1)


def er_from_rows(rows, base, row_id, region):
    """Differentiable er map: each design voxel takes its row er via gather."""
    delta = rows[row_id] - 1.0                 # (Nz,)
    delta = torch.where(region, delta, torch.zeros_like(delta))
    return base + delta[None, None, :]         # broadcast over x, y


def run(cfg, device, er, n_steps, chunk=20):
    c = f3.build_constants_map(cfg, device, er)
    state = fd.init_state(cfg, device)
    planes = list(d3.planes(cfg))
    geom = dict(dt=cfg["dt"], f0=cfg["f0"], idx=1 / cfg["dx"],
                idy=1 / cfg["dy"], idz=1 / cfg["dz"],
                mh=cfg["dt"] / fd.fg.MU0, pulse=cfg.get("pulse"), cfg=cfg,
                planes=planes, freqs_t=torch.tensor(FREQS_HZ, device=device))
    acc = torch.zeros((len(planes), len(FREQS_HZ)), dtype=torch.complex64,
                      device=device)
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
    cfg, n_steps, base, row_id, region, _, _ = build(device)
    with torch.no_grad():
        empty = run(cfg, device, torch.ones_like(base), n_steps)

    rows = torch.full((NZ,), 1.2, device=device, requires_grad=True)
    opt = torch.optim.Adam([rows], lr=0.08)
    for it in range(50):
        er = er_from_rows(rows, base, row_id, region)
        acc = run(cfg, device, er, n_steps)
        s11 = ((acc[0] - empty[0]) / empty[0])
        r = s11.abs()
        loss = torch.mean(r ** 16) ** (1.0 / 16.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            rows.clamp_(1.0, 2.0)
        if it % 5 == 0 or it == 49:
            print("it %2d L16 %.4f max %.4f mean %.4f" % (
                it, loss.item(), r.max().item(), r.mean().item()), flush=True)
    print("final rows er", np.round(rows.detach().cpu().numpy(), 3), flush=True)
    np.save(os.path.join(HERE, "_grin_rows.npy"),
            rows.detach().cpu().numpy())
    print("elapsed %.0f s" % (time.time() - t), flush=True)


if __name__ == "__main__":
    main()
