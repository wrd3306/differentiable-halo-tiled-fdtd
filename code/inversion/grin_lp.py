"""Finer-pitch GRIN with low-frequency emphasis and multiple starts.

28 x-invariant 1-mm layers (28 design variables, 2 FDTD cells each) over the
same 28-mm region, optimised directly on the real FDTD. The objective adds an
explicit penalty on the 8-9 GHz bins that set the minimax floor. Three starts;
the best rows are saved. No surrogate, fitted, or single-mode model.
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "engine"))
sys.path.insert(0, HERE)
import pixel_design as px  # noqa: E402
import grin_design as gd  # noqa: E402

Z_REG0, Z_REG1 = 28.0, 56.0
NZ = 28
DZ_MM = (Z_REG1 - Z_REG0) / NZ
LOW = (0, 1)  # 8 and 9 GHz indices


def build(device):
    cfg, n_steps = px.pixel_config(record_ns=6.0)
    Nx, Ny, Nz = cfg["Nx"], cfg["Ny"], cfg["Nz"]
    base = np.ones((Nx, Ny, Nz), dtype=np.float32)
    k0 = max(1, int(round(Z_REG1 * 1e-3 / cfg["dx"])))
    k1 = min(Nz - 1, int(round((Z_REG1 + gd.SLAB_D) * 1e-3 / cfg["dx"])))
    base[1:Nx - 1, 1:Ny - 1, k0:k1] = gd.SLAB_ER
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
            torch.tensor(region, device=device))


def er_from_rows(rows, base, row_id, region):
    delta = rows[row_id] - 1.0
    delta = torch.where(region, delta, torch.zeros_like(delta))
    return base + delta[None, None, :]


def run(cfg, device, er, n_steps, chunk=20):
    return gd.run(cfg, device, er, n_steps, chunk=chunk)


def main():
    t = time.time()
    device = torch.device("cuda")
    cfg, n_steps, base, row_id, region = build(device)
    with torch.no_grad():
        empty = run(cfg, device, torch.ones_like(base), n_steps)

    best = None
    starts = (1.1, 1.3, 1.5)
    for si, s0 in enumerate(starts):
        rows = torch.full((NZ,), s0, device=device, requires_grad=True)
        opt = torch.optim.Adam([rows], lr=0.06)
        for it in range(70):
            er = er_from_rows(rows, base, row_id, region)
            acc = run(cfg, device, er, n_steps)
            r = ((acc[0] - empty[0]) / empty[0]).abs()
            lall = torch.mean(r ** 16) ** (1.0 / 16.0)
            llow = torch.mean(r[LOW[0]:LOW[1] + 1] ** 16) ** (1.0 / 16.0)
            loss = lall + 0.5 * llow
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                rows.clamp_(1.0, 2.0)
            if it % 10 == 0 or it == 69:
                print("start %d it %2d max %.4f lowmax %.4f mean %.4f" % (
                    si, it, r.max().item(),
                    r[LOW[0]:LOW[1] + 1].max().item(),
                    r.mean().item()), flush=True)
        with torch.no_grad():
            er = er_from_rows(rows, base, row_id, region)
            acc = run(cfg, device, er, n_steps)
            r = ((acc[0] - empty[0]) / empty[0]).abs()
            mx = r.max().item()
        if best is None or mx < best[0]:
            best = (mx, rows.detach().cpu().numpy().copy())
            print("  -> new best max %.4f" % mx, flush=True)
    print("best max %.4f" % best[0], flush=True)
    print("best rows", np.round(best[1], 3), flush=True)
    np.save(os.path.join(HERE, "_grinlp_rows.npy"), best[1])
    print("elapsed %.0f s" % (time.time() - t), flush=True)


if __name__ == "__main__":
    main()
