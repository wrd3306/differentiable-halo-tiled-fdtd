"""Full-width sub-wavelength pixel AR with mirror symmetry.

The 96 physical binary cells (12 x 8) collapse to 48 independent design
variables by mirror symmetry about the guide centre line. Each cell is 2 mm
(vacuum or er = 2); the fixed obstacle is a full-width er = 1.45 slab.
All results are real full-wave solves; no surrogate or fitted model is used.
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

CELL_MM = 2.0
X0, X1 = 0.0, 24.0
Z0, Z1 = 36.0, 52.0
SLAB_ER = 1.45
SLAB_Z0, SLAB_Z1 = 52.0, 56.0
NX, NZ = 12, 8
NHALF = NX // 2
N_IND = NHALF * NZ
FREQS_HZ = np.array(px.FREQS_GHZ) * 1e9


def build(device):
    cfg, n_steps = px.pixel_config(record_ns=6.0)
    Nxg, Nyg, Nzg = cfg["Nx"], cfg["Ny"], cfg["Nz"]
    base = np.ones((Nxg, Nyg, Nzg), dtype=np.float32)
    k0 = max(1, int(round(SLAB_Z0 * 1e-3 / cfg["dx"])))
    k1 = min(Nzg - 1, int(round(SLAB_Z1 * 1e-3 / cfg["dx"])))
    base[1:Nxg - 1, 1:Nyg - 1, k0:k1] = SLAB_ER

    masks = np.zeros((NX, NZ, Nxg, Nyg, Nzg), dtype=np.float32)
    for iz in range(NZ):
        for ix in range(NX):
            cx0, cx1 = X0 + ix * CELL_MM, X0 + (ix + 1) * CELL_MM
            cz0, cz1 = Z0 + iz * CELL_MM, Z0 + (iz + 1) * CELL_MM
            i0 = max(1, int(round(cx0 * 1e-3 / cfg["dx"])))
            i1 = min(Nxg - 1, int(round(cx1 * 1e-3 / cfg["dx"])))
            j0, j1 = max(1, int(round(cz0 * 1e-3 / cfg["dx"]))), min(
                Nzg - 1, int(round(cz1 * 1e-3 / cfg["dx"])))
            masks[ix, iz, i0:i1, 1:Nyg - 1, j0:j1] = 1.0
    return (cfg, n_steps, torch.tensor(base, device=device),
            torch.tensor(masks, device=device))


def expand(theta_half):
    """Mirror the 48 independent values into 12 x 8 full densities."""
    half = theta_half.reshape(NHALF, NZ)
    full = torch.cat([half, torch.flip(half, dims=[0])], dim=0)
    return full.reshape(NX * NZ)


def er_map(theta_half, base, masks, beta):
    p = torch.sigmoid(beta * (expand(theta_half) - 0.5))
    p = (p / torch.sigmoid(torch.tensor(beta * 0.5))).clamp(0.0, 1.0)
    mflat = masks.reshape(NX * NZ, *masks.shape[2:])
    return base + (px.ER_DIELECTRIC - 1.0) * torch.einsum(
        "c,cxyz->xyz", p, mflat)


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
    cfg, n_steps, base, masks = build(device)
    with torch.no_grad():
        empty = run(cfg, device, torch.ones_like(base), n_steps)

    def loss_of(theta_half, beta):
        er = er_map(theta_half, base, masks, beta)
        acc = run(cfg, device, er, n_steps)
        s11 = (acc[0] - empty[0]) / empty[0]
        r = s11.abs()
        return torch.mean(r ** 16) ** (1.0 / 16.0), s11

    theta = torch.full((N_IND,), 0.5, device=device, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=0.10)
    schedule = [(0, 2.0), (20, 4.0), (40, 8.0), (55, 14.0)]
    for it in range(60):
        beta = 2.0
        for bs, bv in schedule:
            if it >= bs:
                beta = bv
        opt.zero_grad()
        loss, s11 = loss_of(theta, beta)
        loss.backward()
        opt.step()
        with torch.no_grad():
            theta.clamp_(0.0, 1.0)
        if it % 5 == 0 or it == 59:
            print("it %2d beta %4.1f L8 %.4f max %.4f" % (
                it, beta, loss.item(), s11.abs().max().item()), flush=True)

    # hard binary symmetric pattern
    with torch.no_grad():
        full_bits = (expand(theta) >= 0.5).to(torch.float32)
        mflat = masks.reshape(NX * NZ, *masks.shape[2:])
        er_b = base + (px.ER_DIELECTRIC - 1.0) * torch.einsum(
            "c,cxyz->xyz", full_bits, mflat)
        acc = run(cfg, device, er_b, n_steps)
        s11 = ((acc[0] - empty[0]) / empty[0]).detach().cpu().numpy()
    print("BINARY max|S11| %.4f per-freq %s" % (
        np.max(np.abs(s11)), np.round(np.abs(s11), 4)), flush=True)
    print(full_bits.reshape(NZ, NX).cpu().numpy().astype(int), flush=True)
    np.save(os.path.join(HERE, "_sym_bits.npy"), full_bits.cpu().numpy())
    print("elapsed %.0f s" % (time.time() - t), flush=True)


if __name__ == "__main__":
    main()
