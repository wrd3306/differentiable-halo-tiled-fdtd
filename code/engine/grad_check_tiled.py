"""Finite-difference gradient check for the differentiable halo-tiled step.

The fused kernel is fp32-only, so a manual central finite difference is used
with the project's established step size (h = 0.02), rather than torch's
double-precision gradcheck. A short multi-step trajectory is driven from the
zero state with the pulse source, a smooth scalar loss is formed from the
on-grid field, and the reverse-mode gradients are compared against the finite
difference for both the material (er) variables and one initial-state value.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import fdtd_diff as fd  # noqa: E402
import fdtd_diff3d as f3  # noqa: E402
import halo_tiled_step as hts  # noqa: E402

# Small grid (mm) with a pulse source, sized for a fast but coupled trajectory.
Nx, Ny, Nz = 16, 10, 48
dx = dy = dz = 1.0e-3
N_STEPS = 60
H = 0.02
device = torch.device("cuda")

# Three independent material cells in a small interior design region.
CELLS = [(6, 5, 20), (7, 5, 22), (8, 5, 24)]
MONITOR = (slice(2, Nx - 2), slice(1, Ny - 1), 30)


def make_cfg():
    return dict(Nx=Nx, Ny=Ny, Nz=Nz, dx=dx, dy=dy, dz=dz,
                dt=fd.fg.cfl_dt(dx, dy, dz), f0=10e9, src_z=5, npml=8,
                pulse=dict(t0=0.4e-9, tw=0.15e-9, fc=10e9))


def geom_for(cfg):
    return dict(dt=cfg["dt"], f0=cfg["f0"], idx=1 / dx, idy=1 / dy,
                idz=1 / dz, mh=cfg["dt"] / fd.fg.MU0, pulse=cfg["pulse"])


def loss_from(er_vars, init_ey=None):
    cfg = make_cfg()
    er = torch.ones((Nx, Ny, Nz), device=device)
    for (i, j, k), v in zip(CELLS, er_vars):
        er[i, j, k] = v
    c = f3.build_constants_map(cfg, device, er)
    state = list(fd.init_state(cfg, device))
    if init_ey is not None:
        state[1] = state[1].clone()
        state[1][4, 5, 10] = init_ey
    state = tuple(state)
    geom = geom_for(cfg)
    for n in range(N_STEPS):
        state = hts.step(state, c, n, geom)
    return (state[1][MONITOR] ** 2).sum()


def main():
    g = torch.Generator(device="cpu").manual_seed(11)
    er_vars = torch.empty(len(CELLS)).uniform_(1.2, 1.8, generator=g).to(device)
    er_vars.requires_grad_(True)
    init_ey = torch.tensor(0.05, device=device, requires_grad=True)

    loss = loss_from(er_vars, init_ey)
    g_er, g_init = torch.autograd.grad(loss, [er_vars, init_ey])

    # Central finite differences.
    fd_er = []
    for idx in range(len(CELLS)):
        vp = er_vars.detach().clone()
        vm = er_vars.detach().clone()
        vp[idx] += H
        vm[idx] -= H
        lp = loss_from(vp, init_ey.detach()).item()
        lm = loss_from(vm, init_ey.detach()).item()
        fd_er.append((lp - lm) / (2 * H))
    fd_er = torch.tensor(fd_er, device=device)

    lp = loss_from(er_vars.detach(), init_ey.detach() + H).item()
    lm = loss_from(er_vars.detach(), init_ey.detach() - H).item()
    fd_init = (lp - lm) / (2 * H)

    def rel(analytic, numeric):
        return (analytic - numeric).abs().max().item() / max(numeric.abs().max().item(), 1e-12)

    r_er = rel(g_er, fd_er)
    r_init = abs(g_init.item() - fd_init) / max(abs(fd_init), 1e-12)
    print(f"loss = {loss.item():.6e}")
    print("er analytic:", np.array(g_er.detach().cpu()))
    print("er finite-d:", np.array(fd_er.cpu()))
    print(f"er max relative error = {r_er:.3e}")
    print(f"init-state analytic = {g_init.item():.6e}, finite-d = {fd_init:.6e}, "
          f"relative = {r_init:.3e}")
    assert r_er < 1e-3 and r_init < 1e-3, "gradient check failed"
    print("PASS: halo-tiled reverse-mode gradients match finite differences.")


if __name__ == "__main__":
    main()
