"""Finite-difference gradient check for the differentiable FDTD (L5)."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fdtd_gpu as fg  # noqa: E402
import fdtd_diff as fd  # noqa: E402


def make_problem():
    cfg = dict(Nx=24, Ny=20, Nz=40, dx=1e-3, dy=1e-3, dz=1e-3,
               dt=fg.cfl_dt(1e-3, 1e-3, 1e-3), f0=10e9, src_z=6, npml=6)
    geom = (8, 16, 6, 14, 12, 28)  # material block in the waveguide path
    return cfg, geom


def objective(er, cfg, geom, n_steps, use_ckpt):
    er_t = torch.tensor(er, dtype=torch.float32, device="cuda",
                        requires_grad=True)
    state, c = fd.forward(cfg, torch.device("cuda"), er_t, geom, n_steps,
                          chunk=15, use_ckpt=use_ckpt)
    Ex, Ey = state[0], state[1]
    # output-plane energy just before the far CPML
    k_out = cfg["Nz"] - cfg["npml"] - 2
    j = (Ey[:, :, k_out] ** 2 + Ex[:, :, k_out] ** 2).sum()
    return j, er_t


def main():
    cfg, geom = make_problem()
    n_steps = 60
    er0 = 3.5

    # Analytic gradient through the checkpointed graph.
    j, er_t = objective(er0, cfg, geom, n_steps, use_ckpt=True)
    j.backward()
    g_analytic = er_t.grad.item()

    # Central finite differences over a range of h (fp32 rounding dominates at
    # small h; truncation grows at large h). The analytic value should be
    # bracketed by the converging middle estimates.
    print(f"objective J(er={er0}) = {j.item():.8e}")
    print(f"analytic grad (checkpointed autograd): {g_analytic:.8e}")
    best = None
    for h in (1e-3, 5e-3, 1e-2, 2e-2, 5e-2):
        jp, _ = objective(er0 + h, cfg, geom, n_steps, use_ckpt=False)
        jm, _ = objective(er0 - h, cfg, geom, n_steps, use_ckpt=False)
        g_fd = (jp.item() - jm.item()) / (2 * h)
        rel = abs(g_analytic - g_fd) / max(abs(g_analytic), 1e-30)
        print(f"h={h:6.3f}  FD grad={g_fd: .8e}  rel discrepancy={rel:8.3e}")
        if best is None or rel < best[1]:
            best = (h, rel)
    print(f"best h={best[0]} rel discrepancy={best[1]:.3e}")
    print("RESULT:", "PASS" if best[1] < 1e-2 else "FAIL")


if __name__ == "__main__":
    main()
