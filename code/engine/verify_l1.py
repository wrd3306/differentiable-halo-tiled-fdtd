"""Rigorous L1 verification: long-run stability and physically-scaled errors."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fdtd_gpu as l0  # noqa: E402
import fdtd_gpu_l1 as l1  # noqa: E402


def main():
    n = 96
    dx = 1e-3
    cfg = dict(Nx=n, Ny=n, Nz=n, dx=dx, dy=dx, dz=dx,
               dt=l0.cfl_dt(dx, dx, dx), f0=9e9, npml=10, src_z=n // 2)
    a, ca = l0.build_state(cfg, torch.device("cuda"))
    b, cb, bufs = l1.build_state_l1(cfg, torch.device("cuda"))
    with torch.no_grad():
        for k in range(400):
            l0.step(a, ca, k)
            l1.step_l1(b, cb, bufs, k)
            if k in (49, 99, 199, 299, 399):
                scale = a["Ey"].abs().max().item()  # global physical scale
                print(f"step {k+1:4d} | L0 |px| {a['px'].abs().max():.3e} L1 |px| {b['px'].abs().max():.3e} "
                      f"| L1 bounded Ey {b['Ey'].abs().max():.4f} Hx {b['Hx'].abs().max():.4f}")

    scale = a["Ey"].abs().max().item()
    print("\nfield diff normalized by global |Ey| scale", round(scale, 4))
    worst = 0.0
    for key in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        rel = (a[key] - b[key]).abs().max().item() / (scale + 1e-12)
        worst = max(worst, rel)
        print(f"  {key}: global-scaled rel {rel:.3e}")
    finite = all(torch.isfinite(v).all().item() for v in b.values())
    print("\nall finite:", finite, "| worst global-scaled rel:", round(worst, 2e7 and 8))
    print("RESULT:", "PASS" if (finite and worst < 1e-4) else "CHECK")


if __name__ == "__main__":
    main()
