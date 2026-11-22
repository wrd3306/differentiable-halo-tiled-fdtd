"""Field-by-field correctness check: L3 fused kernels vs L0 reference.

Each component is judged on its own scale; an absolute guard (relative to the
dominant physical-field scale) keeps near-zero auxiliary psi from producing
misleading large relative numbers.
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fdtd_gpu as fg  # noqa: E402
import fdtd_gpu_l3 as l3  # noqa: E402

PHYS = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
PSI = ["pe", "px", "phy", "phx"]
REL_TOL = 1e-4


def main():
    device = torch.device("cuda")
    cfg = dict(Nx=48, Ny=40, Nz=64, dx=1e-3, dy=1e-3, dz=1e-3,
               dt=fg.cfl_dt(1e-3, 1e-3, 1e-3), f0=10e9, src_z=12, npml=8)
    n_steps = 200

    f0, c0 = fg.build_state(cfg, device)
    f3, c3 = fg.build_state(cfg, device)
    mod = l3.get_module()

    abs_tracker = {k: 0.0 for k in PHYS + PSI}
    bounded = True
    for n in range(n_steps):
        fg.step(f0, c0, n)
        l3.step(mod, f3, c3, n)
        if not all(torch.isfinite(f3[k]).all().item() for k in PHYS + PSI):
            bounded = False
            break
        g = max(torch.abs(f0[k]).max().item() for k in PHYS)
        for k in PHYS + PSI:
            abs_tracker[k] = max(abs_tracker[k],
                                 torch.abs(f3[k] - f0[k]).max().item())

    g = max(torch.abs(f0[k]).max().item() for k in PHYS)
    abs_guard = REL_TOL * g
    print(f"steps={n_steps}  dominant physical scale g={g:.4e}  "
          f"absolute guard={abs_guard:.4e}")
    print(f"{'field':5s} {'abs err':>12s} {'self-scale':>12s} {'self-rel':>12s}  verdict")
    all_ok = bounded
    for k in PHYS + PSI:
        abs_err = abs_tracker[k]
        self_max = max(torch.abs(f0[k]).max().item(), 1e-30)
        self_rel = abs_err / self_max
        ok = (self_rel < REL_TOL) or (abs_err < abs_guard)
        all_ok = all_ok and ok
        print(f"{k:5s} {abs_err:12.4e} {self_max:12.4e} {self_rel:12.4e}  "
              f"{'ok' if ok else 'FAIL'}")
    print("all finite & bounded:", bounded)
    print("RESULT:", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
