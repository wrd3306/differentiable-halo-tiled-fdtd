"""Physics check: L4 bf16 engine vs fp32 L0, judged on the global physical scale."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fdtd_gpu as fg  # noqa: E402
import fdtd_gpu_l4 as l4  # noqa: E402

PHYS = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
GLOBAL_TOL = 2e-2   # global-scale relative error budget
PROBE_TOL = 1e-2


def main():
    device = torch.device("cuda")
    cfg = dict(Nx=48, Ny=40, Nz=64, dx=1e-3, dy=1e-3, dz=1e-3,
               dt=fg.cfl_dt(1e-3, 1e-3, 1e-3), f0=10e9, src_z=12, npml=8)
    n_steps = 200

    f0, c0 = fg.build_state(cfg, device)
    f4, c4 = l4.build_state_l4(cfg, device)
    mod = l4.get_module()

    bounded = True
    abs_tracker = {k: 0.0 for k in PHYS}
    for n in range(n_steps):
        fg.step(f0, c0, n)
        l4.step(mod, f4, c4, n)
        if not all(torch.isfinite(f4[k].float()).all().item() for k in PHYS):
            bounded = False
            break
        for k in PHYS:
            abs_tracker[k] = max(abs_tracker[k],
                                 torch.abs(f4[k].float() - f0[k]).max().item())

    g = max(torch.abs(f0[k]).max().item() for k in PHYS)
    print(f"steps={n_steps} global physical scale g={g:.4e}")
    all_ok = bounded
    for k in PHYS:
        global_rel = abs_tracker[k] / g
        ok = global_rel < GLOBAL_TOL
        all_ok = all_ok and ok
        print(f"{k:4s} abs err={abs_tracker[k]:11.3e} global-rel={global_rel:11.3e} "
              f"{'ok' if ok else 'FAIL'}")

    probe = f0["Ey"][24, 20, :].abs().max().item()
    probe4 = f4["Ey"][24, 20, :].float().abs().max().item()
    pdiff = abs(probe - probe4) / max(probe, 1e-30)
    print(f"probe peak fp32={probe:.4e} bf16={probe4:.4e} rel diff={pdiff:.3e}")
    all_ok = all_ok and pdiff < PROBE_TOL
    print("all finite & bounded:", bounded)
    print("RESULT:", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
