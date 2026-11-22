"""Confirm the functional differentiable forward matches the L0 engine fields."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fdtd_gpu as fg  # noqa: E402
import fdtd_diff as fd  # noqa: E402


def main():
    device = torch.device("cuda")
    cfg = dict(Nx=32, Ny=24, Nz=48, dx=1e-3, dy=1e-3, dz=1e-3,
               dt=fg.cfl_dt(1e-3, 1e-3, 1e-3), f0=10e9, src_z=8, npml=8)
    geom = (10, 22, 8, 16, 16, 32)
    n_steps = 120

    # Reference L0 with a material block.
    block = ("box", geom, 3.5, 0.0)
    f0, c0 = fg.build_state(cfg, device, block=block)
    with torch.no_grad():
        for n in range(n_steps):
            fg.step(f0, c0, n)

    state, c = fd.forward(cfg, device, torch.tensor(3.5), geom, n_steps,
                          chunk=20, use_ckpt=False)
    keys = fd.STATE_KEYS
    g = max(torch.abs(f0[k]).max().item() for k in ("Ex", "Ey", "Ez"))
    print(f"steps={n_steps} global scale g={g:.4e}")
    all_ok = True
    for i, k in enumerate(keys[:6]):
        abs_err = torch.abs(state[i] - f0[k]).max().item()
        rel = abs_err / g
        ok = rel < 1e-5
        all_ok = all_ok and ok
        print(f"{k:4s} abs={abs_err:11.3e} global-rel={rel:11.3e} "
              f"{'ok' if ok else 'FAIL'}")
    print("RESULT:", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
