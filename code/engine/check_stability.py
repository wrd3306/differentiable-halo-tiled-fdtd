"""Physics sanity check for the GPU engine: finite, bounded, propagating wave.

Runs a moderate vacuum simulation with a soft source and records a probe trace
behind the source plane, plus global field statistics. Confirms the time loop is
stable (no NaN/Inf, bounded energy) and that a wave reaches the probe.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fdtd_gpu as fg  # noqa: E402


def main():
    n = 64
    dx = 1e-3
    cfg = dict(Nx=n, Ny=n, Nz=n, dx=dx, dy=dx, dz=dx,
               dt=fg.cfl_dt(dx, dx, dx), f0=9e9, npml=10, src_z=n // 2)
    fields, const = fg.build_state(cfg, torch.device("cuda"))

    probe = (n // 2, n // 2, n // 2 + 8)
    trace = []
    n_steps = 300
    with torch.no_grad():
        for k in range(n_steps):
            fg.step(fields, const, k)
            trace.append(float(fields["Ey"][probe].item()))
    trace = np.array(trace)

    all_finite = all(torch.isfinite(v).all().item() for v in fields.values())
    print("all fields finite:", all_finite)
    print("probe max |Ey|:", round(float(np.abs(trace).max()), 4))
    print("probe first nonzero step:", int(np.argmax(np.abs(trace) > 1e-4)))
    print("global max |Ey|:", round(float(fields["Ey"].abs().max()), 4))
    print("global mean |Ey|:", round(float(fields["Ey"].abs().mean()), 5))
    # A probe 8 cells beyond the source should see a growing then bounded signal.
    reached = bool(np.abs(trace[120:]).max() > 1e-2)
    bounded = bool(np.isfinite(trace).all() and np.abs(trace).max() < 5.0)
    print("wave reached probe:", reached, "| bounded:", bounded)
    print("RESULT:", "PASS" if (all_finite and reached and bounded) else "FAIL")


if __name__ == "__main__":
    main()
