"""Independent validation of the dielectric-window optimization result.

The FDTD differentiable optimizer (run_optimization.py) returns er_star. This
script validates it WITHOUT an inverse crime:
  1. a fresh differentiable-FDTD run at er_star (rebuilt gates, independent run);
  2. the independent analytic Picture B boundary-value solver at er_star;
  3. the analytic er that actually yields |S21|=target, found by a bounded
     bisection on the analytic model (a deterministic root, not a data fit).
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.join(HERE, "..", "engine")
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ENG)
sys.path.insert(0, HERE)
import optimize as op  # noqa: E402
import sparam as sp  # noqa: E402


def analytic_abs_s21(er, cfg, geom):
    _, t = sp.analytic_sparams(er, cfg, geom)
    return abs(t)


def main(n_steps=3000, target=0.95):
    device = torch.device("cuda")
    with open(os.path.join(ROOT, "results", "device_optimization.json"),
              "r", encoding="utf-8") as fh:
        res = json.load(fh)
    er_star = res["er_star"]
    cfg, geom = op.scaled_config(2)

    print("=== 1. fresh differentiable FDTD at er_star=%.4f ===" % er_star)
    gates = op.build_gates(cfg, geom, n_steps, device)
    s11, s21 = op.run_sparams(torch.tensor(er_star), cfg, geom, n_steps,
                              device, gates)
    fdtd_s21, fdtd_s11 = abs(s21), abs(s11)
    print("FDTD |S21| %.5f |S11| %.5f (target %.3f)"
          % (fdtd_s21, fdtd_s11, target))

    print("=== 2. independent analytic Picture B at er_star ===")
    ar, at = sp.analytic_sparams(er_star, cfg, geom)
    print("analytic |S21| %.5f |S11| %.5f" % (abs(at), abs(ar)))
    print("|S21| FDTD vs analytic discrepancy %.5f"
          % abs(fdtd_s21 - abs(at)))

    print("=== 3. analytic er root for |S21|=%.3f ===" % target)
    lo, hi = 1.0, 2.0
    # On the low-er branch |S21| decreases monotonically from 1.0 (er=1) as er
    # grows, so the root is bracketed in [1, 2].
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if analytic_abs_s21(mid, cfg, geom) > target:
            lo = mid  # |S21| still too large -> push er higher
        else:
            hi = mid
        if hi - lo < 1e-6:
            break
    er_analytic = 0.5 * (lo + hi)
    print("analytic er for target: %.4f" % er_analytic)
    print("FDTD er_star %.4f vs analytic er %.4f (difference %.4f)"
          % (er_star, er_analytic, abs(er_star - er_analytic)))

    out = dict(er_star=er_star, fdtd_abs_s21=float(fdtd_s21),
               fdtd_abs_s11=float(fdtd_s11), analytic_abs_s21=float(abs(at)),
               analytic_abs_s11=float(abs(ar)), er_analytic=er_analytic,
               target=target)
    with open(os.path.join(ROOT, "results", "optimization_validation.json"),
              "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("saved results/optimization_validation.json")


if __name__ == "__main__":
    main()
