"""Projected gradient optimization of the dielectric-window permittivity.

Drives the slab er so the gated-FDTD |S21| matches a target, using Adam with a
hard bound [er_min, er_max]. The gate window is fixed from a no-grad pre-run and
the empty-waveguide phasors are computed once and reused. Gated phasors are
accumulated online so memory stays bounded. The trajectory is saved to
results/device_optimization.json.
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


def main(scale=2, n_steps=3000, target=0.95, er_init=2.0,
         er_min=1.0, er_max=8.0, iters=40, lr=0.08, tol=1e-6):
    device = torch.device("cuda")
    cfg, geom = op.scaled_config(scale)
    print("fixing gate window from a no-grad pre-run ...")
    gates = op.build_gates(cfg, geom, n_steps, device)
    print("gate spans:", {z: (g[0], g[1]) for z, g in gates.items()})
    print("computing empty-waveguide phasors once ...")
    empty = op.empty_phasors(cfg, geom, n_steps, device, gates)

    er = torch.tensor(er_init, requires_grad=True, dtype=torch.float32)
    adam = torch.optim.Adam([er], lr=lr)
    traj = []
    best = dict(loss=float("inf"), er=er_init, abs_s21=0.0)
    for it in range(iters):
        adam.zero_grad()
        er_eval = float(er.detach())  # er at which this forward is evaluated
        s11, s21 = op.loaded_sparams(er, cfg, geom, n_steps, device, empty,
                                     gates)
        loss = (abs(s21) - target) ** 2
        loss.backward()
        adam.step()
        with torch.no_grad():
            er.clamp_(er_min, er_max)
        row = dict(iter=it, er_eval=er_eval, er_after=float(er.detach()),
                   abs_s21=float(abs(s21)), abs_s11=float(abs(s11)),
                   loss=float(loss.detach()))
        if row["loss"] < best["loss"]:
            best = dict(loss=row["loss"], er=er_eval, abs_s21=row["abs_s21"])
        traj.append(row)
        mem = torch.cuda.memory_allocated() / 1e9
        print("it %3d er %.4f |S21| %.5f |S11| %.5f loss %.3e mem %.2f GB"
              % (it, er_eval, row["abs_s21"], row["abs_s11"], row["loss"],
                 mem))
        if row["loss"] < tol:
            print("converged at iter %d" % it)
            break
        del s11, s21, loss
        torch.cuda.empty_cache()

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out = dict(scale=scale, n_steps=n_steps, target=target,
               er_min=er_min, er_max=er_max, er_star=best["er"],
               best_er=best["er"], best_abs_s21=best["abs_s21"],
               best_loss=best["loss"], final_abs_s21=best["abs_s21"],
               trajectory=traj)
    with open(os.path.join(ROOT, "results", "device_optimization.json"),
              "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("optimized er* = %.4f (best |S21| %.5f, loss %.2e)"
          % (best["er"], best["abs_s21"], best["loss"]))


if __name__ == "__main__":
    main()
