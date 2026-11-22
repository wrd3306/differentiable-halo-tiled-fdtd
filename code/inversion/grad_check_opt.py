"""Finite-difference gradient check for the gated-DFT optimization objective."""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.join(HERE, "..", "engine")
sys.path.insert(0, ENG)
sys.path.insert(0, HERE)
import optimize as op  # noqa: E402


def main(scale=2, n_steps=3000, target=0.95):
    device = torch.device("cuda")
    cfg, geom = op.scaled_config(scale)
    er0 = 3.0
    print("building fixed gates ...")
    gates = op.build_gates(cfg, geom, n_steps, device)
    print("gate spans:", {z: (g[0], g[1]) for z, g in gates.items()})
    empty = op.empty_phasors(cfg, geom, n_steps, device, gates)

    def loss(er):
        return op.objective(torch.tensor(er, dtype=torch.float32), cfg, geom,
                            n_steps, device, target, empty, gates)

    er = torch.tensor(er0, requires_grad=True, dtype=torch.float32)
    l = op.objective(er, cfg, geom, n_steps, device, target, empty, gates)
    l.backward()
    analytic = float(er.grad)
    print("objective at er=%.2f: %.6e" % (er0, float(l.detach())))
    print("analytic grad %.6e" % analytic)
    for h in (0.02, 0.05, 0.1):
        fdg = (float(loss(er0 + h)) - float(loss(er0 - h))) / (2 * h)
        rel = abs(analytic - fdg) / max(abs(analytic), abs(fdg), 1e-12)
        print("h %.3f finite-diff %.6e rel-err %.3e" % (h, fdg, rel))


if __name__ == "__main__":
    main()
