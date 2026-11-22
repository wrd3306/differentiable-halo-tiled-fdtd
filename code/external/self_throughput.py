"""Throughput of the self-developed GPU engine on the exact 3-D device grid.

Uses the same engine3d map engine and the 48 x 24 x 200 (0.5 mm) device grid,
timed over a fixed number of steps after warmup, so it is directly comparable to
the MEEP and openEMS MCells/s figures measured on the same mesh.
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(HERE, "..", "engine"))
sys.path.insert(0, os.path.join(HERE, "..", "inversion"))
import engine3d as e3  # noqa: E402
import fdtd_gpu as fg  # noqa: E402
import devices3d as d3  # noqa: E402


def main():
    device = torch.device("cuda")
    cfg = d3.config3d(2)
    er_map = np.ones((cfg["Nx"], cfg["Ny"], cfg["Nz"]), dtype=np.float32)
    fields, const = e3.build_state_map(cfg, device, er_map)
    n_cells = cfg["Nx"] * cfg["Ny"] * cfg["Nz"]
    n_steps = 14836

    with torch.no_grad():
        for n in range(200):
            fg.step(fields, const, n)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for n in range(n_steps):
            fg.step(fields, const, n + 200)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    mcells = n_cells * n_steps / 1e6 / elapsed
    print("device:", torch.cuda.get_device_name(0))
    print("cells:", n_cells, "steps:", n_steps, "elapsed: %.4fs" % elapsed)
    print("self GPU MCells/s: %.2f" % mcells)


if __name__ == "__main__":
    main()
