"""L3 fused-CUDA FDTD engine.

Drives the hand-written fused kernels in kernels/fused_fdtd.cu. The state and
coefficients are identical to the L0 reference (fdtd_gpu.build_state); only the
per-step update changes, so the two engines can be compared field-by-field.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "code", "engine"))

import fdtd_gpu as fg  # noqa: E402
from ext_build import load_ext  # noqa: E402

CU_SRC = os.path.join(ROOT, "code", "kernels", "fused_fdtd.cu")
BUILD = os.path.join(ROOT, "code", "kernels", "_fused_build")


def get_module():
    return load_ext("fused_fdtd", [CU_SRC], BUILD)


def step(mod, f, c, n):
    Nx, Ny, Nz = f["Ex"].shape
    idx, idy, idz = 1.0 / c["dx"], 1.0 / c["dy"], 1.0 / c["dz"]
    mh = c["dt"] / fg.MU0
    mod.h_launch(
        f["Ex"], f["Ey"], f["Ez"], c["bz"], c["cz"],
        f["Hx"], f["Hy"], f["Hz"], f["pe"], f["px"],
        Nx, Ny, Nz, idx, idy, idz, mh)
    sv = np.sin(2 * np.pi * c["f0"] * n * c["dt"])
    mod.e_launch(
        f["Ex"], f["Ey"], f["Ez"], f["Hx"], f["Hy"], f["Hz"],
        c["Ce"], c["De"], c["bz"], c["cz"], f["phy"], f["phx"],
        c["jprof"], c["mx"], c["my"],
        Nx, Ny, Nz, idx, idy, idz, float(sv))


def run(cfg, device, n_steps, dtype=torch.float32, block=None, warmup=5, mod=None):
    if mod is None:
        mod = get_module()
    fields, const = fg.build_state(cfg, device, dtype=dtype, block=block)
    with torch.no_grad():
        for n in range(warmup):
            step(mod, fields, const, n)
        torch.cuda.synchronize()
        import time
        t0 = time.perf_counter()
        for n in range(n_steps):
            step(mod, fields, const, n + warmup)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return elapsed, fields
