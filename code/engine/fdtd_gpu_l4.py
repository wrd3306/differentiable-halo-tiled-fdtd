"""L4 mixed-precision engine: bf16 field storage, fp32 psi/coefficients."""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import fdtd_gpu as fg  # noqa: E402
from ext_build import load_ext  # noqa: E402

CU_SRC = os.path.join(ROOT, "code", "kernels", "fused_bf16.cu")
BUILD = os.path.join(ROOT, "code", "kernels", "_bf16_build")
FIELDS = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]


def get_module():
    return load_ext("fused_bf16", [CU_SRC], BUILD)


def build_state_l4(cfg, device, block=None):
    fields, const = fg.build_state(cfg, device, dtype=torch.float32, block=block)
    for k in FIELDS:
        fields[k] = fields[k].to(torch.bfloat16)
    return fields, const


def step(mod, f, c, n):
    Nx, Ny, Nz = f["Ex"].shape
    idx, idy, idz = 1.0 / c["dx"], 1.0 / c["dy"], 1.0 / c["dz"]
    mh = c["dt"] / fg.MU0
    mod.h_launch_bf16(
        f["Ex"], f["Ey"], f["Ez"], c["bz"], c["cz"],
        f["Hx"], f["Hy"], f["Hz"], f["pe"], f["px"],
        Nx, Ny, Nz, idx, idy, idz, mh)
    sv = np.sin(2 * np.pi * c["f0"] * n * c["dt"])
    mod.e_launch_bf16(
        f["Ex"], f["Ey"], f["Ez"], f["Hx"], f["Hy"], f["Hz"],
        c["Ce"], c["De"], c["bz"], c["cz"], f["phy"], f["phx"],
        c["jprof"], c["mx"], c["my"],
        Nx, Ny, Nz, idx, idy, idz, float(sv))


def run(cfg, device, n_steps, block=None, warmup=5, mod=None):
    if mod is None:
        mod = get_module()
    fields, const = build_state_l4(cfg, device, block=block)
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
