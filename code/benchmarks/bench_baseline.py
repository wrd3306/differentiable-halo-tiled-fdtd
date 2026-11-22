"""L0 naive GPU FDTD baseline: TCUPS, speedup, estimated bandwidth.

Sweeps cubic grids on GPU; also runs the exact production-size grid on CPU and
GPU for a realistic speedup. Persists results to results/baseline.json.
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "engine"))
import fdtd_gpu as fg  # noqa: E402

OUT = r"C:\Users\wrd\Doubao\chats\2026-09-22\new-chat-1\systolic_gpu_fdtd\results\baseline.json"

N_STATE = 10   # 6 fields + 4 CPML psi
N_TEMP = 12    # naive derivative tensors per step


def make_cube(n):
    dx = 1e-3
    cfg = dict(Nx=n, Ny=n, Nz=n, dx=dx, dy=dx, dz=dx,
               dt=fg.cfl_dt(dx, dx, dx), f0=9e9, npml=10, src_z=n // 2)
    return cfg


def measure(cfg, device, n_steps, warmup=5):
    elapsed, _ = fg.run(cfg, device, n_steps, warmup=warmup)
    return elapsed


def main():
    gpu = torch.device("cuda")
    sizes = [32, 48, 64, 96, 128, 160, 192, 224]
    steps = 60
    rows = []
    for n in sizes:
        cfg = make_cube(n)
        cells = n ** 3
        t_gpu = measure(cfg, gpu, steps)
        tcups = cells * steps / t_gpu / 1e12
        naive_bytes = (N_STATE + N_TEMP) * 2 * 4 * cells * steps
        bw_est = naive_bytes / t_gpu / 1e9
        state_bytes = N_STATE * 2 * 4 * cells * steps
        rows.append(dict(grid=n, cells=cells, steps=steps, time_s=round(t_gpu, 4),
                         tcups=round(tcups, 4),
                         naive_bw_est_gbs=round(bw_est, 1),
                         fused_floor_bw_gbs=round(state_bytes / t_gpu / 1e9, 1)))
        print(f"n={n:4d} cells={cells:9d} t={t_gpu:7.4f}s TCUPS={tcups:6.3f} "
              f"naiveBW~{bw_est:6.1f} GB/s")

    # Production-size grid: CPU vs GPU.
    dx = 22.86e-3 / 22
    dy = 10.16e-3 / 10
    dz = 1e-3
    pcfg = dict(Nx=23, Ny=11, Nz=150, dx=dx, dy=dy, dz=dz,
                dt=fg.cfl_dt(dx, dy, dz), f0=9e9, npml=24, src_z=28)
    psteps = 100
    pcells = 23 * 11 * 150
    t_gpu_p = measure(pcfg, gpu, psteps)
    t_cpu_p = measure(pcfg, torch.device("cpu"), psteps)
    prod = dict(cells=pcells, steps=psteps,
                gpu_time_s=round(t_gpu_p, 4), cpu_time_s=round(t_cpu_p, 4),
                speedup=round(t_cpu_p / t_gpu_p, 2),
                gpu_tcups=round(pcells * psteps / t_gpu_p / 1e12, 5),
                cpu_tcups=round(pcells * psteps / t_cpu_p / 1e12, 5))
    print("\nProduction grid 23x11x150:")
    print(json.dumps(prod, indent=2))

    out = dict(device=torch.cuda.get_device_name(0), gpu_rows=rows, production=prod)
    with open(OUT, "w") as fobj:
        json.dump(out, fobj, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
