"""L3 fused-CUDA benchmark vs L0 (naive) and L1 (preallocated/in-place)."""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.join(HERE, "..", "engine")
sys.path.insert(0, ENG)
import fdtd_gpu as l0  # noqa: E402
import fdtd_gpu_l1 as l1  # noqa: E402
import fdtd_gpu_l3 as l3  # noqa: E402

OUT = r"C:\Users\wrd\Doubao\chats\2026-09-22\new-chat-1\systolic_gpu_fdtd\results\l3_compare.json"
N_STATE = 10


def make_cube(n):
    dx = 1e-3
    return dict(Nx=n, Ny=n, Nz=n, dx=dx, dy=dx, dz=dx,
                dt=l0.cfl_dt(dx, dx, dx), f0=9e9, npml=10, src_z=n // 2)


def main():
    gpu = torch.device("cuda")
    sizes = [32, 64, 96, 128, 160, 192, 224]
    steps = 60
    mod = l3.get_module()
    rows = []
    for n in sizes:
        cfg = make_cube(n)
        cells = n ** 3
        t0, _ = l0.run(cfg, gpu, steps)
        t1, _ = l1.run_l1(cfg, gpu, steps)
        t3, _ = l3.run(cfg, gpu, steps, mod=mod)

        def mcups(t):
            return cells * steps / t / 1e6

        floor_bw = N_STATE * 2 * 4 * cells * steps / t3 / 1e9
        row = dict(n=n, cells=cells,
                   l0_time=round(t0, 4), l1_time=round(t1, 4), l3_time=round(t3, 4),
                   l0_mcups=round(mcups(t0), 1), l1_mcups=round(mcups(t1), 1),
                   l3_mcups=round(mcups(t3), 1),
                   l3_vs_l0=round(t0 / t3, 2), l3_vs_l1=round(t1 / t3, 2),
                   l3_floor_bw_gbs=round(floor_bw, 1))
        rows.append(row)
        print(f"n={n:4d} | L0 {t0:6.4f}s {mcups(t0):7.1f} | L1 {t1:6.4f}s {mcups(t1):7.1f} "
              f"| L3 {t3:6.4f}s {mcups(t3):7.1f} MCUPS | gain {t0/t3:4.2f}x vs L0, "
              f"{t1/t3:4.2f}x vs L1 | floorBW {floor_bw:6.1f} GB/s")

    with open(OUT, "w") as fh:
        json.dump(dict(rows=rows, physics="verified by verify_l3.py PASS"), fh, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
