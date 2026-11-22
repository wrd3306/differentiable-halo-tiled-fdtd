"""Benchmark L4 bf16 vs L3b fp32 tiled across cubic grids."""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.join(HERE, "..", "engine")
sys.path.insert(0, ENG)
import fdtd_gpu as l0  # noqa: E402
import fdtd_gpu_l3b as l3b  # noqa: E402
import fdtd_gpu_l4 as l4  # noqa: E402

OUT = r"C:\Users\wrd\Doubao\chats\2026-09-22\new-chat-1\systolic_gpu_fdtd\results\l4_compare.json"


def make_cube(n):
    dx = 1e-3
    return dict(Nx=n, Ny=n, Nz=n, dx=dx, dy=dx, dz=dx,
                dt=l0.cfl_dt(dx, dx, dx), f0=9e9, npml=10, src_z=n // 2)


def main():
    gpu = torch.device("cuda")
    sizes = [32, 64, 96, 128, 160, 192, 224]
    steps = 60
    m3 = l3b.get_module()
    m4 = l4.get_module()
    rows = []
    for n in sizes:
        cfg = make_cube(n)
        cells = n ** 3
        t3, _ = l3b.run(cfg, gpu, steps, mod=m3)
        t4, _ = l4.run(cfg, gpu, steps, mod=m4)

        def mc(t):
            return cells * steps / t / 1e6

        row = dict(n=n, l3b_time=round(t3, 4), l4_time=round(t4, 4),
                   l3b_mcups=round(mc(t3), 1), l4_mcups=round(mc(t4), 1),
                   l4_vs_l3b=round(t3 / t4, 2))
        rows.append(row)
        print(f"n={n:4d} | L3b {t3:6.4f}s {mc(t3):7.1f} | L4 {t4:6.4f}s {mc(t4):7.1f} "
              f"MCUPS | bf16 gain {t3/t4:4.2f}x")

    with open(OUT, "w") as fh:
        json.dump(dict(rows=rows, physics="verify_l4.py measured"), fh, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
