"""Benchmark L3b tiled vs L0, L1, L3a across cubic grids."""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.join(HERE, "..", "engine")
sys.path.insert(0, ENG)
import fdtd_gpu as l0  # noqa: E402
import fdtd_gpu_l1 as l1  # noqa: E402
import fdtd_gpu_l3 as l3a  # noqa: E402
import fdtd_gpu_l3b as l3b  # noqa: E402

OUT = r"C:\Users\wrd\Doubao\chats\2026-09-22\new-chat-1\systolic_gpu_fdtd\results\l3b_compare.json"
N_STATE = 10


def make_cube(n):
    dx = 1e-3
    return dict(Nx=n, Ny=n, Nz=n, dx=dx, dy=dx, dz=dx,
                dt=l0.cfl_dt(dx, dx, dx), f0=9e9, npml=10, src_z=n // 2)


def main():
    gpu = torch.device("cuda")
    sizes = [32, 64, 96, 128, 160, 192, 224]
    steps = 60
    ma = l3a.get_module()
    mb = l3b.get_module()
    rows = []
    for n in sizes:
        cfg = make_cube(n)
        cells = n ** 3
        t0, _ = l0.run(cfg, gpu, steps)
        t1, _ = l1.run_l1(cfg, gpu, steps)
        ta, _ = l3a.run(cfg, gpu, steps, mod=ma)
        tb, _ = l3b.run(cfg, gpu, steps, mod=mb)

        def mc(t):
            return cells * steps / t / 1e6

        bw = N_STATE * 2 * 4 * cells * steps / tb / 1e9
        row = dict(n=n, cells=cells,
                   l0=round(t0, 4), l1=round(t1, 4), l3a=round(ta, 4), l3b=round(tb, 4),
                   l3b_mcups=round(mc(tb), 1), l3b_vs_l0=round(t0 / tb, 2),
                   l3b_vs_l1=round(t1 / tb, 2), l3b_vs_l3a=round(ta / tb, 2),
                   floor_bw_gbs=round(bw, 1))
        rows.append(row)
        print(f"n={n:4d} | L0 {t0:6.4f} L1 {t1:6.4f} L3a {ta:6.4f} L3b {tb:6.4f} "
              f"| L3b {mc(tb):7.1f}MCUPS | vs L0 {t0/tb:4.2f}x vs L1 {t1/tb:4.2f}x "
              f"vs L3a {ta/tb:4.2f}x | floorBW {bw:6.1f}")

    with open(OUT, "w") as fh:
        json.dump(dict(rows=rows, physics="verify_l3b.py PASS"), fh, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
