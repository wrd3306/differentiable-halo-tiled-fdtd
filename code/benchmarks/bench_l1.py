"""Compare L0 (naive) vs L1 (preallocated/in-place) and verify physics match."""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.join(HERE, "..", "engine")
sys.path.insert(0, ENG)
import fdtd_gpu as l0  # noqa: E402
import fdtd_gpu_l1 as l1  # noqa: E402

OUT = r"C:\Users\wrd\Doubao\chats\2026-09-22\new-chat-1\systolic_gpu_fdtd\results\l1_compare.json"


def make_cube(n):
    dx = 1e-3
    return dict(Nx=n, Ny=n, Nz=n, dx=dx, dy=dx, dz=dx,
                dt=l0.cfl_dt(dx, dx, dx), f0=9e9, npml=10, src_z=n // 2)


def main():
    gpu = torch.device("cuda")
    sizes = [32, 64, 96, 128, 160, 192, 224]
    steps = 60
    rows = []
    for n in sizes:
        cfg = make_cube(n)
        cells = n ** 3
        t0s, _ = l0.run(cfg, gpu, steps)
        t1s, _ = l1.run_l1(cfg, gpu, steps)
        ncups0 = cells * steps / t0s / 1e6
        ncups1 = cells * steps / t1s / 1e6
        rows.append(dict(n=n, cells=cells, l0_time=round(t0s, 4), l1_time=round(t1s, 4),
                         l0_mcups=round(ncups0, 2), l1_mcups=round(ncups1, 2),
                         speedup=round(t0s / t1s, 3)))
        print(f"n={n:4d} L0 {t0s:6.4f}s/{ncups0:7.1f}MCUPS  L1 {t1s:6.4f}s/{ncups1:7.1f}MCUPS  gain {t0s/t1s:4.2f}x")

    # Physics equivalence: same config, same step count, compare fields.
    cfg = make_cube(96)
    f0state, c0 = l0.build_state(cfg, gpu)
    f1state, c1, bufs = l1.build_state_l1(cfg, gpu)
    with torch.no_grad():
        for k in range(150):
            l0.step(f0state, c0, k)
            l1.step_l1(f1state, c1, bufs, k)
    diffs = {}
    for key in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        a, b = f0state[key], f1state[key]
        diffs[key] = float((a - b).abs().max() / (a.abs().max() + 1e-12))
    print("\nrelative max field diff L1 vs L0:", json.dumps({k: round(v, 2e7 and 6) for k, v in diffs.items()}))
    match = all(v < 1e-4 for v in diffs.values())
    print("physics match:", match)

    out = dict(rows=rows, field_rel_diff=diffs, physics_match=match)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
