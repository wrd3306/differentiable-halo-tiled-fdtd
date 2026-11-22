"""Throughput benchmark for the tiled differentiable engine on the design grid."""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "code", "inversion"))

import fdtd_diff_tiled as fdt  # noqa: E402
import pixel_design as px  # noqa: E402

FREQS_HZ = [f * 1e9 for f in px.FREQS_GHZ]


def main(n_steps=1200):
    device = torch.device("cuda")
    cfg, _ = px.pixel_config(record_ns=6.0)
    base, cell_masks, _ = px.build_geometry(cfg, device)
    # Baseline: obstacle only, all design densities zero.
    theta = torch.zeros(cell_masks.shape[0], device=device)
    er_map, _ = px.er_map_from(theta, base, cell_masks, beta=2.0)
    res = fdt.benchmark(cfg, device, er_map, n_steps, FREQS_HZ, chunk=20)
    print(json.dumps(res, indent=2))
    out = os.path.join(ROOT, "results", "tiled_throughput.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
