"""Batch full-wave run of the whole 3-D device library.

Runs every device at the chosen scale with the ungated-DFT extractor, stores
the two-port S-parameters over the single-mode band and the residual energy in
a compact JSON, and keeps the modal time series in a .npz for figures.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(HERE, "..", "engine"))
sys.path.insert(0, HERE)
import devices3d as d3  # noqa: E402
import run3d  # noqa: E402


def main(scale=2, record_ns=10.0):
    device = torch.device("cuda")
    frequencies = np.linspace(7e9, 12e9, 26)
    summary = []
    series_npz = {}
    for name in d3.DEVICES:
        r = run3d.run_device(name, scale=scale, device=device,
                             physical_record=record_ns * 1e-9,
                             frequencies=frequencies)
        s11, s21 = r["s11"], r["s21"]
        row = dict(name=name, scale=scale, n_steps=r["n_steps"],
                   S11_abs=abs(s11), S21_abs=abs(s21),
                   S11_phase=float(np.angle(s11)),
                   S21_phase=float(np.angle(s21)),
                   energy_balance=r["energy_balance"],
                   residual_energy=r["residual_energy"],
                   sweep=[dict(frequency=x["frequency"],
                               S11_abs=abs(x["s11"]),
                               S21_abs=abs(x["s21"]),
                               S11_phase=float(np.angle(x["s11"])),
                               S21_phase=float(np.angle(x["s21"])),
                               energy=x["energy"])
                          for x in r["sweep"]])
        summary.append(row)
        for k, v in r["series"].items():
            series_npz[f"{name}__{k}"] = v
        print(json.dumps({k: v for k, v in row.items() if k != "sweep"}))

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "devices3d.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    np.savez_compressed(os.path.join(ROOT, "results", "devices3d_series.npz"),
                        **series_npz)
    print("saved devices3d.json and devices3d_series.npz")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
