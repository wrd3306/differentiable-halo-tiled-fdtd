"""Self-engine voxel benchmark reading the canonical voxel index.

The self GPU engine reads the identical voxel npz as MEEP and openEMS, maps
the material index to the shared dielectric database, and runs the same TE10
current sheet with the same monitor planes and rectangular e^{+i omega t}
modal DFT. This makes the discrete material provenance identical across all
three solvers at the file level.
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "code", "engine"))
sys.path.insert(0, os.path.join(ROOT, "code", "inversion"))
import engine3d as e3  # noqa: E402
import fdtd_gpu as fg  # noqa: E402
import sparam as sp  # noqa: E402
import devices3d as d3  # noqa: E402

VOX = os.path.join(ROOT, "results", "external", "voxels")
OUT_DIR = os.path.join(ROOT, "results", "external")
FMIN_HZ, FMAX_HZ, NFREQ = 7.0e9, 12.0e9, 26
RECORD_S = 10.0e-9


def load(tag):
    z = np.load(os.path.join(VOX, tag + ".npz"))
    return z["idx"], z["db_er"].astype(np.float32), float(z["grid_mm"])


def run_case(cfg, er_map, n_steps, z_ref, z_tr, device):
    fields, const = e3.build_state_map(cfg, device, er_map)
    rec = {z_ref: [], z_tr: []}
    t0 = time.time()
    with torch.no_grad():
        for n in range(n_steps):
            fg.step(fields, const, n)
            rec[z_ref].append(sp.modal_amp(fields["Ey"], cfg, z_ref).item())
            rec[z_tr].append(sp.modal_amp(fields["Ey"], cfg, z_tr).item())
    elapsed = time.time() - t0
    return {z: np.array(v) for z, v in rec.items()}, elapsed


def phasor(ts, cfg, f):
    nn = np.arange(len(ts))
    return np.sum(ts * np.exp(1j * 2 * np.pi * f * nn * cfg["dt"])) * cfg["dt"]


def run_one(device, grid, device_t):
    scale = int(round(1.0 / grid))
    tag = "%s_%gmm" % (device, grid)
    idx, db, _ = load(tag)
    cfg = d3.config3d(scale)
    z_ref, z_tr = d3.planes(cfg)
    n_steps = int(np.ceil(RECORD_S / cfg["dt"]))

    empty_map = np.ones((cfg["Nx"], cfg["Ny"], cfg["Nz"]), dtype=np.float32)
    loaded_map = db[idx].astype(np.float32)

    re, te = run_case(cfg, empty_map, n_steps, z_ref, z_tr, device_t)
    rl, tl = run_case(cfg, loaded_map, n_steps, z_ref, z_tr, device_t)

    freqs = np.linspace(FMIN_HZ, FMAX_HZ, NFREQ)
    s11 = np.empty(NFREQ, dtype=complex)
    s21 = np.empty(NFREQ, dtype=complex)
    for fi, f in enumerate(freqs):
        s11[fi] = (phasor(rl[z_ref], cfg, f) - phasor(re[z_ref], cfg, f)) \
            / phasor(re[z_ref], cfg, f)
        s21[fi] = phasor(rl[z_tr], cfg, f) / phasor(re[z_tr], cfg, f)
    i10 = int(np.argmin(abs(freqs - cfg["f0"])))
    n_cells = cfg["Nx"] * cfg["Ny"] * cfg["Nz"]
    return {
        "name": device, "solver": "self-voxel", "grid_mm": grid,
        "n_cells": n_cells, "n_steps": n_steps, "dt_s": cfg["dt"],
        "elapsed_s": te + tl,
        "mcells_per_s": (n_cells * n_steps * 2 / 1e6) / (te + tl),
        "frequencies_hz": freqs.tolist(),
        "S11": [[float(z.real), float(z.imag)] for z in s11],
        "S21": [[float(z.real), float(z.imag)] for z in s21],
        "at10": {
            "S11_abs": float(abs(s11[i10])), "S21_abs": float(abs(s21[i10])),
            "energy": float(abs(s11[i10]) ** 2 + abs(s21[i10]) ** 2)},
    }


def main():
    grid = float(sys.argv[1])
    devices = sys.argv[2:] or sorted({
        f.split("_%gmm" % grid)[0]
        for f in os.listdir(VOX) if f.endswith("_%gmm.npz" % grid)})
    device_t = torch.device("cuda")
    results = []
    for device in devices:
        print("=== self-voxel", device, grid, "===", flush=True)
        r = run_one(device, grid, device_t)
        print("  |S11|=%.4f |S21|=%.4f E=%.4f %.2f MCells/s" % (
            r["at10"]["S11_abs"], r["at10"]["S21_abs"],
            r["at10"]["energy"], r["mcells_per_s"]), flush=True)
        results.append(r)
    out = os.path.join(OUT_DIR, "self_voxel_%gmm.json" % grid)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
