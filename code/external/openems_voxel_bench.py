"""Aligned openEMS voxel benchmark (current sheet + self-managed modal DFT).

openEMS reads the canonical DiscMaterial voxel file and is driven by the same
soft TE10 current sheet and the same narrowband pulse as the self-engine. Field
dumps at the identical monitor planes are projected with the exact modal
functional and the S-parameters are recovered with an empty-run subtraction
and the same rectangular e^{+i omega t} DFT. This isolates the field solver from
the port decomposition: physics setup is identical across all three solvers.
"""
import json
import os
import sys
import time

import h5py
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0

HERE = os.path.dirname(os.path.abspath(__file__))
VOX = os.path.join(HERE, "..", "..", "results", "external", "voxels")
OUT_DIR = os.path.join(HERE, "..", "..", "results", "external")
A, B, L = 24.0, 12.0, 100.0
SRC_Z = 12.0
IN_Z, OUT_Z = 25.0, 70.0
FMIN, FMAX, NFREQ = 7.0e9, 12.0e9, 26
F0 = 10.0e9
RECORD_S = 10.0e-9
PULSE = "exp(-((t-1e-9)/3e-10)^2)*sin(2*pi*10e9*t)"


def modal_series(h5_path, nx, dx_mm):
    """Read a per-step Ey dump and apply the exact TE10 modal functional."""
    with h5py.File(h5_path, "r") as fh:
        frames = sorted(fh["FieldData/TD"].keys())
        ey = np.stack([fh["FieldData/TD"][k][1, ..., 0] for k in frames])
        times = np.array([int(k) for k in frames], dtype=np.float64)
    # interior x nodes: self projects indices 1..Nx-2 (Nx-2 values).
    x_int = ey[:, 1:nx - 1, :].mean(axis=2)
    a_m = A * 1e-3
    dx_m = dx_mm * 1e-3
    xg = np.arange(1, nx - 1)
    p_int = np.sin(np.pi * xg * dx_m / a_m)
    modal = (2.0 / a_m) * np.sum(x_int * p_int[None, :], axis=1) * dx_m
    return modal, times


def setup(grid, loaded, tag):
    fdtd = openEMS(CellConstantMaterial=True)
    fdtd.SetBoundaryCond(["PEC", "PEC", "PEC", "PEC", "PML_8", "PML_8"])
    fdtd.SetCustomExcite(PULSE, F0, FMAX)
    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(1e-3)
    mesh.AddLine("x", np.arange(0, A + grid / 2, grid).tolist())
    mesh.AddLine("y", np.arange(0, B + grid / 2, grid).tolist())
    mesh.AddLine("z", np.arange(0, L + grid / 2, grid).tolist())
    if loaded:
        disc = csx.AddDiscMaterial(
            "voxels", filename=os.path.join(VOX, tag + ".h5"),
            filetype=0, scale=1.0 / 1e-3)
        disc.AddBox([0, 0, 0], [A, B, L], priority=0)
    exc = csx.AddExcitation("te10", 0, [0, 1, 0])
    exc.AddBox([0, 0, SRC_Z], [A, B, SRC_Z])
    exc.SetWeightFunction(["0", "sin(pi*x/24)", "0"])
    for name, zc in (("in", IN_Z), ("out", OUT_Z)):
        d = csx.AddDump("dump_" + name, dump_type=0, dump_mode=0,
                        file_type=1, over_sampling=60)
        d.AddBox([0, 0, zc], [A, B, zc])
    fdtd.SetMaxTime(RECORD_S)
    return fdtd


def run_case(grid, loaded, tag, sim_path):
    fdtd = setup(grid, loaded, tag)
    t0 = time.time()
    fdtd.Run(sim_path, cleanup=False, verbose=0)
    elapsed = time.time() - t0
    nx = int(round(A / grid))
    mi, steps = modal_series(os.path.join(sim_path, "dump_in.h5"), nx, grid)
    mo, _ = modal_series(os.path.join(sim_path, "dump_out.h5"), nx, grid)
    return mi, mo, steps, elapsed


def phasor(ts, steps, dt, f_hz):
    t = steps * dt
    return np.sum(ts * np.exp(1j * 2 * np.pi * f_hz * t)) * dt


def run_one(device, grid):
    scale = int(round(1.0 / grid))
    tag = "%s_%gmm" % (device, grid)
    base = os.path.join(OUT_DIR, "openems_vx_" + tag)
    dt = UNIT = grid * 1e-3 / (C0 * np.sqrt(3.0))
    ei, eo, st, te = run_case(grid, False, tag, base + "_empty")
    li, lo, sl, tl = run_case(grid, True, tag, base + "_load")
    freqs = np.linspace(FMIN, FMAX, NFREQ)
    s11 = np.empty(NFREQ, dtype=complex)
    s21 = np.empty(NFREQ, dtype=complex)
    for fi, f in enumerate(freqs):
        pe = phasor(ei, st, dt, f)
        pt = phasor(eo, st, dt, f)
        lr = phasor(li, sl, dt, f)
        lt = phasor(lo, sl, dt, f)
        s11[fi] = (lr - pe) / pe
        s21[fi] = lt / pt
    i10 = int(np.argmin(abs(freqs - F0)))
    n_cells = int(round(A / grid)) * int(round(B / grid)) * int(round(L / grid))
    n_steps = int(round(st[-1]))
    return {
        "name": device, "solver": "openEMS-aligned", "grid_mm": grid,
        "n_cells": n_cells, "n_steps": n_steps, "dt_s": dt,
        "elapsed_s": te + tl,
        "mcells_per_s": (n_cells * n_steps / 1e6) / (te + tl),
        "frequencies_hz": freqs.tolist(),
        "S11": [[float(z.real), float(z.imag)] for z in s11],
        "S21": [[float(z.real), float(z.imag)] for z in s21],
        "at10": {
            "S11_abs": float(abs(s11[i10])), "S21_abs": float(abs(s21[i10])),
            "energy": float(abs(s11[i10]) ** 2 + abs(s21[i10]) ** 2)},
    }


def main():
    grid = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    names = sys.argv[2:] or [n for n in os.listdir(VOX)
                             if n.endswith("_%gmm.npz" % grid)]
    devices = sorted({n.split("_%gmm" % grid)[0] for n in names})
    results = []
    for device in devices:
        print("=== openEMS-voxel", device, grid, "===", flush=True)
        r = run_one(device, grid)
        print("  |S11|=%.4f |S21|=%.4f E=%.4f %.2f MCells/s" % (
            r["at10"]["S11_abs"], r["at10"]["S21_abs"],
            r["at10"]["energy"], r["mcells_per_s"]), flush=True)
        results.append(r)
    out = os.path.join(OUT_DIR, "openems_voxel_%gmm.json" % grid)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
