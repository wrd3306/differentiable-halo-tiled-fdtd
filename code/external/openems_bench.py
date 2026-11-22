"""openEMS external benchmark for the 3-D waveguide devices.

Native Windows build (openEMS 0.37, MSVC). It replicates the exact geometry and
the uniform 0.5 mm grid of the self-developed engine. PEC walls are applied as
boundary conditions and PML absorbs the two z ends. Two rectangular waveguide
ports with analytic TE10 mode profiles provide the two-port S-parameters using
openEMS's own incident/reflected decomposition, an independent forward model
with no shared code (no inverse crime).
"""
import json
import os
import sys
import time

import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0

HERE = os.path.dirname(os.path.abspath(__file__))
GEOM = os.path.join(HERE, "devices_geom.json")
OUT_DIR = os.path.join(HERE, "..", "..", "results", "external")
os.makedirs(OUT_DIR, exist_ok=True)

with open(GEOM) as fh:
    spec = json.load(fh)

UNIT = 1e-3                       # drawing unit: millimetres
A = spec["waveguide_mm"]["a"]     # mm
B = spec["waveguide_mm"]["b"]
L = spec["waveguide_mm"]["L"]
# Grid may be overridden for the convergence study (env OPENEMS_GRID_MM).
GRID = float(os.environ.get("OPENEMS_GRID_MM", spec["grid_mm"]))  # mm
F0 = spec["f0_hz"]
FMIN, FMAX, NFREQ = 7.0e9, 12.0e9, 26
P1_Z, P2_Z = 25.0, 70.0           # reference planes in mm
PORT_LEN = 5.0
# Fixed physical record (10 ns) matching the self-developed engine. SetMaxTime
# takes seconds; this caps the otherwise open-ended -60 dB auto-stop that rings
# on resonant posts.
RECORD_S = 10.0e-9
T_MAX = RECORD_S


def run_one(device_name, sim_path):
    fdtd = openEMS()
    fdtd.SetGaussExcite(0.5 * (FMIN + FMAX), 0.5 * (FMAX - FMIN))
    # PEC walls on x and y, PML on the two z ends.
    fdtd.SetBoundaryCond(["PEC", "PEC", "PEC", "PEC", "PML_8", "PML_8"])

    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(UNIT)

    # Uniform 0.5 mm mesh on every axis.
    mesh.AddLine("x", np.arange(0, A + GRID / 2, GRID).tolist())
    mesh.AddLine("y", np.arange(0, B + GRID / 2, GRID).tolist())
    mesh.AddLine("z", np.arange(0, L + GRID / 2, GRID).tolist())

    # Dielectric materials (skip vacuum boxes and the PEC-wall region).
    for box in spec["devices"][device_name]:
        x0, x1, y0, y1, z0, z1, er = box
        if abs(er - 1.0) < 1e-9:
            continue
        mat = csx.AddMaterial("er_%g" % er, epsilon=er)
        mat.AddBox([x0, y0, z0], [x1, y1, z1])

    # Port 1 (active, +z); reference plane at start z = P1_Z.
    start1 = [0, 0, P1_Z]
    stop1 = [A, B, P1_Z + PORT_LEN]
    port1 = fdtd.AddRectWaveGuidePort(
        0, start1, stop1, "z", A * UNIT, B * UNIT, "TE10", 1)
    # Port 2 (passive, reversed so its direction is -z); plane at P2_Z.
    start2 = [0, 0, P2_Z]
    stop2 = [A, B, P2_Z - PORT_LEN]
    port2 = fdtd.AddRectWaveGuidePort(
        1, start2, stop2, "z", A * UNIT, B * UNIT, "TE10", 0)

    fdtd.SetMaxTime(T_MAX)
    # Optional explicit thread count (env OPENEMS_THREADS); 0/absent lets the
    # built-in auto-tuner choose. Used for a controlled single-core comparison.
    run_kw = {}
    _threads = os.environ.get("OPENEMS_THREADS")
    if _threads is not None:
        run_kw["numThreads"] = int(_threads)
    t0 = time.time()
    fdtd.Run(sim_path, cleanup=False, verbose=0, **run_kw)
    elapsed = time.time() - t0

    freq = np.linspace(FMIN, FMAX, NFREQ)
    port1.CalcPort(sim_path, freq)
    port2.CalcPort(sim_path, freq)
    s11 = port1.uf_ref / port1.uf_inc
    s21 = port2.uf_ref / port1.uf_inc

    i10 = int(np.argmin(abs(freq - F0)))
    n_cells = int(round(A / GRID)) * int(round(B / GRID)) * int(round(L / GRID))
    # The voltage probe decimates (writes every 10th step); recover the total
    # FDTD step count from the last recorded time and the CFL time step.
    dt_phys = UNIT * GRID / (C0 * np.sqrt(3.0))
    n_steps = int(round(port1.u_time[-1] / dt_phys))
    mcells_s = (n_cells * n_steps / 1e6) / elapsed if n_steps else 0.0
    return {
        "name": device_name,
        "solver": "openEMS",
        "openems_version": "0.37.0",
        "grid_mm": GRID,
        "n_cells": n_cells,
        "n_steps": n_steps,
        "dt_s": dt_phys,
        "elapsed_s": elapsed,
        "mcells_per_s": mcells_s,
        "frequencies_hz": freq.tolist(),
        "S11": [[float(z.real), float(z.imag)] for z in s11],
        "S21": [[float(z.real), float(z.imag)] for z in s21],
        "at10": {
            "S11_abs": float(abs(s11[i10])), "S21_abs": float(abs(s21[i10])),
            "S11_phase": float(np.angle(s11[i10])),
            "S21_phase": float(np.angle(s21[i10])),
            "energy": float(abs(s11[i10]) ** 2 + abs(s21[i10]) ** 2),
        },
    }


def main():
    names = sys.argv[1:] or [n for n in spec["devices"].keys() if n != "ar_match"]
    results = []
    for name in names:
        print("=== openEMS running", name, "===", flush=True)
        sim_path = os.path.join(OUT_DIR, "openems_sim_" + name)
        res = run_one(name, sim_path)
        print("  |S11|=%.4f |S21|=%.4f energy=%.4f elapsed=%.2fs nsteps=%d" % (
            res["at10"]["S11_abs"], res["at10"]["S21_abs"],
            res["at10"]["energy"], res["elapsed_s"], res["n_steps"]), flush=True)
        results.append(res)
    out_path = os.path.join(OUT_DIR, "openems_results.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=1)
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
