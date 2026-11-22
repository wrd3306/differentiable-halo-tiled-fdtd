"""Aligned MEEP voxel benchmark (current sheet + self-managed modal DFT).

MEEP reads the canonical voxel index array through MaterialGrid
(do_averaging=False) and is driven by the same soft TE10 current sheet, the
same narrowband pulse and the same monitor planes as the self-engine. The
modal time series use the identical rectangular e^{+i omega t} DFT with an
empty-run subtraction. The discrete material provenance is therefore identical
across all three solvers; only each solver's internal field operator differs.
"""
import json
import os
import sys
import time

import meep as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VOX = os.path.join(HERE, "..", "..", "results", "external", "voxels")
OUT_DIR = os.path.join(HERE, "..", "..", "results", "external")
A, B, L = 24e-3, 12e-3, 100e-3
C = 3.0e8
F0 = 10.0e9
FMIN_HZ, FMAX_HZ, NFREQ = 7.0e9, 12.0e9, 26
PML_T = 4.0e-3
RUN_T = 3.0
COURANT = 0.4044
SRC_Z = -0.038
IN_Z = -0.025
OUT_Z = 0.020
P_T0, P_TW, P_FC = 1.0e-9 * C, 0.3e-9 * C, 10.0e9 / C


def load(tag):
    z = np.load(os.path.join(VOX, tag + ".npz"))
    return z["idx"], z["db_er"].astype(np.float64), float(z["grid_mm"])


def pulse_waveform(t):
    return float(np.exp(-((t - P_T0) / P_TW) ** 2)
                 * np.sin(2 * np.pi * P_FC * t))


def te10_profile(pt):
    return float(np.sin(np.pi * (pt.x + A / 2) / A))


def build_voxel_geometry(idx, db, grid):
    dx = grid * 1e-3
    blocks = []
    wall_t = 1.0e-3
    for mi in range(1, db.shape[0]):
        occ = (idx == mi)
        if not occ.any():
            continue
        ix, iy, iz = np.where(occ)
        imin, imax = ix.min(), ix.max() + 1
        jmin, jmax = iy.min(), iy.max() + 1
        kmin, kmax = iz.min(), iz.max() + 1
        x0, x1 = imin * dx - A / 2, imax * dx - A / 2
        y0, y1 = jmin * dx - B / 2, jmax * dx - B / 2
        z0, z1 = kmin * dx - L / 2, kmax * dx - L / 2
        sub = np.ascontiguousarray(
            occ[imin:imax, jmin:jmax, kmin:kmax], dtype=np.float64)
        mg = mp.MaterialGrid(
            mp.Vector3(imax - imin, jmax - jmin, kmax - kmin),
            mp.Medium(epsilon=1.0), mp.Medium(epsilon=float(db[mi])),
            weights=sub.flatten(), do_averaging=False)
        blocks.append(mp.Block(
            material=mg,
            center=mp.Vector3((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2),
            size=mp.Vector3(x1 - x0, y1 - y0, z1 - z0)))
    for xc in (-A / 2, A / 2):
        blocks.append(mp.Block(material=mp.perfect_electric_conductor,
                               center=mp.Vector3(xc, 0, 0),
                               size=mp.Vector3(wall_t, B + 2 * wall_t,
                                               L + 2 * wall_t)))
    for yc in (-B / 2, B / 2):
        blocks.append(mp.Block(material=mp.perfect_electric_conductor,
                               center=mp.Vector3(0, yc, 0),
                               size=mp.Vector3(A + 2 * wall_t, wall_t,
                                               L + 2 * wall_t)))
    return blocks


def modal_projection(dft_xy, nx):
    if dft_xy.shape[0] == nx:
        dft_xy = dft_xy[1:nx - 1, :]
    prof = dft_xy.mean(axis=1)
    x_int = np.arange(1, nx - 1)
    p_int = np.sin(np.pi * x_int * (A / nx) / A)
    return (2.0 / A) * np.sum(prof * p_int) * (A / nx)


def run_case(idx, db, grid, loaded):
    dx = grid * 1e-3
    nx, ny, nz = idx.shape
    geom = build_voxel_geometry(idx, db, grid) if loaded else build_walls()
    wave = mp.CustomSource(pulse_waveform)
    src = mp.Source(wave, mp.Ey, center=mp.Vector3(0, 0, SRC_Z),
                    size=mp.Vector3(A - dx, B - dx, 0), amp_func=te10_profile)
    sim = mp.Simulation(
        cell_size=mp.Vector3(nx * dx, ny * dx, nz * dx), geometry=geom,
        sources=[src], resolution=1.0 / dx, Courant=COURANT,
        eps_averaging=False, boundary_layers=[mp.PML(PML_T, direction=mp.Z)])
    series = {"in": [], "out": [], "t": []}
    reg_in = mp.Volume(center=mp.Vector3(0, 0, IN_Z),
                       size=mp.Vector3(A - dx, B - dx, 0))
    reg_out = mp.Volume(center=mp.Vector3(0, 0, OUT_Z),
                        size=mp.Vector3(A - dx, B - dx, 0))

    def sample(sim):
        series["in"].append(modal_projection(
            np.asarray(sim.get_array(mp.Ey, reg_in)), nx))
        series["out"].append(modal_projection(
            np.asarray(sim.get_array(mp.Ey, reg_out)), nx))
        series["t"].append(sim.meep_time())

    t0 = time.time()
    sim.run(sample, until=RUN_T)
    elapsed = time.time() - t0
    return sim, {k: np.array(v) for k, v in series.items()}, elapsed


def build_walls():
    return _plain_walls()


def _plain_walls():
    wall_t = 1.0e-3
    blocks = []
    for xc in (-A / 2, A / 2):
        blocks.append(mp.Block(material=mp.perfect_electric_conductor,
                               center=mp.Vector3(xc, 0, 0),
                               size=mp.Vector3(wall_t, B + 2 * wall_t,
                                               L + 2 * wall_t)))
    for yc in (-B / 2, B / 2):
        blocks.append(mp.Block(material=mp.perfect_electric_conductor,
                               center=mp.Vector3(0, yc, 0),
                               size=mp.Vector3(A + 2 * wall_t, wall_t,
                                               L + 2 * wall_t)))
    return blocks


def phasor(ts, t, f_hz):
    dt = t[1] - t[0]
    return np.sum(ts * np.exp(1j * 2 * np.pi * (f_hz / C) * t)) * dt


def run_one(device, grid):
    tag = "%s_%gmm" % (device, grid)
    idx, db, _ = load(tag)
    freqs = np.linspace(FMIN_HZ, FMAX_HZ, NFREQ)
    _, de, te = run_case(idx, db, grid, False)
    _, dl, tl = run_case(idx, db, grid, True)
    s11 = np.empty(NFREQ, dtype=complex)
    s21 = np.empty(NFREQ, dtype=complex)
    for fi, f in enumerate(freqs):
        s11[fi] = (phasor(dl["in"], dl["t"], f)
                   - phasor(de["in"], de["t"], f)) \
            / phasor(de["in"], de["t"], f)
        s21[fi] = phasor(dl["out"], dl["t"], f) \
            / phasor(de["out"], de["t"], f)
    i10 = int(np.argmin(abs(freqs - F0)))
    nx, ny, nz = idx.shape
    n_cells = nx * ny * nz
    return {
        "name": device, "solver": "MEEP-aligned", "grid_mm": grid,
        "n_cells": n_cells, "n_steps": len(dl["t"]),
        "dt_s": (dl["t"][1] - dl["t"][0]) / C,
        "elapsed_s": te + tl,
        "mcells_per_s": (n_cells * len(dl["t"]) / 1e6) / (te + tl),
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
    results = []
    for device in devices:
        print("=== MEEP-voxel", device, grid, "===", flush=True)
        r = run_one(device, grid)
        print("  |S11|=%.4f |S21|=%.4f E=%.4f %.2f MCells/s" % (
            r["at10"]["S11_abs"], r["at10"]["S21_abs"],
            r["at10"]["energy"], r["mcells_per_s"]), flush=True)
        results.append(r)
    out = os.path.join(OUT_DIR, "meep_voxel_%gmm.json" % grid)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
