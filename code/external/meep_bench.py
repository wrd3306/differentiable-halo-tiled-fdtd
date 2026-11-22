"""MEEP external benchmark for the 3-D waveguide devices.

Runs inside WSL (conda env pmp). It replicates the exact geometry and the 0.5 mm
grid of the self-developed engine, including the same CFL time step (0.7 of the
3-D Courant limit). A PEC-walled guide is driven by a TE10 current sheet (no MPB
eigenmode source). The modal time series are recorded at both monitor planes and
the two-port S-parameters are recovered with an empty-run reference and an
analytic TE10 modal projection, using the same rectangular, ungated DFT the
self-engine uses. The independent implementation shares no forward-model code,
so there is no inverse crime.
"""
import json
import os
import sys
import time

import meep as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GEOM = os.path.join(HERE, "devices_geom.json")
OUT_DIR = os.path.join(HERE, "..", "..", "results", "external")
os.makedirs(OUT_DIR, exist_ok=True)

with open(GEOM) as fh:
    spec = json.load(fh)

A = spec["waveguide_mm"]["a"] * 1e-3
B = spec["waveguide_mm"]["b"] * 1e-3
L = spec["waveguide_mm"]["L"] * 1e-3
# Grid may be overridden for the convergence study (env MEEP_GRID_MM).
_grid_mm = float(os.environ.get("MEEP_GRID_MM", spec["grid_mm"]))
DX = _grid_mm * 1e-3
RES = 1.0 / DX
F0 = spec["f0_hz"]
C = 3.0e8
F0_C = F0 / C
FMIN_HZ, FMAX_HZ, NFREQ = 7.0e9, 12.0e9, 26
FW_C = (3.0e9) / C
PML_T = 4.0e-3
RUN_T = 3.0          # code time units = 10 ns physical record
COURANT = 0.4044     # matches the self-engine dt = 0.7 dx/(c sqrt(3))

SRC_Z = -0.038       # global z = 12 mm
IN_Z = -0.025        # global z = 25 mm
OUT_Z = 0.020        # global z = 70 mm

Nx = int(round(A / DX))
Ny = int(round(B / DX))
Nz = int(round(L / DX))
X_FULL = -A / 2 + DX / 2 + DX * np.arange(Nx)
X_INT = X_FULL[1:Nx - 1]
P_INT = np.sin(np.pi * (X_INT + A / 2) / A)

# Exact self-engine narrowband pulse exp(-((t-t0)/tw)^2) sin(2 pi fc t),
# converted from SI seconds to MEEP code time (multiply by c).
P_T0 = 1.0e-9 * C          # 1 ns envelope centre
P_TW = 0.3e-9 * C          # 0.3 ns Gaussian width
P_FC = 10.0e9 / C          # 10 GHz carrier
SRC_KIND = os.environ.get("MEEP_SRC", "pulse")


def pulse_waveform(t):
    env = np.exp(-((t - P_T0) / P_TW) ** 2)
    return float(env * np.sin(2 * np.pi * P_FC * t))


def te10_profile(pt):
    gx = pt.x + A / 2
    return float(np.sin(np.pi * gx / A))


def global_to_meep(box):
    x0, x1, y0, y1, z0, z1, er = box
    return (x0 * 1e-3 - A / 2, x1 * 1e-3 - A / 2,
            y0 * 1e-3 - B / 2, y1 * 1e-3 - B / 2,
            z0 * 1e-3 - L / 2, z1 * 1e-3 - L / 2, er)


def build_walls():
    blocks = []
    wall_t = 1.0e-3
    for xc in (-A / 2, A / 2):
        blocks.append(mp.Block(
            material=mp.perfect_electric_conductor,
            center=mp.Vector3(xc, 0, 0),
            size=mp.Vector3(wall_t, B + 2 * wall_t, L + 2 * wall_t)))
    for yc in (-B / 2, B / 2):
        blocks.append(mp.Block(
            material=mp.perfect_electric_conductor,
            center=mp.Vector3(0, yc, 0),
            size=mp.Vector3(A + 2 * wall_t, wall_t, L + 2 * wall_t)))
    return blocks


def build_geometry(device_name):
    # Dielectric blocks first; walls appended last so they override any
    # dielectric overlap at the wall cells, matching the self-engine clamping.
    blocks = []
    for box in spec["devices"][device_name]:
        x0, x1, y0, y1, z0, z1, er = global_to_meep(box)
        if abs(er - 1.0) < 1e-9:
            continue
        blocks.append(mp.Block(
            material=mp.Medium(epsilon=er),
            center=mp.Vector3((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2),
            size=mp.Vector3(x1 - x0, y1 - y0, z1 - z0)))
    blocks.extend(build_walls())
    return blocks


def modal_projection(dft_xy):
    """TE10 modal scalar from an Ey field (mean over uniform y)."""
    if dft_xy.shape[0] == Nx:
        dft_xy = dft_xy[1:Nx - 1, :]
    prof = dft_xy.mean(axis=1)
    return (2.0 / A) * np.sum(prof * P_INT) * DX


def run_case(device_name, loaded):
    geom = build_geometry(device_name) if loaded else build_walls()
    if SRC_KIND == "gauss":
        wave = mp.GaussianSource(F0_C, fwidth=FW_C)
    else:
        wave = mp.CustomSource(pulse_waveform)
    src = mp.Source(
        wave, component=mp.Ey,
        center=mp.Vector3(0, 0, SRC_Z),
        size=mp.Vector3(A - DX, B - DX, 0), amp_func=te10_profile)
    sim = mp.Simulation(
        cell_size=mp.Vector3(A, B, L), geometry=geom, sources=[src],
        resolution=RES, Courant=COURANT, eps_averaging=False,
        boundary_layers=[mp.PML(PML_T, direction=mp.Z)])

    series = {"in": [], "out": [], "t": []}
    reg_in = mp.Volume(center=mp.Vector3(0, 0, IN_Z),
                       size=mp.Vector3(A - DX, B - DX, 0))
    reg_out = mp.Volume(center=mp.Vector3(0, 0, OUT_Z),
                        size=mp.Vector3(A - DX, B - DX, 0))

    def sample(sim):
        ey_in = np.asarray(sim.get_array(mp.Ey, reg_in))
        ey_out = np.asarray(sim.get_array(mp.Ey, reg_out))
        series["in"].append(modal_projection(ey_in))
        series["out"].append(modal_projection(ey_out))
        series["t"].append(sim.meep_time())

    t0 = time.time()
    sim.run(sample, until=RUN_T)
    elapsed = time.time() - t0
    return sim, {k: np.array(v) for k, v in series.items()}, elapsed


def ungated_phasor(ts, t, f_hz):
    """Rectangular DFT in the e^{+i omega t} convention (matches self-engine)."""
    dt = t[1] - t[0]
    return np.sum(ts * np.exp(1j * 2 * np.pi * (f_hz / C) * t)) * dt


def run_one(device_name, frequencies=None):
    frequencies = np.linspace(FMIN_HZ, FMAX_HZ, NFREQ) if frequencies is None \
        else np.asarray(frequencies)
    sim_e, de, te = run_case(device_name, False)
    sim_l, dl, tl = run_case(device_name, True)

    s11 = np.empty(len(frequencies), dtype=complex)
    s21 = np.empty(len(frequencies), dtype=complex)
    for fi, f in enumerate(frequencies):
        pe = ungated_phasor(de["in"], de["t"], f)
        pt = ungated_phasor(de["out"], de["t"], f)
        lr = ungated_phasor(dl["in"], dl["t"], f)
        lt = ungated_phasor(dl["out"], dl["t"], f)
        s11[fi] = (lr - pe) / pe
        s21[fi] = lt / pt

    i10 = int(np.argmin(abs(frequencies - F0)))
    n_steps = len(dl["t"])
    n_cells = Nx * Ny * Nz
    dt_phys = (dl["t"][1] - dl["t"][0]) / C
    mcells_s = (n_cells * n_steps / 1e6) / tl
    return {
        "name": device_name,
        "solver": "MEEP",
        "meep_version": mp.__version__,
        "grid_mm": _grid_mm,
        "n_cells": n_cells,
        "n_steps": n_steps,
        "dt_s": dt_phys,
        "elapsed_s": te + tl,
        "mcells_per_s": mcells_s,
        "frequencies_hz": frequencies.tolist(),
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
        print("=== MEEP running", name, "===", flush=True)
        res = run_one(name)
        print("  |S11|=%.4f |S21|=%.4f energy=%.4f elapsed=%.2fs %.2f MCells/s" % (
            res["at10"]["S11_abs"], res["at10"]["S21_abs"],
            res["at10"]["energy"], res["elapsed_s"],
            res["mcells_per_s"]), flush=True)
        results.append(res)
    out_path = os.path.join(OUT_DIR, "meep_results.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=1)
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
