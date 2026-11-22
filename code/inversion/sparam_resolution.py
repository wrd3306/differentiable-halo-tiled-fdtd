"""Grid-resolution convergence of the dielectric-window S-parameters.

The physical device is held fixed (24x12 mm guide, 100 mm long, 8 mm slab at
z=40..48 mm, source at 15 mm, monitors at 25/70 mm, 10 mm PML). The number of
cells per mm (scale) is swept, so the FDTD S-parameters must converge to the
independent analytic transfer-matrix values as the grid is refined. Results are
saved to results/sparam_resolution.json.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.join(HERE, "..", "engine")
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ENG)
sys.path.insert(0, HERE)
import fdtd_gpu as fg  # noqa: E402
import sparam as sp  # noqa: E402


def make_scaled_config(scale):
    dx = 1e-3 / scale
    s = scale
    cfg = dict(Nx=24 * s, Ny=12 * s, Nz=100 * s, dx=dx, dy=dx, dz=dx,
               dt=fg.cfl_dt(dx, dx, dx), f0=10e9, src_z=15 * s, npml=10 * s,
               pulse=dict(t0=1.0e-9, tw=0.3e-9, fc=10e9))
    geom = (1, cfg["Nx"] - 1, 1, cfg["Ny"] - 1, 40 * s, 48 * s)
    return cfg, geom


def main(er=4.0, scales=(1, 2, 4, 8)):
    device = torch.device("cuda")
    z_ref_phys, z_tr_phys = 25, 70
    out = []
    for scale in scales:
        cfg, geom = make_scaled_config(scale)
        z_ref = z_ref_phys * scale
        z_tr = z_tr_phys * scale
        n_steps = 2000 * scale
        empty = sp.run_gated_dft(cfg, None, None, z_ref, z_tr, n_steps, device)
        loaded = sp.run_gated_dft(cfg, geom, er, z_ref, z_tr, n_steps, device)
        s11 = (loaded[z_ref] - empty[z_ref]) / empty[z_ref]
        s21 = loaded[z_tr] / empty[z_tr]
        ana = sp.analytic_sparams(er, cfg, geom)
        a11, a21 = ana
        row = dict(scale=scale, dx_mm=cfg["dx"] * 1e3,
                   cells_per_wavelength_dielectric=round(
                       (fg.C0 / np.sqrt(er) / cfg["f0"]) / cfg["dx"], 1),
                   S11=abs(s11), S21=abs(s21),
                   energy=abs(s11) ** 2 + abs(s21) ** 2,
                   analytic_S11=abs(a11), analytic_S21=abs(a21),
                   dS11=abs(abs(s11) - abs(a11)),
                   dS21=abs(abs(s21) - abs(a21)),
                   S11_phase=float(np.angle(s11)),
                   S21_phase=float(np.angle(s21)),
                   analytic_S11_phase=float(np.angle(a11)),
                   analytic_S21_phase=float(np.angle(a21)),
                   complex_dS11=float(abs(s11 - a11)),
                   complex_dS21=float(abs(s21 - a21)),
                   phase_dS11=float(np.angle(s11 * np.conj(a11))),
                   phase_dS21=float(np.angle(s21 * np.conj(a21))))
        out.append(row)
        print(json.dumps(row, indent=2))
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "sparam_resolution.json"),
              "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
