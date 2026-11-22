"""Grid-convergence check of the best 28-var GRIN at 0.25 mm (scale 4).

Non-autograd production engine (engine3d + fdtd_gpu.step), 12-ns record.
Re-rasterises the same physical rows on the finer grid and extracts S11/S21
per frequency, so the ~0.11 floor can be judged as a converged physical result
rather than a coarse-grid artifact.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "engine"))
import grin_design as gd  # noqa: E402
import grin_lp as glp  # noqa: E402
import devices3d as d3  # noqa: E402
import engine3d as e3  # noqa: E402
import fdtd_gpu as fg  # noqa: E402
import sparam as sp  # noqa: E402

device = torch.device("cuda")
rows = np.load(os.path.join(HERE, "_grinlp_rows.npy"))

cfg = d3.config3d(4)
cfg["pulse"] = dict(t0=0.8e-9, tw=0.15e-9, fc=10e9)
Nx, Ny, Nz = cfg["Nx"], cfg["Ny"], cfg["Nz"]

row_id = np.zeros(Nz, dtype=np.int64)
region = np.zeros(Nz, dtype=bool)
reg_k0 = max(1, int(round(glp.Z_REG0 * 1e-3 / cfg["dx"])))
reg_k1 = min(Nz - 1, int(round(glp.Z_REG1 * 1e-3 / cfg["dx"])))
for k in range(reg_k0, reg_k1):
    z_mm = k * cfg["dx"] * 1e3
    rid = int((z_mm - glp.Z_REG0) / glp.DZ_MM)
    row_id[k] = min(glp.NZ - 1, max(0, rid))
    region[k] = True

er = np.ones((Nx, Ny, Nz), dtype=np.float32)
for k in range(reg_k0, reg_k1):
    er[1:Nx - 1, 1:Ny - 1, k] = rows[row_id[k]]
sk0 = max(1, int(round(glp.Z_REG1 * 1e-3 / cfg["dx"])))
sk1 = min(Nz - 1, int(round((glp.Z_REG1 + gd.SLAB_D) * 1e-3 / cfg["dx"])))
er[1:Nx - 1, 1:Ny - 1, sk0:sk1] = gd.SLAB_ER

z_ref, z_tr = d3.planes(cfg)
record_ns = 12.0
n_steps = int(np.ceil(record_ns * 1e-9 / cfg["dt"]))


def record(er_np):
    fields, const = e3.build_state_map(cfg, device, er_np)
    rec = {z_ref: [], z_tr: []}
    with torch.no_grad():
        for n in range(n_steps):
            fg.step(fields, const, n)
            rec[z_ref].append(sp.modal_amp(fields["Ey"], cfg, z_ref).item())
            rec[z_tr].append(sp.modal_amp(fields["Ey"], cfg, z_tr).item())
    resid = sum(float((fields[k] ** 2).sum()) for k in
                ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"))
    return rec, resid


def phasor(ts, f):
    nn = np.arange(len(ts))
    return np.sum(ts * np.exp(1j * 2 * np.pi * f * nn * cfg["dt"])) * cfg["dt"]


rec_e, _ = record(np.ones_like(er))
rec_l, resid = record(er)
print("0.25 mm grid, residual %.3e" % resid)
s11s = []
for f in gd.FREQS_HZ:
    pe = phasor(rec_e[z_ref], f)
    pt = phasor(rec_e[z_tr], f)
    s11 = (phasor(rec_l[z_ref], f) - pe) / pe
    s21 = phasor(rec_l[z_tr], f) / pt
    s11s.append(abs(s11))
    print("%.0f GHz |S11| %.4f |S21| %.4f E %.4f" % (
        f / 1e9, abs(s11), abs(s21),
        abs(s11) ** 2 + abs(s21) ** 2))
print("max |S11| %.4f  mean |S11| %.4f" % (max(s11s), float(np.mean(s11s))))
