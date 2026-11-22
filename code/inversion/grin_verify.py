"""Independent non-autograd verification of the GRIN design (no inverse crime).

Rebuilds the er map from the saved rows and runs the production engine
(engine3d + fdtd_gpu.step, no autograd, no checkpoint) for a 12-ns record,
then extracts S11/S21 per frequency with energy conservation and residual.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "engine"))
import grin_design as gd  # noqa: E402
import devices3d as d3  # noqa: E402
import engine3d as e3  # noqa: E402
import fdtd_gpu as fg  # noqa: E402
import sparam as sp  # noqa: E402

device = torch.device("cuda")
rows = np.load(os.path.join(HERE, "_grin_rows.npy"))

cfg, _, base_t, row_id_t, region_t, reg_k0, reg_k1 = gd.build(device)
Nx, Ny, Nz = cfg["Nx"], cfg["Ny"], cfg["Nz"]
row_id = row_id_t.cpu().numpy()
region = region_t.cpu().numpy()

# numpy er map identical to the differentiable er_from_rows construction
er = np.ones((Nx, Ny, Nz), dtype=np.float32)
for k in range(reg_k0, reg_k1):
    if region[k]:
        er[1:Nx - 1, 1:Ny - 1, k] = rows[row_id[k]]
# fixed slab is already in base_t; transfer it into the numpy map
base = base_t.cpu().numpy()
for k in range(Nz):
    if base[1, 1, k] > 1.0 + 1e-6 and not region[k]:
        er[1:Nx - 1, 1:Ny - 1, k] = base[1, 1, k]

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
print("rows er", np.round(rows, 3))
print("residual %.3e" % resid)
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
