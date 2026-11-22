"""Fabrication sensitivity of the continuous 28-var GRIN (non-autograd).

Applies realistic process tolerances to the nominal design and re-simulates
each with the production engine (engine3d + fdtd_gpu.step, no autograd, 10-ns
record): (i) uniform er over/under-shoot by +/-5% of the dielectric contrast;
(ii) independent random er noise (5% std); (iii) +/-1-voxel layer-boundary
shift. All results are real full-wave solves.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "engine"))
import grin_lp as glp  # noqa: E402
import grin_design as gd  # noqa: E402
import devices3d as d3  # noqa: E402
import engine3d as e3  # noqa: E402
import fdtd_gpu as fg  # noqa: E402
import sparam as sp  # noqa: E402

device = torch.device("cuda")
nom_rows = np.load(os.path.join(HERE, "_grinlp_rows.npy"))
cfg, _, base_t, row_id_t, region_t = glp.build(device)
Nx, Ny, Nz = cfg["Nx"], cfg["Ny"], cfg["Nz"]
row_id = row_id_t.cpu().numpy()
region = region_t.cpu().numpy()
base = base_t.cpu().numpy()
z_ref, z_tr = d3.planes(cfg)
record_ns = 10.0
n_steps = int(np.ceil(record_ns * 1e-9 / cfg["dt"]))


def er_from(rows):
    er = np.ones((Nx, Ny, Nz), dtype=np.float32)
    for k in range(Nz):
        if region[k]:
            er[1:Nx - 1, 1:Ny - 1, k] = rows[row_id[k]]
        elif base[1, 1, k] > 1.0 + 1e-6:
            er[1:Nx - 1, 1:Ny - 1, k] = base[1, 1, k]
    return er


def record(er_np):
    fields, const = e3.build_state_map(cfg, device, er_np)
    rec = {z_ref: [], z_tr: []}
    with torch.no_grad():
        for n in range(n_steps):
            fg.step(fields, const, n)
            rec[z_ref].append(sp.modal_amp(fields["Ey"], cfg, z_ref).item())
            rec[z_tr].append(sp.modal_amp(fields["Ey"], cfg, z_tr).item())
    return rec


def phasor(ts, f):
    nn = np.arange(len(ts))
    return np.sum(ts * np.exp(1j * 2 * np.pi * f * nn * cfg["dt"])) * cfg["dt"]


rec_e = record(np.ones((Nx, Ny, Nz), dtype=np.float32))


def max_s11(er_np):
    rec = record(er_np)
    vals = []
    for f in gd.FREQS_HZ:
        pe = phasor(rec_e[z_ref], f)
        s11 = (phasor(rec[z_ref], f) - pe) / pe
        vals.append(abs(s11))
    return max(vals), float(np.mean(vals))


def shift_boundaries(er_np, direction):
    """Move every design-region row boundary by one voxel (in x-invariant z)."""
    out = er_np.copy()
    reg_k = np.where(region)[0]
    # boundaries between distinct row ids
    boundaries = [reg_k[0]]
    for a, b in zip(reg_k[:-1], reg_k[1:]):
        if row_id[b] != row_id[a]:
            boundaries.append(b)
    if direction > 0:
        for b in reversed(boundaries):
            if b + 1 < Nz and region[b + 1]:
                out[1:Nx - 1, 1:Ny - 1, b + 1] = out[1, 1, b]
    else:
        for b in boundaries:
            if b - 1 >= 1 and region[b - 1]:
                out[1:Nx - 1, 1:Ny - 1, b - 1] = out[1, 1, b]
    return out


print("nominal max|S11| %.4f mean %.4f" % max_s11(er_from(nom_rows)),
      flush=True)
for sign, lab in ((1.05, "er +5%"), (0.95, "er -5%")):
    rows = 1.0 + sign * (nom_rows - 1.0)
    print("%s max|S11| %.4f mean %.4f" % (lab, *max_s11(er_from(rows))),
          flush=True)

rng = np.random.default_rng(0)
rows = 1.0 + (nom_rows - 1.0) * (
    1.0 + 0.05 * rng.standard_normal(glp.NZ))
print("er noise 5%% max|S11| %.4f mean %.4f" % max_s11(er_from(rows)),
      flush=True)

nom_er = er_from(nom_rows)
print("boundary +1 voxel max|S11| %.4f mean %.4f" %
      max_s11(shift_boundaries(nom_er, +1)), flush=True)
print("boundary -1 voxel max|S11| %.4f mean %.4f" %
      max_s11(shift_boundaries(nom_er, -1)), flush=True)
