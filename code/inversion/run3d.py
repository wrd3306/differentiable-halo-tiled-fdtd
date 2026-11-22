"""Full-wave 3-D device runner and two-port scattering extraction.

The empty and loaded cases are run with the map-based engine and the TE10 modal
time series are recorded at both monitor planes. Phasors are obtained with an
ungated DFT over a record long enough for all fields to decay; the well-matched
CPML makes late-time wrap-around negligible. A time gate is deliberately not
used because strongly dispersive 3-D devices stretch the pulse beyond any fixed
gate window, and gating then violates the energy balance. The residual field
energy is returned as a convergence check.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(HERE, "..", "engine"))
sys.path.insert(0, HERE)
import engine3d as e3  # noqa: E402
import fdtd_gpu as fg  # noqa: E402
import sparam as sp  # noqa: E402
import devices3d as d3  # noqa: E402


def n_steps_for(cfg, physical_record=4.0e-9):
    return int(np.ceil(physical_record / cfg["dt"]))


def _record(cfg, er_map, n_steps, z_ref, z_tr, device):
    """Run one case; return modal series and final residual field energy."""
    fields, const = e3.build_state_map(cfg, device, er_map)
    rec = {z_ref: [], z_tr: []}
    with torch.no_grad():
        for n in range(n_steps):
            fg.step(fields, const, n)
            rec[z_ref].append(sp.modal_amp(fields["Ey"], cfg, z_ref).item())
            rec[z_tr].append(sp.modal_amp(fields["Ey"], cfg, z_tr).item())
    resid = sum(float((fields[k] ** 2).sum()) for k in
                ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"))
    return {z: np.array(v) for z, v in rec.items()}, resid


def _ungated_phasor(ts, cfg, f=None):
    """Full-record DFT phasor at frequency f (default: the centre frequency).

    The e^{+i omega t} convention matches the analytic transfer-matrix model
    and the differentiable DFT accumulator, so the complex S-parameters share
    one phase reference."""
    f = cfg["f0"] if f is None else f
    nn = np.arange(len(ts))
    return np.sum(ts * np.exp(1j * 2 * np.pi * f * nn * cfg["dt"])) * cfg["dt"]


def run_device(name, scale=2, device=None, er_override=None,
               physical_record=4.0e-9, frequencies=None):
    """Full-wave two-port result for one device."""
    device = device or torch.device("cuda")
    cfg = d3.config3d(scale)
    z_ref, z_tr = d3.planes(cfg)
    n_steps = n_steps_for(cfg, physical_record)

    empty_map = np.ones((cfg["Nx"], cfg["Ny"], cfg["Nz"]), dtype=np.float32)
    loaded_map = d3.er_map(name, cfg, er_override)

    rec_e, resid_e = _record(cfg, empty_map, n_steps, z_ref, z_tr, device)
    rec_l, resid_l = _record(cfg, loaded_map, n_steps, z_ref, z_tr, device)

    if frequencies is None:
        frequencies = np.array([cfg["f0"]])
    sparams = []
    for f in frequencies:
        pe = _ungated_phasor(rec_e[z_ref], cfg, f)
        pt = _ungated_phasor(rec_e[z_tr], cfg, f)
        lr = _ungated_phasor(rec_l[z_ref], cfg, f)
        lt = _ungated_phasor(rec_l[z_tr], cfg, f)
        s11 = (lr - pe) / pe
        s21 = lt / pt
        sparams.append(dict(frequency=float(f), s11=complex(s11),
                            s21=complex(s21),
                            energy=abs(s11) ** 2 + abs(s21) ** 2))
    at0 = sparams[np.argmin(abs(frequencies - cfg["f0"]))]
    return dict(
        name=name, scale=scale, n_steps=n_steps,
        z_ref=z_ref, z_tr=z_tr,
        s11=at0["s11"], s21=at0["s21"],
        energy_balance=at0["energy"], sweep=sparams,
        residual_energy=resid_l,
        series=dict(empty_ref=rec_e[z_ref], empty_tr=rec_e[z_tr],
                    loaded_ref=rec_l[z_ref], loaded_tr=rec_l[z_tr]),
    )
