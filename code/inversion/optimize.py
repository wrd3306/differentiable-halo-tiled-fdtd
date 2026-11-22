"""Differentiable device-optimization closed loop.

The differentiable FDTD (fdtd_diff) records the TE10 modal time series at the
two monitor planes; the same time-gated DFT used for extraction produces complex
S-parameters, and an optimizer drives the slab permittivity er toward a target.
The optimized design is independently verified with the analytic Picture B
reference (no inverse crime).
"""
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
import fdtd_diff as fd  # noqa: E402
import sparam as sp  # noqa: E402


def scaled_config(scale):
    s = scale
    dx = 1e-3 / s
    cfg = dict(Nx=24 * s, Ny=12 * s, Nz=100 * s, dx=dx, dy=dx, dz=dx,
               dt=fg.cfl_dt(dx, dx, dx), f0=10e9, src_z=15 * s, npml=10 * s,
               pulse=dict(t0=1.0e-9, tw=0.3e-9, fc=10e9))
    geom = (1, cfg["Nx"] - 1, 1, cfg["Ny"] - 1, 40 * s, 48 * s)
    return cfg, geom


def gated_phasor(rec, cfg, plane):
    """Build the time-gated Tukey-windowed phasor from recorded modal tensors."""
    ts = torch.stack(rec[plane])
    ts_np = ts.detach().cpu().numpy()
    s0, s1 = sp._gate_span(ts_np, cfg)
    win = torch.tensor(sp._tukey(s1 - s0 + 1), dtype=torch.float32,
                       device=ts.device)
    omega = 2 * np.pi * cfg["f0"]
    nn = torch.arange(s0, s1 + 1, device=ts.device)
    kernel = torch.exp(1j * omega * nn * cfg["dt"])
    return torch.sum(ts[s0:s1 + 1] * win * kernel) * cfg["dt"]


def _planes(cfg):
    scale = int(round(1e-3 / cfg["dx"]))
    return 25 * scale, 70 * scale


def build_gates(cfg, geom, n_steps, device, er_probe=2.0):
    """No-grad pre-run to fix gate spans; build Tukey window tensors.

    The gate location is deterministic and er-insensitive, so it is held fixed
    during differentiable optimization (argmax is not differentiated).
    """
    z_ref, z_tr = _planes(cfg)
    gates = {}
    for z in (z_ref, z_tr):
        block = ("box", geom, er_probe, 0.0)
        fields, const = fg.build_state(cfg, device, block=block)
        ts = []
        with torch.no_grad():
            for n in range(n_steps):
                fg.step(fields, const, n)
                ts.append(sp.modal_amp(fields["Ey"], cfg, z).item())
        ts = np.array(ts)
        s0, s1 = sp._gate_span(ts, cfg)
        win = torch.tensor(sp._tukey(s1 - s0 + 1), dtype=torch.float32,
                           device=device)
        nn = np.arange(s0, s1 + 1)
        ker = torch.tensor(np.exp(1j * 2 * np.pi * cfg["f0"] * nn * cfg["dt"]),
                           device=device)
        gates[z] = (s0, s1, win, ker)
    return gates


def empty_phasors(cfg, geom, n_steps, device, gates, chunk=15):
    """Empty-waveguide gated phasors (fixed reference constants).

    Computed under no_grad so they carry no autograd graph; they are reused as
    constants in every differentiable loaded forward.
    """
    with torch.no_grad():
        _, _, acc = fd.forward(cfg, device, torch.tensor(1.0), geom, n_steps,
                               chunk=chunk, gates=gates)
    z_ref, z_tr = _planes(cfg)
    return acc[z_ref].detach(), acc[z_tr].detach()


def loaded_sparams(er, cfg, geom, n_steps, device, empty, gates, chunk=15):
    """Differentiable gated S-parameters for a scalar er, reusing empty phasors."""
    z_ref, z_tr = _planes(cfg)
    pe, pt = empty
    _, _, acc = fd.forward(cfg, device, er, geom, n_steps, chunk=chunk,
                           gates=gates)
    lr, lt = acc[z_ref], acc[z_tr]
    return (lr - pe) / pe, lt / pt


def run_sparams(er, cfg, geom, n_steps, device, gates, chunk=15):
    """Differentiable empty/loaded gated S-parameters for a scalar er."""
    empty = empty_phasors(cfg, geom, n_steps, device, gates, chunk)
    return loaded_sparams(er, cfg, geom, n_steps, device, empty, gates, chunk)


def objective(er, cfg, geom, n_steps, device, target_abs_s21, empty, gates,
              chunk=15):
    _, s21 = loaded_sparams(er, cfg, geom, n_steps, device, empty, gates, chunk)
    return (abs(s21) - target_abs_s21) ** 2
