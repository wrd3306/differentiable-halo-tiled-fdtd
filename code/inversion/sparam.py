"""S-parameter extraction and an independent analytic slab reference.

FDTD S11/S21 come from a single-frequency DFT of the TE10 modal amplitude on
two monitor planes, using an empty-waveguide run to separate the incident and
reflected waves. The analytic reference is the exact transfer matrix of a
uniform dielectric slab using the waveguide modal wavenumbers and TE
impedances - a structurally independent forward model (no inverse crime).
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.join(HERE, "..", "engine")
sys.path.insert(0, ENG)
import fdtd_gpu as fg  # noqa: E402


def make_config():
    dx = 1e-3
    cfg = dict(Nx=24, Ny=12, Nz=100, dx=dx, dy=dx, dz=dx,
               dt=fg.cfl_dt(dx, dx, dx), f0=10e9, src_z=15, npml=10,
               pulse=dict(t0=1.0e-9, tw=0.3e-9, fc=10e9))
    geom = (1, cfg["Nx"] - 1, 1, cfg["Ny"] - 1, 40, 48)  # full-cross-section slab
    return cfg, geom


def modal_amp(Ey, cfg, z):
    """TE10 modal amplitude on plane z (mean over the uniform y)."""
    Nx, dx = cfg["Nx"], cfg["dx"]
    a = Nx * dx
    xi = torch.arange(1, Nx - 1, device=Ey.device)
    p = torch.sin(np.pi * xi * dx / a)
    prof = Ey[1:Nx - 1, :, z].mean(dim=1)
    return (2.0 / a) * torch.sum(prof * p) * dx


def source_ramp(n, cfg, ramp_periods=3.0):
    """Hann turn-on over the first ramp_periods to suppress switch-on transients."""
    t_ramp = ramp_periods / cfg["f0"]
    t = n * cfg["dt"]
    if t >= t_ramp:
        return 1.0
    return 0.5 * (1.0 - np.cos(np.pi * t / t_ramp))


def run_dft(cfg, geom, er, z_ref, z_tr, n_steps, dft_start=0, device=None):
    """Run one pulse FDTD case; full-record DFT modal phasors at monitors."""
    block = None if er is None else ("box", geom, er, 0.0)
    fields, const = fg.build_state(cfg, device, block=block)
    omega = 2 * np.pi * cfg["f0"]
    re = {z_ref: 0.0, z_tr: 0.0}
    im = {z_ref: 0.0, z_tr: 0.0}
    with torch.no_grad():
        for n in range(n_steps):
            fg.step(fields, const, n)
            if n >= dft_start:
                wt = omega * n * cfg["dt"]
                cw, sw = np.cos(wt), np.sin(wt)
                for z in (z_ref, z_tr):
                    a = modal_amp(fields["Ey"], cfg, z).item()
                    re[z] += a * cw * cfg["dt"]
                    im[z] += a * sw * cfg["dt"]
    return {z: complex(re[z], im[z]) for z in (z_ref, z_tr)}


def _gate_span(ts, cfg, k=2.1):
    """Deterministic first-pulse gate centered on the pulse peak. The Gaussian
    envelope width tw sets a window of +/- k*tw, capturing the direct pulse and
    excluding the later round-trip boundary reflections."""
    a = np.abs(ts)
    peak_idx = int(np.argmax(a))
    half = k * cfg["pulse"]["tw"] / cfg["dt"]
    return max(0, int(peak_idx - half)), int(peak_idx + half)


def _tukey(n, alpha=0.4):
    """Tukey window: flat centre, cosine taper over the outer alpha fraction."""
    idx = np.arange(n)
    p = alpha * n / 2
    win = np.ones(n)
    m = idx < p
    win[m] = 0.5 * (1 + np.cos(np.pi * (idx[m] / p - 1)))
    m = idx >= n - p
    win[m] = 0.5 * (1 + np.cos(np.pi * ((idx[m] - (n - p)) / p)))
    return win


def run_gated_dft(cfg, geom, er, z_ref, z_tr, n_steps, device=None,
                  smooth=True):
    """Record the modal time series and time-gate the first pulse for DFT.
    A Tukey window on the gate suppresses hard-edge spectral leakage."""
    block = None if er is None else ("box", geom, er, 0.0)
    fields, const = fg.build_state(cfg, device, block=block)
    omega = 2 * np.pi * cfg["f0"]
    rec = {z_ref: [], z_tr: []}
    with torch.no_grad():
        for n in range(n_steps):
            fg.step(fields, const, n)
            for z in (z_ref, z_tr):
                rec[z].append(modal_amp(fields["Ey"], cfg, z).item())
    out = {}
    for z in (z_ref, z_tr):
        ts = np.array(rec[z])
        s0, s1 = _gate_span(ts, cfg)
        seg = ts[s0:s1 + 1]
        win = _tukey(len(seg)) if smooth else np.ones(len(seg))
        nn = np.arange(s0, s1 + 1)
        out[z] = np.sum(seg * win * np.exp(1j * omega * nn * cfg["dt"])) \
            * cfg["dt"]
    return out


def fdtd_sparams(er, cfg, geom, n_steps=2000, dft_start=0, gated=True):
    device = torch.device("cuda")
    z_ref, z_tr = 25, 70
    if gated:
        empty = run_gated_dft(cfg, None, z_ref, z_tr, n_steps, device)
        loaded = run_gated_dft(cfg, geom, er, z_ref, z_tr, n_steps, device)
    else:
        empty = run_dft(cfg, None, None, z_ref, z_tr, n_steps, dft_start, device)
        loaded = run_dft(cfg, geom, er, z_ref, z_tr, n_steps, dft_start, device)
    s11 = (loaded[z_ref] - empty[z_ref]) / empty[z_ref]
    s21 = loaded[z_tr] / empty[z_tr]
    return s11, s21


def analytic_sparams(er, cfg, geom):
    """Exact S-parameters of a uniform dielectric slab.

    Solved as a boundary-value problem in the same phasor convention the FDTD
    DFT uses: the forward (+z) wave carries e^{+ikz}, the backward (-z) wave
    e^{-ikz}, with modal TE impedances Z. This keeps the phase reference aligned
    with the FDTD phasor (no inverse crime and no convention mismatch).
    """
    omega = 2 * np.pi * cfg["f0"]
    a = cfg["Nx"] * cfg["dx"]
    dz = cfg["dz"]
    z0 = geom[4] * dz
    z1 = geom[5] * dz
    k0 = omega / fg.C0
    kc = np.pi / a
    k1 = np.sqrt(k0 ** 2 - kc ** 2)
    k2 = np.sqrt(er * k0 ** 2 - kc ** 2)
    Z1 = omega * fg.MU0 / k1
    Z2 = omega * fg.MU0 / k2

    # Unknowns r, a2, b2, t. Four continuity equations (V and I) at z0, z1.
    M = np.zeros((4, 4), dtype=complex)
    y = np.zeros(4, dtype=complex)
    # V continuity at z0: incident + r = a2 + b2
    M[0] = [np.exp(-1j * k1 * z0), -np.exp(1j * k2 * z0),
            -np.exp(-1j * k2 * z0), 0]
    y[0] = -np.exp(1j * k1 * z0)
    # I continuity at z0: (incident - r)/Z1 = (a2 - b2)/Z2
    M[1] = [-np.exp(-1j * k1 * z0) / Z1, -np.exp(1j * k2 * z0) / Z2,
            np.exp(-1j * k2 * z0) / Z2, 0]
    y[1] = -np.exp(1j * k1 * z0) / Z1
    # V continuity at z1: a2 + b2 = t
    M[2] = [0, np.exp(1j * k2 * z1), np.exp(-1j * k2 * z1),
            -np.exp(1j * k1 * z1)]
    # I continuity at z1: (a2 - b2)/Z2 = t/Z1
    M[3] = [0, np.exp(1j * k2 * z1) / Z2, -np.exp(-1j * k2 * z1) / Z2,
            -np.exp(1j * k1 * z1) / Z1]
    r, a2, b2, t = np.linalg.solve(M, y)
    return r, t


def compare(er):
    cfg, geom = make_config()
    s11, s21 = fdtd_sparams(er, cfg, geom)
    a11, a21 = analytic_sparams(er, cfg, geom)
    print(f"er={er}  slab d={(geom[5]-geom[4])*cfg['dz']*1e3:.1f} mm  f0={cfg['f0']/1e9:.1f} GHz")
    print(f"  FDTD     |S11|={abs(s11):.4f} |S21|={abs(s21):.4f}")
    print(f"  analytic |S11|={abs(a11):.4f} |S21|={abs(a21):.4f}")
    print(f"  abs discrepancy  d|S11|={abs(abs(s11)-abs(a11)):.4f} "
          f"d|S21|={abs(abs(s21)-abs(a21)):.4f}")
    return s11, s21, a11, a21


if __name__ == "__main__":
    compare(4.0)
