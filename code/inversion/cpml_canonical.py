"""Canonical 1D CPML in physical units (Taflove & Hagness), for ground truth."""
import numpy as np

EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6
ETA0 = np.sqrt(MU0 / EPS0)
C0 = 1 / np.sqrt(EPS0 * MU0)

dz = 1e-3
dt = 0.5 * dz / C0
N = 600
NP = 12
SRC, MON = 60, 160
R0 = 1e-6
m = 3
KAPPA_MAX = 5
ALPHA_MAX = 0.0  # traveling waves first

sigma = np.zeros(N)
kappa = np.ones(N)
alpha = np.zeros(N)
for k in range(NP):
    d = (k + 1) / NP
    sig = -(m + 1) * np.log(R0) / (2 * ETA0 * dz) * d ** m
    kap = 1 + (KAPMA_MAX := KAPPA_MAX - 1) * d ** m if False else 1 + (KAPPA_MAX - 1) * d ** m
    for z in (NP - 1 - k, N - NP + k):
        sigma[z] = sig
        kappa[z] = kap

b = np.exp(-(sigma / kappa + alpha) / EPS0 * dt)
denom = kappa * (sigma + alpha * kappa)
acoef = np.where(denom > 0, (b - 1) * sigma / np.where(denom > 0, denom, 1), 0.0)

ex = np.zeros(N)
hy = np.zeros(N)
psi_h = np.zeros(N)
psi_e = np.zeros(N)
n_steps = 2100
rec = np.zeros(n_steps)
for n in range(n_steps):
    # H update.
    de = np.zeros(N)
    de[:-1] = (ex[1:] - ex[:-1]) / dz
    psi_h[:] = b * psi_h + acoef * de
    hy[:] += dt / MU0 * (de / kappa + psi_h)
    hy[SRC] += np.exp(-((n - 30.0) / 8.0) ** 2)
    # E update.
    dh = np.zeros(N)
    dh[1:] = (hy[1:] - hy[:-1]) / dz
    psi_e[:] = b * psi_e + acoef * dh
    ex[:] += dt / EPS0 * (dh / kappa + psi_e)
    rec[n] = ex[MON]

gate = 1200
inc = rec[:gate].copy()
ref = np.zeros(n_steps)
ref[gate:] = rec[gate:]
for cpw in (20, 30, 38, 50, 60):
    w = 2 * np.pi / cpw
    nn = np.arange
    di = np.sum(inc * np.exp(1j * w * np.arange(gate)))
    dr = np.sum(ref * np.exp(1j * w * np.arange(n_steps)))
    r = abs(dr / di)
    print(f"{cpw:2d} cells/lambda: |R|={r:.3e} ({20*np.log10(max(r,1e-12)):.1f} dB)")
