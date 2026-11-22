"""Tune and verify a standard CPML (sigma + kappa + alpha) in normalized 1D.

Dimensionless per-step quantities:
  sig_n = (sigma/eps0) dt,  alp_n = (alpha/eps0) dt,  kappa >= 1
  b = exp(-(sig_n/kappa + alp_n))
  a = (b-1)*sig_n / (kappa*(sig_n + alp_n*kappa))
  psi = b*psi + a*diff ;  curl_eff = diff/kappa + psi
The old simplified (sigma-only) PML is measured for comparison.
"""
import numpy as np

N_PML = 12
S = 0.5
NZ = 600
MON = 160
n_steps = 2100
gate = 1200


def profiles(kind):
    sig = np.zeros(NZ)
    kap = np.ones(NZ)
    alp = np.zeros(NZ)
    for k in range(N_PML):
        d = (N_PML - k) / N_PML  # d=1 at outer boundary, small at inner edge
        z = NZ - 1 - k
        if kind == "old":
            sig[z] = 0.465 * d ** 3
        else:
            sig[z] = 1.293 * d ** 3          # 0.8*(m+1) optimal, cubic
            kap[z] = 1 + (5.0 - 1) * d ** 3  # kappa_max = 5
            alp[z] = 0.05 * d                # alpha ramps in
    return sig, kap, alp


def simulate(kind):
    sig, kap, alp = profiles(kind)
    if kind == "old":
        b = np.exp(-sig)
        c = b - 1.0
    else:
        b = np.exp(-(sig / kap + alp))
        denom = kap * (sig + alp * kap)
        c = np.where(denom > 0, (b - 1.0) * sig / np.where(denom > 0, denom, 1), 0.0)
    e = np.zeros(NZ)
    h = np.zeros(NZ)
    psi_h = np.zeros(NZ)
    psi_e = np.zeros(NZ)
    rec = np.zeros(n_steps)
    for n in range(n_steps):
        de = np.zeros(NZ)
        de[:-1] = e[1:] - e[:-1]
        if kind == "old":
            psi_h[:] = b * psi_h + c * de
            h[:] += S * (de + psi_h)
        else:
            psi_h[:] = b * psi_h + c * de
            h[:] += S * (de / kap + psi_h)
        h[60] += np.exp(-((n - 30.0) / 8.0) ** 2)
        dh = np.zeros(NZ)
        dh[1:] = h[1:] - h[:-1]
        if kind == "old":
            psi_e[:] = b * psi_e + c * dh
            e[:] += S * (dh + psi_e)
        else:
            psi_e[:] = b * psi_e + c * dh
            e[:] += S * (dh / kap + psi_e)
        rec[n] = e[MON]
    inc = rec[:gate].copy()
    ref = np.zeros(n_steps)
    ref[gate:] = rec[gate:]
    return inc, ref


def dft(sig_arr, w):
    return np.sum(sig_arr * np.exp(1j * w * np.arange(len(sig_arr))))


for kind in ("old", "cpml"):
    inc, ref = simulate(kind)
    print(f"--- {kind} ---")
    for cpw in (25, 30, 34, 38, 42, 50, 60):
        w = 2 * np.pi / cpw
        r = abs(dft(ref, w) / dft(inc, w))
        print(f"  {cpw:2d} cells/lambda: |R|={r:.3e} "
              f"({20*np.log10(max(r,1e-12)):.1f} dB)")
