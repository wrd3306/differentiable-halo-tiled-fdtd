"""Focused CPML parameter sweep in normalized 1D FDTD (Courant 0.5)."""
import numpy as np

N_PML = 12
S = 0.5
NZ = 600
MON = 160
n_steps = 2100
gate = 1200


def simulate(sig_peak, kap_max, alp_peak, m=3):
    sig = np.zeros(NZ)
    kap = np.ones(NZ)
    alp = np.zeros(NZ)
    for k in range(N_PML):
        d = (N_PML - k) / N_PML
        zr = NZ - 1 - k
        zl = k
        sig[zr] = sig_peak * d ** m
        sig[zl] = sig_peak * d ** m
        kap[zr] = 1 + (kap_max - 1) * d ** m
        kap[zl] = 1 + (kap_max - 1) * d ** m
        alp[zr] = alp_peak * d
        alp[zl] = alp_peak * d
    b = np.exp(-(sig / kap + alp))
    denom = kap * (sig + alp * kap)
    ccoef = np.where(denom > 0, (b - 1) * sig / np.where(denom > 0, denom, 1), 0.0)

    e = np.zeros(NZ)
    h = np.zeros(NZ)
    psi_h = np.zeros(NZ)
    psi_e = np.zeros(NZ)
    rec = np.zeros(n_steps)
    for n in range(n_steps):
        de = np.zeros(NZ)
        de[:-1] = e[1:] - e[:-1]
        psi_h[:] = b * psi_h + ccoef * de
        h[:] += S * (de / kap + psi_h)
        h[60] += np.exp(-((n - 30.0) / 8.0) ** 2)
        dh = np.zeros(NZ)
        dh[1:] = h[1:] - h[:-1]
        psi_e[:] = b * psi_e + ccoef * dh
        e[:] += S * (dh / kap + psi_e)
        rec[n] = e[MON]
    inc = rec[:gate].copy()
    ref = np.zeros(n_steps)
    ref[gate:] = rec[gate:]
    out = {}
    for cpw in (30, 38, 50):
        w = 2 * np.pi / cpw
        tw = np.arange
        dft_i = np.sum(inc * np.exp(1j * w * np.arange(len(inc))))
        dft_r = np.sum(ref * np.exp(1j * w * np.arange(n_steps)))
        out[cpw] = abs(dft_r / dft_i)
    return out


configs = [
    (1.6, 1, 0.0),
    (1.6, 3, 0.0),
    (1.6, 5, 0.0),
    (1.6, 5, 0.02),
    (1.6, 7, 0.02),
    (1.6, 11, 0.0),
    (1.6, 11, 0.02),
    (0.9, 5, 0.02),
    (2.0, 7, 0.0),
]
print("sig_p  kap_m  alp_p | R@30      R@38      R@50")
for sp_p, km, ap in configs:
    r = simulate(sp_p, km, ap)
    print(f"{sp_p:4.1f}  {km:4d}  {ap:4.2f} | "
          f"{r[30]:.2e}  {r[38]:.2e}  {r[50]:.2e}")
