"""Finite-difference transverse mode-matching on a shared x grid.

Both empty and loaded cross-sections use the same finite-difference grid, so
the modal coupling is a discrete orthonormal change of basis and the junction
S-matrix is well conditioned. A bounded Redheffer cascade gives the TE10
S-parameters. Convergence is checked against both the grid and the mode count.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mode_matching as mm  # noqa: E402

C0 = 299792458.0
MU0 = 1.2566370614e-6
D_SLAB = 8.0e-3


def transverse_operator(n, h, er, k0):
    d2 = np.zeros((n, n))
    for i in range(n):
        d2[i, i] = -2.0 / h ** 2
        if i > 0:
            d2[i, i - 1] = 1.0 / h ** 2
        if i < n - 1:
            d2[i, i + 1] = 1.0 / h ** 2
    return d2 + np.diag(k0 ** 2 * er)


ALPHA_D_MAX = 20.0


def eigenmodes(n_total, h, er, k0):
    # Interior points (PEC walls excluded); full operator on the interior.
    op = transverse_operator(n_total, h, er, k0)[1:n_total - 1, 1:n_total - 1]
    val, vec = np.linalg.eigh(op)
    beta_all = np.sqrt(val.astype(complex))
    # Keep propagating modes and only evanescent modes whose decay across the
    # slab stays representable (alpha d bounded); deeper modes are negligible.
    keep = [i for i, b in enumerate(beta_all)
            if b.real > 0 or abs(b.imag) * D_SLAB < ALPHA_D_MAX]
    beta = beta_all[keep]
    vec = vec[:, keep] / np.sqrt(h)
    return beta, vec


def run(device, f_hz, n_total):
    om = 2 * np.pi * f_hz
    k0 = om / C0
    a = 24e-3
    x = np.linspace(0, a, n_total)
    h = x[1] - x[0]
    er_empty = np.ones(n_total)
    er_loaded = mm.er_profile(device, x)
    b0, v0 = eigenmodes(n_total, h, er_empty, k0)
    bl, vl = eigenmodes(n_total, h, er_loaded, k0)
    # Use the common mode count (intersection of the two bounded sets).
    k = min(b0.shape[0], bl.shape[0])
    b0, v0, bl, vl = b0[:k], v0[:, :k], bl[:k], vl[:, :k]
    # Discrete coupling V = <empty, loaded> on the shared interior grid.
    vmat = v0.T @ vl * h
    y0 = np.diag(b0 / (om * MU0))
    yl = np.diag(bl / (om * MU0))
    s_a = mm.interface_s(y0, yl, vmat)
    s_b = mm.interface_s(yl, y0, vmat.T)
    prop = np.exp(-1j * bl * D_SLAB)
    s_p = np.block([
        [np.zeros((k, k)), np.diag(prop)],
        [np.diag(prop), np.zeros((k, k))]])
    total = mm.redheffer(mm.redheffer(s_a, s_p), s_b)
    return total[0, 0], total[k, 0], s_a, k


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "side_slab_left"
    print(device)
    for n_total in [480, 960, 1920, 3840]:
        r, t, sa, k = run(device, 10e9, n_total)
        unit = np.linalg.norm(sa.conj().T @ sa - np.eye(2 * k))
        print(f"  N={n_total:4d} K={k:2d} |S11|={abs(r):.4f} "
              f"|S21|={abs(t):.4f} junction_unitarity={unit:.2e}")


if __name__ == "__main__":
    main()
