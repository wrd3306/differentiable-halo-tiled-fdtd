"""From-scratch modal mode-matching reference (independent of the FDTD codes).

For full-height, y-uniform waveguide obstacles the field is y-invariant and
only the TE_m0 family (Ey, Hx, Hz; sin(m pi x / a) transverse profiles) is
excited. The empty-guide TE_m0 modes form an orthonormal basis. The
dielectric-loaded section is solved as a real symmetric transverse
eigenproblem in that basis, giving its discrete eigenmodes and propagation
constants. Tangential E and H are matched at the two interfaces and the
global block system is solved for the TE10 reflection and transmission.

This is a spectral modal method with no time stepping and no shared code with
the FDTD solvers, so it is an independent fourth reference.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "code", "engine"))
import fdtd_gpu as fg  # noqa: E402

A_WG = 24e-3
D_SLAB = 8.0e-3
NQUAD = 4001
FMIN_HZ, FMAX_HZ, NFREQ = 7.0e9, 12.0e9, 26


def er_profile(device, x):
    """Piecewise permittivity of the full-height device along x."""
    er = np.ones_like(x)
    if device == "full_slab":
        er[:] = 4.0
    elif device == "side_slab_left":
        er[x < 12e-3] = 4.0
    elif device == "side_slab_right":
        er[x >= 12e-3] = 4.0
    elif device == "centered_slab":
        er[(x >= 6e-3) & (x < 18e-3)] = 4.0
    elif device == "two_material":
        er[x < 12e-3] = 4.0
        er[x >= 12e-3] = 2.0
    else:
        raise ValueError("no full-height analytic profile for " + device)
    return er


def solve_section(device, f_hz, nmodes):
    omega = 2 * np.pi * f_hz
    k0 = omega / fg.C0
    x = np.linspace(0, A_WG, NQUAD)
    dx = x[1] - x[0]
    er = er_profile(device, x)

    # Orthonormal empty-guide TE_m0 basis on the quadrature grid.
    e = np.zeros((nmodes, NQUAD))
    kc = np.zeros(nmodes)
    for m in range(nmodes):
        m1 = m + 1
        e[m] = np.sqrt(2.0 / A_WG) * np.sin(m1 * np.pi * x / A_WG)
        kc[m] = m1 * np.pi / A_WG

    # Overlap matrix <m|er|n> by trapezoidal quadrature of known functions.
    ov = np.zeros((nmodes, nmodes))
    for m in range(nmodes):
        for n in range(m, nmodes):
            v = np.trapz(e[m] * er * e[n], x)
            ov[m, n] = v
            ov[n, m] = v

    # Empty-guide propagation constants. Evanescent modes carry an imaginary
    # beta (i alpha); clamping them to zero would discard the reactive
    # near-field at the interface.
    beta0 = np.sqrt((k0 ** 2 - kc ** 2).astype(complex))
    # TE wave admittance. A below-cutoff TE mode is inductive: its impedance is
    # +j X and its admittance is -j B, so the evanescent branch must use the
    # negative-imaginary sign (Y = -i alpha/(w mu0)), not +i.
    a0 = np.abs(np.imag(beta0))
    p0 = np.real(beta0) > 0
    y0 = np.where(p0, np.real(beta0) / (omega * fg.MU0),
                  -1j * a0 / (omega * fg.MU0))

    # Loaded transverse eigenproblem: A = -kc^2 + k0^2 <er>.
    amat = np.diag(-kc ** 2) + k0 ** 2 * ov
    eig2, c = np.linalg.eigh(amat)
    beta_l = np.sqrt(eig2.astype(complex))
    # Loaded eigenmodes f_p = sum_m c_mp e_m; same inductive admittance sign.
    al = np.abs(np.imag(beta_l))
    pl = np.real(beta_l) > 0
    yl = np.where(pl, np.real(beta_l) / (omega * fg.MU0),
                  -1j * al / (omega * fg.MU0))
    return y0, yl, c, beta_l, beta0


def interface_s(ya, yb, vmat):
    """Scattering matrix of a single modal junction.

    ya, yb are diagonal admittance matrices of the two regions; vmap holds the
    mode-overlap coupling <e_a, e_b>. Returns the 2x2 block S-matrix mapping
    incoming modal amplitudes to outgoing ones.
    """
    mmat = ya @ vmat + vmat @ yb
    m_inv = np.linalg.inv(mmat)
    two_ya = 2.0 * ya
    s21 = m_inv @ two_ya
    s11 = vmat @ s21 - np.eye(vmat.shape[0])
    s22 = -m_inv @ (ya @ vmat - vmat @ yb)
    s12 = vmat - vmat @ (ya @ vmat - vmat @ yb) @ m_inv
    return np.block([[s11, s12], [s21, s22]])


def redheffer(s, sp):
    """Redheffer star product of two compatible two-port S-matrices."""
    n = s.shape[0] // 2
    s11, s12 = s[:n, :n], s[:n, n:]
    s21, s22 = s[n:, :n], s[n:, n:]
    q11, q12 = sp[:n, :n], sp[:n, n:]
    q21, q22 = sp[n:, :n], sp[n:, n:]
    i_s = np.linalg.inv(np.eye(n) - s22 @ q11)
    i_q = np.linalg.inv(np.eye(n) - q11 @ s22)
    c11 = s11 + s12 @ q11 @ i_s @ s21
    c12 = s12 @ i_q @ q12
    c21 = q21 @ i_s @ s21
    c22 = q22 + q21 @ s22 @ i_q @ q12
    return np.block([[c11, c12], [c21, c22]])


def run_device(device, f_hz, nmodes=24):
    """Direct global block solve with interface-normalized loaded amplitudes.

    Loaded forward/backward amplitudes are defined at the entry interface and
    carry bounded propagation factors, so the evanescent exponentials never
    grow and the continuity system stays well conditioned.
    """
    y0, yl, c, beta_l, beta0 = solve_section(device, f_hz, nmodes)
    k = nmodes
    yi = np.diag(y0)
    yld = np.diag(yl)
    # Bounded one-way decay factor. Propagating modes carry a unit-modulus
    # phase; evanescent modes decay (e^{-alpha d}) rather than grow, so the
    # continuity system stays well conditioned at large mode counts.
    real_b = np.real(beta_l)
    alpha = np.abs(np.imag(beta_l))
    is_prop = real_b > 0
    fwd = np.where(is_prop, np.exp(-1j * real_b * D_SLAB),
                   np.exp(-alpha * D_SLAB)).astype(complex)
    inc = np.zeros(k, dtype=complex)
    inc[0] = 1.0
    # A = forward amplitude at entry; B = backward amplitude at exit. At the
    # opposite interface each becomes fwd times its originating value.
    # Entry E: inc + R = c(A + fwd B); H: yi(inc - R) = c yld(A - fwd B)
    # Exit  E: c(fwd A + B) = T;       H: c yld(fwd A - B) = yi T
    rows = [
        np.block([np.eye(k), -c, -c * fwd, np.zeros((k, k))]),
        np.block([-yi, -c @ yld, c @ yld * fwd, np.zeros((k, k))]),
        np.block([np.zeros((k, k)), c * fwd, c, -np.eye(k)]),
        np.block([np.zeros((k, k)), c @ yld * fwd, -c @ yld, -yi])]
    rhs = np.concatenate([-inc, -yi @ inc, np.zeros(k), np.zeros(k)])
    sol = np.linalg.solve(np.vstack(rows), rhs)
    return sol[0], sol[3 * k]


def run_device_old(device, f_hz, nmodes=24):
    y0, yl, c, beta_l, beta0 = solve_section(device, f_hz, nmodes)
    yi = np.diag(y0)
    yl_diag = np.diag(yl)
    s_a = interface_s(yi, yl_diag, c)
    s_b = interface_s(yl_diag, yi, c.T)
    prop = np.exp(-1j * beta_l * D_SLAB)
    s_p = np.block([
        [np.zeros((nmodes, nmodes)), np.diag(prop)],
        [np.diag(prop), np.zeros((nmodes, nmodes))]])
    total = redheffer(redheffer(s_a, s_p), s_b)
    return total[0, 0], total[nmodes, 0]


def main():
    devices = sys.argv[1:] or [
        "full_slab", "side_slab_left", "centered_slab", "two_material"]
    freqs = np.linspace(FMIN_HZ, FMAX_HZ, NFREQ)
    out = []
    for device in devices:
        s11 = np.zeros(NFREQ, dtype=complex)
        s21 = np.zeros(NFREQ, dtype=complex)
        for fi, f in enumerate(freqs):
            s11[fi], s21[fi] = run_device(device, f)
        i10 = int(np.argmin(abs(freqs - 10e9)))
        out.append({
            "name": device, "solver": "mode-matching",
            "frequencies_hz": freqs.tolist(),
            "S11": [[float(z.real), float(z.imag)] for z in s11],
            "S21": [[float(z.real), float(z.imag)] for z in s21],
            "at10": {
                "S11_abs": float(abs(s11[i10])),
                "S21_abs": float(abs(s21[i10])),
                "energy": float(abs(s11[i10]) ** 2 + abs(s21[i10]) ** 2)},
        })
        print("%-16s |S11|=%.4f |S21|=%.4f E=%.4f" % (
            device, abs(s11[i10]), abs(s21[i10]),
            abs(s11[i10]) ** 2 + abs(s21[i10]) ** 2))
    path = os.path.join(ROOT, "results", "external",
                        "mode_matching_results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
