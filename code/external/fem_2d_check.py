"""Independent 2-D finite-element (FEM) tie-breaker for the full-height step.

Bilinear quadrilateral elements solve the weak-form TE (Ey) equation

    int [s dW/dx dE/dx + (1/s) dW/dz dE/dz - s k0^2 er W E] dA = i w mu0 int W J

with PEC walls in x (E=0) and a stretched-coordinate PML in z through the
anisotropic factor s. The weak form enforces tangential E and H continuity
across the dielectric interface variationally, so it converges to the true
continuous-domain answer without the hard-node staircase bias of the
finite-difference models.
"""
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")

A_WG = 24.0e-3
L_WG = 100.0e-3
C0 = 3.0e8
EPS0 = 8.854187817e-12
MU0 = 1.2566370614e-6
ETA0 = np.sqrt(MU0 / EPS0)

SRC_Z_MM = 15.0
IN_Z_MM = 25.0
OUT_Z_MM = 70.0
PML_CELLS = 20
SIGMA_MULT = 0.5
F_HZ = 10.0e9

# 2x2 Gauss quadrature on the reference element [0,1]^2.
GP = np.array([0.2113248654051871, 0.7886751345948129])
GW = np.array([0.5, 0.5])


def shape(xi, eta):
    n = np.array([(1 - xi) * (1 - eta), xi * (1 - eta),
                  (1 - xi) * eta, xi * eta])
    dn_dxi = np.array([-(1 - eta), (1 - eta), -eta, eta])
    dn_det = np.array([-(1 - xi), -xi, (1 - xi), xi])
    return n, dn_dxi, dn_det


def er_cell(device, xc_mm, zc_mm):
    """Cell-centered permittivity (constant within each element)."""
    z_in = (zc_mm >= 40.0) & (zc_mm < 48.0)
    if device == "full_slab":
        return 4.0 if z_in else 1.0
    if device == "side_slab_left":
        return 4.0 if (z_in and xc_mm < 12.0) else 1.0
    if device == "centered_slab":
        return 4.0 if (z_in and 6.0 <= xc_mm < 18.0) else 1.0
    if device == "vacuum":
        return 1.0
    raise ValueError("no profile for " + device)


def pml_factor(zc_mm, h, omega):
    """Stretched-coordinate factor s at an element center (z in mm)."""
    d = PML_CELLS * h
    sigma_opt = SIGMA_MULT * (-4.0 * np.log(1e-6) / (2.0 * ETA0 * d))
    zc = zc_mm * 1e-3
    s = 1.0
    if zc < d:
        s = 1.0 + sigma_opt * (1.0 - zc / d) ** 3 / (1j * omega * EPS0)
    elif zc > L_WG - d:
        s = 1.0 + sigma_opt * ((zc - (L_WG - d)) / d) ** 3 / (1j * omega * EPS0)
    return s


def build_and_solve(device, h, omega):
    nx = int(round(A_WG / h))
    nz = int(round(L_WG / h))
    k0 = omega / C0
    nxn, nzn = nx + 1, nz + 1

    # Precompute reference element contribution accumulators at Gauss points.
    rows, cols, vals = [], [], []

    def gdof(ix, jz):
        return ix * nzn + jz

    for ix in range(nx):
        xc = (ix + 0.5) * h
        for jz in range(nz):
            zc = (jz + 0.5) * h
            s = pml_factor(zc * 1e3, h, omega)
            er = er_cell(device, xc * 1e3, zc * 1e3)
            ke = np.zeros((4, 4), dtype=complex)
            me = np.zeros((4, 4), dtype=complex)
            for a in range(2):
                for b in range(2):
                    xi, eta = GP[a], GP[b]
                    n, dxi, deta = shape(xi, eta)
                    dnx = dxi / h
                    dnz = deta / h
                    wq = GW[a] * GW[b] * h ** 2
                    ke += wq * (s * np.outer(dnx, dnx)
                                + (1.0 / s) * np.outer(dnz, dnz))
                    me += wq * s * er * np.outer(n, n)
            ae = ke - k0 ** 2 * me
            local = [(ix, jz), (ix + 1, jz), (ix, jz + 1), (ix + 1, jz + 1)]
            for li in range(4):
                for lj in range(4):
                    rows.append(gdof(*local[li]))
                    cols.append(gdof(*local[lj]))
                    vals.append(ae[li, lj])

    n = nxn * nzn
    A = sp.coo_matrix((vals, (rows, cols))).tocsr()

    # PEC walls in x: enforce E=0 on the two x-boundary node columns.
    wall_dofs = [gdof(0, jz) for jz in range(nzn)] + \
                [gdof(nx, jz) for jz in range(nzn)]
    A = A.tolil()
    for d in wall_dofs:
        A.rows[d] = [d]
        A.data[d] = [1.0]
    A = A.tocsr()

    # TE10 current sheet RHS at the source plane.
    rhs = np.zeros(n, dtype=complex)
    src_j = int(round(SRC_Z_MM * 1e-3 / h))
    for ix in range(1, nx):
        xnode = ix * h
        prof = np.sin(np.pi * xnode / A_WG)
        rhs[gdof(ix, src_j)] = 1j * omega * MU0 * prof
    sol = spsolve(A, rhs)
    return sol.reshape(nxn, nzn), h


def modal_coef(field, plane_z, h):
    j = int(round(plane_z * 1e-3 / h))
    xnodes = np.arange(field.shape[0]) * h
    plane = field[:, j]
    prof = np.sin(np.pi * xnodes / A_WG)
    return (2.0 / A_WG) * np.sum(plane[1:-1] * prof[1:-1]) * h


def run_device(device, h):
    omega = 2 * np.pi * F_HZ
    fe, hh = build_and_solve("vacuum", h, omega)
    fl, _ = build_and_solve(device, h, omega)
    in_e, out_e = modal_coef(fe, IN_Z_MM, h), modal_coef(fe, OUT_Z_MM, h)
    in_l, out_l = modal_coef(fl, IN_Z_MM, h), modal_coef(fl, OUT_Z_MM, h)
    s11 = (in_l - in_e) / in_e
    s21 = out_l / out_e
    return abs(s11), abs(s21), abs(s11) ** 2 + abs(s21) ** 2


def main():
    devices = sys.argv[1:] or ["full_slab", "side_slab_left", "centered_slab"]
    grids_mm = [0.5, 0.25, 0.125]
    report = {}
    for device in devices:
        report[device] = []
        for g in grids_mm:
            r, t, e = run_device(device, g * 1e-3)
            report[device].append({"grid_mm": g, "S11": r, "S21": t, "E": e})
            print("%-16s h=%.4fmm |S11|=%.4f |S21|=%.4f E=%.4f"
                  % (device, g, r, t, e))
    path = os.path.join(ROOT, "results", "external", "fem_2d_results.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
