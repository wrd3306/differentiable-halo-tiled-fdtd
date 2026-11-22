"""Independent 2-D frequency-domain finite-difference (FDFD) tie-breaker.

Solves the driven Helmholtz equation for the TE (Ey) field on a uniform grid
with a stretched-coordinate PML in z and PEC walls in x:

    d/dx (1/sx dEy/dx) + d/dz (1/sz dEy/dz) + k0^2 er Ey = i w mu0 J

There is no time stepping and the linear system is assembled and solved
directly, so the spatial discretization error is isolated from any
time-integration error. A TE10 current sheet drives the problem; empty and
loaded geometries are solved separately and the modal S-parameters are
extracted by the same subtraction used by the FDTD benchmarks. Grid
refinement reveals the true continuous-domain limit of the hard-material
finite-difference model.
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


def er_at(device, x_mm, z_mm):
    """Hard piecewise permittivity matching the voxel material convention."""
    er = np.ones_like(x_mm)
    z_in = (z_mm >= 40.0) & (z_mm < 48.0)
    if device == "full_slab":
        er = np.where(z_in, 4.0, 1.0)
    elif device == "side_slab_left":
        er = np.where(z_in & (x_mm < 12.0), 4.0, 1.0)
    elif device == "centered_slab":
        er = np.where(z_in & (x_mm >= 6.0) & (x_mm < 18.0), 4.0, 1.0)
    elif device == "vacuum":
        er = np.ones_like(x_mm)
    else:
        raise ValueError("no profile for " + device)
    return er


def link_stretch(nz, h, omega):
    """Stretched-coordinate PML factor s on each z-link (between nodes j,j+1).

    s = 1 + sigma/(i w eps0) with a cubic conductivity ramp defined on the
    link midpoint, so the conservative finite-difference operator uses the
    stretching exactly where the flux link sits.
    """
    nlink = nz - 1
    s = np.ones(nlink, dtype=complex)
    d = PML_CELLS * h
    # Cubic-ramp conductivity for a 1e-6 normal-incidence reflection target.
    sigma_opt = SIGMA_MULT * (-4.0 * np.log(1e-6) / (2.0 * ETA0 * d))
    for j in range(nlink):
        # Link midpoint depth into each PML.
        if j < PML_CELLS:
            rho = (PML_CELLS - j - 0.5) * h / d
            sig = sigma_opt * max(rho, 0.0) ** 3
            s[j] = 1.0 + sig / (1j * omega * EPS0)
        elif j >= nz - 1 - PML_CELLS:
            rho = (j + 0.5 - (nz - 1 - PML_CELLS)) * h / d
            sig = sigma_opt * max(rho, 0.0) ** 3
            s[j] = 1.0 + sig / (1j * omega * EPS0)
    return s


def build_and_solve(device, h, omega):
    nx = int(round(A_WG / h))
    nz = int(round(L_WG / h))
    k0 = omega / C0
    x = (np.arange(nx) + 0.5) * h
    z = (np.arange(nz) + 0.5) * h
    er = np.zeros((nx, nz))
    for j in range(nz):
        er[:, j] = er_at(device, x * 1e3, np.full(nx, z[j] * 1e3))

    s_link = link_stretch(nz, h, omega)
    a_link = 1.0 / s_link  # inner flux coefficient on each z-link
    # Node stretching for the outer 1/s factor: the double-s SC operator
    # (1/s) d/dz[(1/s) dE/dz] gives the correct PML wavenumber s k0.
    s_node = np.ones(nz, dtype=complex)
    s_node[:PML_CELLS] = s_link[:PML_CELLS]
    s_node[nz - PML_CELLS:] = s_link[nz - 1 - PML_CELLS:]
    outer = 1.0 / s_node

    def idx(ix, jz):
        return ix * nz + jz

    rows, cols, vals = [], [], []
    n = nx * nz

    for ix in range(nx):
        for jz in range(nz):
            c = idx(ix, jz)
            # x-direction: PEC walls (Ey=0), interior central difference.
            if ix == 0 or ix == nx - 1:
                rows.append(c); cols.append(c); vals.append(1.0)
                continue
            coeff_x = 1.0 / h ** 2
            rows.append(c); cols.append(idx(ix + 1, jz)); vals.append(coeff_x)
            rows.append(c); cols.append(idx(ix - 1, jz)); vals.append(coeff_x)
            # z-direction double-s stretched difference.
            a_p = a_link[jz] if jz + 1 < nz else 0.0
            a_m = a_link[jz - 1] if jz - 1 >= 0 else 0.0
            if jz + 1 < nz:
                rows.append(c); cols.append(idx(ix, jz + 1))
                vals.append(outer[jz] * a_p / h ** 2)
            if jz - 1 >= 0:
                rows.append(c); cols.append(idx(ix, jz - 1))
                vals.append(outer[jz] * a_m / h ** 2)
            center = (-2.0 / h ** 2 + outer[jz] * (-(a_p + a_m) / h ** 2)
                      + k0 ** 2 * er[ix, jz])
            rows.append(c); cols.append(c); vals.append(center)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    # RHS: TE10 current sheet at SRC plane.
    src_j = int(round(SRC_Z_MM * 1e-3 / h)) - 1
    rhs = np.zeros(n, dtype=complex)
    for ix in range(1, nx - 1):
        prof = np.sin(np.pi * x[ix] / A_WG)
        rhs[idx(ix, src_j)] = 1j * omega * MU0 * prof / h

    sol = spsolve(A, rhs)
    field = sol.reshape(nx, nz)
    return x, z, field


def modal_coef(field, x, plane_z, h):
    j = int(round(plane_z * 1e-3 / h)) - 1
    plane = field[:, j]
    prof = np.sin(np.pi * x / A_WG)
    return (2.0 / A_WG) * np.sum(plane[1:-1] * prof[1:-1]) * h


def run_device(device, h):
    omega = 2 * np.pi * F_HZ
    xe, ze, fe = build_and_solve("vacuum", h, omega)
    xl, zl, fl = build_and_solve(device, h, omega)
    in_e = modal_coef(fe, xe, IN_Z_MM, h)
    out_e = modal_coef(fe, xe, OUT_Z_MM, h)
    in_l = modal_coef(fl, xl, IN_Z_MM, h)
    out_l = modal_coef(fl, xl, OUT_Z_MM, h)
    s11 = (in_l - in_e) / in_e
    s21 = out_l / out_e
    return abs(s11), abs(s21), abs(s11) ** 2 + abs(s21) ** 2


def main():
    devices = sys.argv[1:] or ["full_slab", "side_slab_left", "centered_slab"]
    grids_mm = [0.5, 0.25, 0.125, 0.0625]
    report = {}
    for device in devices:
        report[device] = []
        for g in grids_mm:
            h = g * 1e-3
            r, t, e = run_device(device, h)
            report[device].append({"grid_mm": g, "S11": r, "S21": t, "E": e})
            print("%-16s h=%.4fmm |S11|=%.4f |S21|=%.4f E=%.4f"
                  % (device, g, r, t, e))
    path = os.path.join(ROOT, "results", "external", "fdfd_2d_results.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
