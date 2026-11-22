"""2-D (x-z) MEEP check of the full-height side slab, fine grid.

A clean 2-D TE (Ey/Hx/Hz) current-sheet run with a fine grid isolates the
same 1-D-in-x modal physics as the mode-matching reference and converges
quickly, giving an independent confirmation of the side-slab TE10 S-answer.
"""
import os
import sys

import meep as mp
import numpy as np

A, L = 24e-3, 100e-3
C = 3.0e8
GRID = float(os.environ.get("MEEP2D_GRID_MM", "0.125")) * 1e-3
PML_T = 4.0e-3
RUN_T = 3.0
COURANT = 0.4044
SRC_Z, IN_Z, OUT_Z = -0.038, -0.025, 0.020
P_T0, P_TW, P_FC = 1.0e-9 * C, 0.3e-9 * C, 10.0e9 / C


def pulse(t):
    return float(np.exp(-((t - P_T0) / P_TW) ** 2)
                 * np.sin(2 * np.pi * P_FC * t))


def profile(pt):
    return float(np.sin(np.pi * (pt.x + A / 2) / A))


def build(loaded):
    geom = []
    if loaded:
        geom.append(mp.Block(
            material=mp.Medium(epsilon=4.0),
            center=mp.Vector3(-A / 4, 0, -0.006),
            size=mp.Vector3(A / 2, mp.inf, 8e-3)))
    geom += [mp.Block(material=mp.perfect_electric_conductor,
                      center=mp.Vector3(xc, 0, 0),
                      size=mp.Vector3(1e-3, mp.inf, L + 2e-3))
             for xc in (-A / 2, A / 2)]
    return geom


def modal(dft, nx):
    if dft.shape[0] == nx:
        dft = dft[1:nx - 1]
    xi = np.arange(1, nx - 1)
    p = np.sin(np.pi * xi * (A / nx) / A)
    return (2 / A) * np.sum(dft * p) * (A / nx)


def run(loaded):
    nx, nz = int(round(A / GRID)), int(round(L / GRID))
    sim = mp.Simulation(
        cell_size=mp.Vector3(nx * GRID, 0, nz * GRID),
        geometry=build(loaded),
        sources=[mp.Source(mp.CustomSource(pulse), mp.Ey,
                           center=mp.Vector3(0, 0, SRC_Z),
                           size=mp.Vector3(A - GRID, 0, 0),
                           amp_func=profile)],
        resolution=1 / GRID, Courant=COURANT, eps_averaging=False,
        boundary_layers=[mp.PML(PML_T, direction=mp.Z)])
    series = {"in": [], "out": [], "t": []}
    ri = mp.Volume(center=mp.Vector3(0, 0, IN_Z),
                   size=mp.Vector3(A - GRID, 0, 0))
    ro = mp.Volume(center=mp.Vector3(0, 0, OUT_Z),
                   size=mp.Vector3(A - GRID, 0, 0))

    def sample(sim):
        series["in"].append(modal(np.asarray(sim.get_array(mp.Ey, ri)), nx))
        series["out"].append(modal(np.asarray(sim.get_array(mp.Ey, ro)), nx))
        series["t"].append(sim.meep_time())

    sim.run(sample, until=RUN_T)
    return {k: np.array(v) for k, v in series.items()}


def phasor(ts, t, f):
    dt = t[1] - t[0]
    return np.sum(ts * np.exp(1j * 2 * np.pi * (f / C) * t)) * dt


def main():
    de = run(False)
    dl = run(True)
    f = 10.0e9
    s11 = (phasor(dl["in"], dl["t"], f) - phasor(de["in"], de["t"], f)) \
        / phasor(de["in"], de["t"], f)
    s21 = phasor(dl["out"], dl["t"], f) / phasor(de["out"], de["t"], f)
    print("2D MEEP side_slab  grid=0.125mm  |S11|=%.4f |S21|=%.4f E=%.4f" % (
        abs(s11), abs(s21), abs(s11) ** 2 + abs(s21) ** 2))


if __name__ == "__main__":
    main()
