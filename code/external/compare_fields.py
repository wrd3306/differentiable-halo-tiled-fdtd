"""Field-level verification: self vs aligned openEMS on the empty and loaded guide.

Compares the modal time series and the dt-normalized DFT phasors, which are
independent of the different time steps. The empty-guide pulse isolates the
source and projection normalization; the loaded pulse then tests the field
solver through the dielectric.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, "..", "..", "results", "external")
RES = os.path.join(HERE, "..", "..", "results")
A = 24.0
GRID = 0.5
NX = int(round(A / GRID))
C0 = 299792458.0
DT_OE = GRID * 1e-3 / (C0 * np.sqrt(3.0))
DT_SELF = 0.7 * DT_OE


def oe_modal(sim_path):
    import h5py
    with h5py.File(os.path.join(sim_path, "dump_in.h5"), "r") as fh:
        frames = sorted(fh["FieldData/TD"].keys())
        ey = np.stack([fh["FieldData/TD"][k][1, ..., 0] for k in frames])
        steps = np.array([int(k) for k in frames], dtype=np.float64)
    x_int = ey[:, 1:NX - 1, :].mean(axis=2)
    a_m = A * 1e-3
    dx_m = GRID * 1e-3
    xg = np.arange(1, NX - 1)
    p_int = np.sin(np.pi * xg * dx_m / a_m)
    modal = (2.0 / a_m) * np.sum(x_int * p_int[None, :], axis=1) * dx_m
    return modal, steps


def phasor(ts, steps, dt, f):
    t = steps * dt
    return np.sum(ts * np.exp(1j * 2 * np.pi * f * t)) * dt


def main():
    oe_empty = os.path.join(EXT, "openems_vx_full_slab_0.5mm_empty")
    oe_load = os.path.join(EXT, "openems_vx_full_slab_0.5mm_load")
    oe_m, oe_s = oe_modal(oe_empty)
    oe_l, oe_ls = oe_modal(oe_load)

    z = np.load(os.path.join(RES, "devices3d_series.npz"))
    self_empty = z["full_slab__empty_ref"]
    self_load = z["full_slab__loaded_ref"]
    self_steps = np.arange(len(self_empty))

    f = 10e9
    pe_oe = phasor(oe_m, oe_s, DT_OE, f)
    pe_self = phasor(self_empty, self_steps, DT_SELF, f)
    print("empty input modal peak  self=%.5e openEMS=%.5e" % (
        np.abs(self_empty).max(), np.abs(oe_m).max()))
    print("empty DFT phasor @10G   self=%.5e openEMS=%.5e ratio=%.4f" % (
        abs(pe_self), abs(pe_oe), abs(pe_oe) / abs(pe_self)))
    pl_oe = phasor(oe_l, oe_ls, DT_OE, f)
    pl_self = phasor(self_load, self_steps, DT_SELF, f)
    print("loaded DFT phasor @10G  self=%.5e openEMS=%.5e ratio=%.4f" % (
        abs(pl_self), abs(pl_oe), abs(pl_oe) / abs(pl_self)))


if __name__ == "__main__":
    main()
