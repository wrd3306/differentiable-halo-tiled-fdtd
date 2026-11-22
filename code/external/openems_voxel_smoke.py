"""Smoke test: openEMS ingests the canonical voxel h5 with a soft current sheet."""
import os

import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0

HERE = os.path.dirname(os.path.abspath(__file__))
VOX = os.path.join(HERE, "..", "..", "results", "external", "voxels")
A, B, L = 24.0, 12.0, 100.0
GRID = 0.5
SRC_Z = 12.0
IN_Z, OUT_Z = 25.0, 70.0


def main():
    tag = "full_slab_0.5mm"
    h5 = os.path.join(VOX, tag + ".h5")
    sim_path = os.path.join(VOX, "..", "openems_smoke")
    sim_path = os.path.abspath(sim_path)

    fdtd = openEMS(CellConstantMaterial=True)
    fdtd.SetBoundaryCond(["PEC", "PEC", "PEC", "PEC", "PML_8", "PML_8"])
    # Same narrowband pulse as the self-engine (function-parser string, seconds).
    fdtd.SetCustomExcite(
        "exp(-((t-1e-9)/3e-10)^2)*sin(2*pi*10e9*t)", 10e9, 12e9)

    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(1e-3)
    mesh.AddLine("x", np.arange(0, A + GRID / 2, GRID).tolist())
    mesh.AddLine("y", np.arange(0, B + GRID / 2, GRID).tolist())
    mesh.AddLine("z", np.arange(0, L + GRID / 2, GRID).tolist())

    disc = csx.AddDiscMaterial("voxels", filename=h5)
    print("disc material:", disc.GetName())

    # Soft E-field current sheet on Ey with the TE10 sin spatial weight.
    exc = csx.AddExcitation("te10", 0, [0, 1, 0])
    exc.AddBox([0, 0, SRC_Z], [A, B, SRC_Z])
    exc.SetWeightFunction(["0", "sin(pi*x/24)", "0"])

    # Time-domain E-field dumps (no interpolation, hdf5) at both monitor planes.
    for name, zc in (("in", IN_Z), ("out", OUT_Z)):
        d = csx.AddDump("dump_" + name, dump_type=0, dump_mode=0, file_type=1)
        d.AddBox([0, 0, zc], [A, B, zc])

    fdtd.SetNumberOfTimeSteps(200)
    fdtd.Run(sim_path, cleanup=False, verbose=1)
    print("smoke run finished; files:", os.listdir(sim_path)[:10])


if __name__ == "__main__":
    main()
