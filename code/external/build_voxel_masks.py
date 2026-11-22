"""Build the canonical hard-voxel material mask shared by every solver.

The self-developed engine rasterizes each device with devices3d.rasterize
(hard staircase, material clamped to interior cells). This script regenerates
that exact mask for every device and every grid, and exports two portable
artifacts so that MEEP and openEMS consume the identical voxel array rather
than re-deriving geometry from analytic boxes:

  1. <device>_<grid>mm.npz : uint8 index array + mesh metadata. A loader
     reconstructs the float relative-permittivity map from the material DB.
  2. <device>_<grid>mm.h5  : CSXCAD DiscMaterial HDF5 (version 2) read
     directly by openEMS AddDiscMaterial.

No geometry is re-derived here: the numpy array is the same array the
self-engine steps with, so the cross-solver comparison has one source of truth.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "inversion"))

import devices3d as d3  # noqa: E402

OUT_DIR = os.path.join(HERE, "..", "..", "results", "external", "voxels")
os.makedirs(OUT_DIR, exist_ok=True)

# Global material database. Index 0 is the background (vacuum); the PEC walls
# are enforced separately by every solver, never as a voxel material.
DB_ER = np.array([1.0, 2.0, 4.0], dtype=np.float32)
DB_KAPPA = np.array([0.0, 0.0, 0.0], dtype=np.float32)
DB_DENSITY = np.array([0.0, 0.0, 0.0], dtype=np.float32)
DB_NAMES = ["Background", "er2", "er4"]
ER_TO_INDEX = {float(er): i for i, er in enumerate(DB_ER)}

GRIDS_MM = [0.5, 0.25, 0.125]
DEVICE_NAMES = [n for n in d3.DEVICES.keys() if n != "ar_match"]


def index_array(er_map):
    """Map a float er array to the uint8 material-index array."""
    flat = er_map.reshape(-1)
    out = np.empty(flat.shape, dtype=np.uint8)
    for er, idx in ER_TO_INDEX.items():
        out[np.isclose(flat, er)] = idx
    return out.reshape(er_map.shape)


def write_openems_h5(path, idx, dx):
    """Write a CSXCAD DiscMaterial version-2 file (mesh in metres).

    CSXCAD reads x fastest; h5py stores C-order, so the array is transposed to
    (nz, ny, nx). The mesh lines are the primary-cell boundaries in metres.
    """
    import h5py

    nx, ny, nz = idx.shape
    data_c = np.ascontiguousarray(idx.transpose(2, 1, 0), dtype=np.uint8)
    mesh_x = np.arange(nx + 1, dtype=np.float64) * dx
    mesh_y = np.arange(ny + 1, dtype=np.float64) * dx
    mesh_z = np.arange(nz + 1, dtype=np.float64) * dx

    if os.path.exists(path):
        os.remove(path)
    with h5py.File(path, "w") as fh:
        fh.attrs["Version"] = np.float64(2.0)
        ds = fh.create_dataset(
            "/DiscData", data=data_c, dtype=np.uint8,
            compression="gzip", compression_opts=9)
        ds.attrs["DB_Size"] = np.int32(len(DB_ER))
        ds.attrs["epsR"] = DB_ER
        ds.attrs["kappa"] = DB_KAPPA
        ds.attrs["density"] = DB_DENSITY
        ds.attrs["Name"] = ",".join(DB_NAMES)
        fh.create_dataset("/mesh/x", data=mesh_x.astype(np.float32))
        fh.create_dataset("/mesh/y", data=mesh_y.astype(np.float32))
        fh.create_dataset("/mesh/z", data=mesh_z.astype(np.float32))


def main():
    manifest = {}
    for grid in GRIDS_MM:
        scale = int(round(1.0 / grid))
        cfg = d3.config3d(scale)
        dx = cfg["dx"]
        for name in DEVICE_NAMES:
            er_map = d3.rasterize(d3.DEVICES[name], cfg)
            idx = index_array(er_map)

            tag = "%s_%gmm" % (name, grid)
            npz_path = os.path.join(OUT_DIR, tag + ".npz")
            h5_path = os.path.join(OUT_DIR, tag + ".h5")

            np.savez_compressed(
                npz_path, idx=idx, grid_mm=np.float64(grid),
                db_er=DB_ER, Nx=cfg["Nx"], Ny=cfg["Ny"], Nz=cfg["Nz"])
            write_openems_h5(h5_path, idx, dx)

            n_nonvac = int(np.count_nonzero(idx))
            manifest[tag] = {
                "device": name, "grid_mm": grid,
                "shape": list(idx.shape), "nonvacuum_voxels": n_nonvac,
                "npz": os.path.relpath(npz_path, OUT_DIR),
                "h5": os.path.relpath(h5_path, OUT_DIR)}
            print("%-28s shape=%-18s nonvacuum=%d" % (
                tag, str(list(idx.shape)), n_nonvac), flush=True)

    with open(os.path.join(OUT_DIR, "voxel_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print("wrote manifest:", len(manifest), "artifacts ->", OUT_DIR)


if __name__ == "__main__":
    main()
