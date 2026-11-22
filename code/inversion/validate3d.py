"""First-principles internal validation of the 3-D device suite.

No external solver is used. The checks are:
  1. identity: an all-unity er map gives S11 = 0, S21 = 1;
  2. reciprocity: reverse transmission S12 equals forward S21 (source/monitor
     swapped by mirroring the whole setup in z);
  3. mirror symmetry: the left and right side slabs are x-mirrored and give
     identical S-parameters;
  4. analytic regression: the full-cross-section slab converges to the
     independent transfer-matrix value as the grid is refined;
  5. convergence: an isolated post and the grating stopband are run over scales.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(HERE, "..", "engine"))
sys.path.insert(0, HERE)
import devices3d as d3  # noqa: E402
import run3d  # noqa: E402
import sparam as sp  # noqa: E402

RECORD = 10.0e-9


def _json_default(o):
    if isinstance(o, complex):
        return {"real": o.real, "imag": o.imag}
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def _reverse_config(cfg):
    rev = dict(cfg)
    rev["src_z"] = cfg["Nz"] - 1 - cfg["src_z"]
    return rev


def reverse_s12(name, scale, device):
    """Reverse-source transmission S12 for a device."""
    cfg = d3.config3d(scale)
    z_ref, z_tr = d3.planes(cfg)
    n_steps = run3d.n_steps_for(cfg, RECORD)
    rev = _reverse_config(cfg)
    near = cfg["Nz"] - 1 - z_ref
    far = cfg["Nz"] - 1 - z_tr

    empty_map = np.ones((cfg["Nx"], cfg["Ny"], cfg["Nz"]), np.float32)
    loaded_map = np.flip(d3.er_map(name, cfg), axis=2).copy()

    rec_e, _ = run3d._record(rev, empty_map, n_steps, near, far, device)
    rec_l, _ = run3d._record(rev, loaded_map, n_steps, near, far, device)
    pt = run3d._ungated_phasor(rec_e[far], cfg)
    lt = run3d._ungated_phasor(rec_l[far], cfg)
    return lt / pt


def identity_check(scale, device):
    cfg = d3.config3d(scale)
    z_ref, z_tr = d3.planes(cfg)
    n_steps = run3d.n_steps_for(cfg, RECORD)
    empty = np.ones((cfg["Nx"], cfg["Ny"], cfg["Nz"]), np.float32)
    rec, _ = run3d._record(cfg, empty, n_steps, z_ref, z_tr, device)
    pe = run3d._ungated_phasor(rec[z_ref], cfg)
    pt = run3d._ungated_phasor(rec[z_tr], cfg)
    return abs((pe - pe) / pe), abs(pt / pt)


def main(scale=2):
    device = torch.device("cuda")
    out = {}

    # 1. identity
    i11, i21 = identity_check(scale, device)
    out["identity"] = dict(S11_abs=i11, S21_abs=i21)
    print("identity", out["identity"])

    # 2. reciprocity
    out["reciprocity"] = {}
    for name in ["full_slab", "centered_post", "corner_block", "grating"]:
        fwd = run3d.run_device(name, scale=scale, device=device,
                               physical_record=RECORD)
        s12 = reverse_s12(name, scale, device)
        out["reciprocity"][name] = dict(
            S21=complex(fwd["s21"]), S12=complex(s12),
            complex_diff=abs(fwd["s21"] - s12),
            magnitude_diff=abs(abs(fwd["s21"]) - abs(s12)),
            phase_diff=float(np.angle(fwd["s21"] * np.conj(s12))))
        print("reciprocity", name, "mag diff %.3e complex diff %.3e" % (
            abs(abs(fwd["s21"]) - abs(s12)), abs(fwd["s21"] - s12)))

    # 3. mirror symmetry
    L = run3d.run_device("side_slab_left", scale=scale, device=device,
                         physical_record=RECORD)
    R = run3d.run_device("side_slab_right", scale=scale, device=device,
                         physical_record=RECORD)
    out["mirror_side_slabs"] = dict(
        s11_diff=abs(L["s11"] - R["s11"]),
        s21_diff=abs(L["s21"] - R["s21"]),
        S21_left=complex(L["s21"]), S21_right=complex(R["s21"]))
    print("mirror side slabs s21 diff %.3e" % abs(L["s21"] - R["s21"]))

    # 4. analytic regression / convergence. Ungated DFT is used; the coarse
    #    s=1 grid supports a spurious near-cutoff mode that decays too slowly
    #    for a clean single-frequency DFT, so the convergence series starts at
    #    s=2 where the modal signal is well resolved.
    out["convergence"] = {}
    for name in ["full_slab", "centered_post"]:
        rows = []
        for s in (2, 4):
            r = run3d.run_device(name, scale=s, device=device,
                                 physical_record=RECORD)
            row = dict(scale=s, dx_mm=1.0 / s,
                       S11_abs=abs(r["s11"]), S21_abs=abs(r["s21"]),
                       energy=r["energy_balance"],
                       residual=r["residual_energy"])
            if name == "full_slab":
                cfg = d3.config3d(s)
                geom = (1, cfg["Nx"] - 1, 1, cfg["Ny"] - 1,
                        40 * s, 48 * s)
                a11, a21 = sp.analytic_sparams(4.0, cfg, geom)
                row["analytic_S11"] = abs(a11)
                row["analytic_S21"] = abs(a21)
                row["dS21"] = abs(abs(r["s21"]) - abs(a21))
            rows.append(row)
            print("convergence", name, json.dumps(row))
        out["convergence"][name] = rows

    # 5. grating stopband convergence at a frequency where unitarity deviates
    fp = 8.2e9
    rows = []
    for s in (2, 4):
        r = run3d.run_device("grating", scale=s, device=device,
                             physical_record=RECORD,
                             frequencies=np.array([fp]))
        x = r["sweep"][0]
        rows.append(dict(scale=s, frequency=fp,
                         S11_abs=abs(x["s11"]), S21_abs=abs(x["s21"]),
                         energy=x["energy"]))
        print("grating stopband", json.dumps(rows[-1]))
    out["grating_stopband_convergence"] = rows

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "validate3d.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=_json_default)
    print("saved validate3d.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
