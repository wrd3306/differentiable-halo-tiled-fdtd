"""Streamlined P0-4 binary design validation: optimizer comparison + verification.

Runs the same binary pixelated AR problem with three optimisers (sigmoid-Adam,
GA, PSO) at bounded cost, selects the best, then does a non-autograd 10-ns
verification and erosion/dilation fabrication sensitivity. All evaluations are
real full-wave solves. No surrogate, fitted, or interpolated model.
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, HERE)
import pixel_design as px  # noqa: E402


def clean(x):
    return [complex(v) for v in np.asarray(x).ravel()]


def main():
    t0 = time.time()
    device = torch.device("cuda")
    cfg, n_steps = px.pixel_config(record_ns=6.0)
    freqs_hz = np.array(px.FREQS_GHZ) * 1e9
    base, cell_masks, _ = px.build_geometry(cfg, device)
    nc = cell_masks.shape[0]

    with torch.no_grad():
        _, empty = px.forward_broadband(
            cfg, device, torch.ones_like(base), n_steps, freqs_hz,
            use_ckpt=False)

    bl, bs11, _ = px.eval_bits(
        torch.zeros(nc, device=device), base, cell_masks, cfg, device,
        n_steps, freqs_hz, empty)
    print("baseline mean|S11|^2 %.4f max|S11| %.4f" % (
        bl, np.max(np.abs(bs11))), flush=True)

    # Adam (two starts)
    adam = []
    for name, theta0 in (
            ("uniform", torch.full((nc,), 0.5, device=device)),
            ("low", torch.full((nc,), 0.2, device=device))):
        r = px.adam_design(theta0, base, cell_masks, cfg, device, n_steps,
                           freqs_hz, empty, iters=25, lr=0.15)
        r["name"] = name
        adam.append(r)
        print("Adam %s max|S11| %.4f" % (
            name, np.max(np.abs(r["binary_S11"]))), flush=True)

    ga = px.genetic_algorithm(base, cell_masks, cfg, device, n_steps,
                              freqs_hz, empty, pop_size=8, generations=6,
                              seed=0)
    print("GA max|S11| %.4f evals %d" % (
        np.max(np.abs(ga["S11"])), ga["n_evals"]), flush=True)

    pso = px.particle_swarm(base, cell_masks, cfg, device, n_steps,
                            freqs_hz, empty, n_particles=8, iterations=6,
                            seed=0)
    print("PSO max|S11| %.4f evals %d" % (
        np.max(np.abs(pso["S11"])), pso["n_evals"]), flush=True)

    cands = []
    for r in adam:
        cands.append(("adam:" + r["name"], r["bits"],
                      float(np.mean(np.abs(r["binary_S11"]) ** 2))))
    cands.append(("ga", ga["bits"], ga["loss"]))
    cands.append(("pso", pso["bits"], pso["loss"]))
    best_method, best_bits, best_loss = min(cands, key=lambda x: x[2])
    print("best method %s loss %.4f" % (best_method, best_loss), flush=True)

    verify = px.independent_verify(best_bits, base, cell_masks,
                                   record_ns=10.0, device=device)
    print("independent max|S11| %.4f resid %.3e" % (
        verify["max_S11"], verify["residual_energy"]), flush=True)

    fab = px.fabrication_perturb(best_bits, base, cell_masks, cfg, device,
                                 n_steps, freqs_hz, empty)
    print("fabrication erosion max|S11| %.4f dilation %.4f" % (
        fab["erosion"]["max_S11"], fab["dilation"]["max_S11"]), flush=True)

    out = dict(
        config=dict(n_cells=int(nc), cell_grid=list(px.cell_grid()),
                    cell_mm=px.CELL_MM, freqs_ghz=list(px.FREQS_GHZ),
                    er_dielectric=px.ER_DIELECTRIC, record_ns=6.0),
        baseline=dict(loss=bl, max_S11=float(np.max(np.abs(bs11))),
                      S11=clean(bs11)),
        adam=[dict(name=r["name"], bits=r["bits"].tolist(),
                   max_S11=float(np.max(np.abs(r["binary_S11"]))),
                   S11=clean(r["binary_S11"])) for r in adam],
        ga=dict(bits=ga["bits"].tolist(), n_evals=ga["n_evals"],
                max_S11=float(np.max(np.abs(ga["S11"]))), S11=clean(ga["S11"])),
        pso=dict(bits=pso["bits"].tolist(), n_evals=pso["n_evals"],
                 max_S11=float(np.max(np.abs(pso["S11"]))),
                 S11=clean(pso["S11"])),
        best=dict(method=best_method, bits=best_bits.tolist(), loss=best_loss),
        independent=[dict(frequency=o["frequency"], s11=complex(o["s11"]),
                          s21=complex(o["s21"]), energy=o["energy"])
                     for o in verify["sweep"]],
        independent_max=verify["max_S11"],
        fabrication=dict(
            erosion=dict(max_S11=fab["erosion"]["max_S11"],
                         S11=clean(fab["erosion"]["S11"])),
            dilation=dict(max_S11=fab["dilation"]["max_S11"],
                          S11=clean(fab["dilation"]["S11"]))),
        elapsed_s=time.time() - t0)

    def jsonable(o):
        if isinstance(o, dict):
            return {k: jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [jsonable(v) for v in o]
        if isinstance(o, complex):
            return dict(real=o.real, imag=o.imag)
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    path = os.path.join(ROOT, "results", "pixel_trim.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(jsonable(out), fh, indent=2)
    print("saved", path, "elapsed %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
