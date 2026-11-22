"""Full P0-4 pipeline driver: broadband pixelated AR design and validation."""
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


def gradient_check(base, cell_masks, cfg, device, n_steps, freqs_hz, empty,
                   h=0.02, beta=6.0):
    nc = cell_masks.shape[0]
    theta = torch.full((nc,), 0.5, device=device, requires_grad=True)
    loss, _, _, _ = px.objective(theta, base, cell_masks, beta, cfg, device,
                                 n_steps, freqs_hz, empty)
    loss.backward()
    auto = theta.grad.detach().cpu().numpy()

    def f(tvec):
        with torch.no_grad():
            l, _, _, _ = px.objective(
                torch.tensor(tvec, device=device), base, cell_masks, beta,
                cfg, device, n_steps, freqs_hz, empty, use_ckpt=False)
        return l.item()

    fd = np.zeros(nc)
    base_vec = np.full(nc, 0.5, dtype=np.float32)
    for c in range(nc):
        up = base_vec.copy(); up[c] += h
        dn = base_vec.copy(); dn[c] -= h
        fd[c] = (f(up) - f(dn)) / (2 * h)
    rel = np.abs(auto - fd) / (np.abs(fd) + 1e-12)
    return dict(autograd=auto.tolist(), finite_diff=fd.tolist(),
                rel_err=rel.tolist(), max_rel=float(rel.max()))


def main():
    t_start = time.time()
    device = torch.device("cuda")
    cfg, n_steps = px.pixel_config(record_ns=6.0)
    freqs_hz = np.array(px.FREQS_GHZ) * 1e9
    base, cell_masks, cells = px.build_geometry(cfg, device)
    nc = cell_masks.shape[0]

    with torch.no_grad():
        _, acc_e = px.forward_broadband(
            cfg, device, torch.ones_like(base), n_steps, freqs_hz,
            use_ckpt=False)
    empty = acc_e

    # baseline obstacle (no design cells)
    bl, bs11, bs21 = px.eval_bits(
        torch.zeros(nc, device=device), base, cell_masks, cfg, device,
        n_steps, freqs_hz, empty)
    print("baseline loss %.4f |S11| %s" % (
        bl, np.round(np.abs(bs11), 4)), flush=True)

    print("gradient check...", flush=True)
    gcheck = gradient_check(base, cell_masks, cfg, device, n_steps,
                            freqs_hz, empty)
    print("  max rel err %.3e" % gcheck["max_rel"], flush=True)

    # Adam multi-start
    adam_starts = [
        ("uniform", torch.full((nc,), 0.5, device=device)),
        ("random", torch.rand(nc, device=device)),
        ("low", torch.full((nc,), 0.2, device=device)),
    ]
    adam_runs = []
    for name, t0 in adam_starts:
        r = px.adam_design(t0, base, cell_masks, cfg, device, n_steps,
                           freqs_hz, empty, iters=25, lr=0.15)
        r["name"] = name
        adam_runs.append(r)
        print("Adam", name, "binary max|S11| %.4f" %
              np.max(np.abs(r["binary_S11"])), flush=True)

    ga = px.genetic_algorithm(base, cell_masks, cfg, device, n_steps,
                              freqs_hz, empty, pop_size=10, generations=8,
                              seed=0)
    print("GA max|S11| %.4f evals %d" % (
        np.max(np.abs(ga["S11"])), ga["n_evals"]), flush=True)

    pso = px.particle_swarm(base, cell_masks, cfg, device, n_steps,
                            freqs_hz, empty, n_particles=10, iterations=8,
                            seed=0)
    print("PSO max|S11| %.4f evals %d" % (
        np.max(np.abs(pso["S11"])), pso["n_evals"]), flush=True)

    # select the best binary design by the 6-ns objective across all methods
    candidates = []
    for r in adam_runs:
        candidates.append(("adam:" + r["name"], r["bits"],
                           float(np.mean(np.abs(r["binary_S11"]) ** 2))))
    candidates.append(("ga", ga["bits"], ga["loss"]))
    candidates.append(("pso", pso["bits"], pso["loss"]))
    best_method, best_bits, best_loss = min(candidates, key=lambda x: x[2])
    print("best method:", best_method, "loss %.4f" % best_loss, flush=True)

    # independent non-autograd verification at 10 ns
    verify = px.independent_verify(best_bits, base, cell_masks,
                                   record_ns=10.0, device=device)
    print("independent max|S11| %.4f resid %.3e" % (
        verify["max_S11"], verify["residual_energy"]), flush=True)

    # fabrication sensitivity (erosion/dilation by one voxel)
    fab = px.fabrication_perturb(best_bits, base, cell_masks, cfg, device,
                                 n_steps, freqs_hz, empty)
    print("fabrication max|S11| erosion %.4f dilation %.4f" % (
        fab["erosion"]["max_S11"], fab["dilation"]["max_S11"]), flush=True)

    out = dict(
        config=dict(n_cells=int(nc), cell_grid=list(px.cell_grid()),
                    cell_mm=px.CELL_MM, freqs_ghz=list(px.FREQS_GHZ),
                    er_dielectric=px.ER_DIELECTRIC, record_ns=6.0),
        baseline=dict(loss=bl, S11=_clean(bs11), S21=_clean(bs21)),
        gradient_check=_serializable(gcheck),
        adam_runs=[dict(name=r["name"], bits=r["bits"].tolist(),
                        history=r["history"],
                        binary_S11=_clean(r["binary_S11"]),
                        binary_S21=_clean(r["binary_S21"])) for r in adam_runs],
        ga=dict(bits=ga["bits"].tolist(), loss=ga["loss"],
                history=ga["history"], n_evals=ga["n_evals"],
                S11=_clean(ga["S11"]), S21=_clean(ga["S21"])),
        pso=dict(bits=pso["bits"].tolist(), loss=pso["loss"],
                 history=pso["history"], n_evals=pso["n_evals"],
                 S11=_clean(pso["S11"]), S21=_clean(pso["S21"])),
        best=dict(method=best_method, bits=best_bits.tolist(),
                  loss=best_loss),
        independent=[dict(frequency=o["frequency"],
                          s11=complex(o["s11"]), s21=complex(o["s21"]),
                          energy=o["energy"]) for o in verify["sweep"]],
        fabrication=dict(
            erosion=dict(S11=_clean(fab["erosion"]["S11"]),
                         max_S11=fab["erosion"]["max_S11"]),
            dilation=dict(S11=_clean(fab["dilation"]["S11"]),
                          max_S11=fab["dilation"]["max_S11"])),
        elapsed_s=time.time() - t_start,
    )
    path = os.path.join(ROOT, "results", "pixel_design.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(out), fh, indent=2)
    print("saved", path, "elapsed %.0f s" % (time.time() - t_start),
          flush=True)


def _clean(x):
    return [complex(v) for v in np.asarray(x).ravel()]


def _serializable(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (list, tuple)):
            out[k] = v
        else:
            out[k] = v
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, complex):
        return dict(real=o.real, imag=o.imag)
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


if __name__ == "__main__":
    main()
