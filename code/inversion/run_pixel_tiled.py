"""Binary pixel design driven end-to-end by the differentiable halo-tiled engine.

The gradient-based sigmoid-Adam optimisation is re-run with every forward step
and reverse pass routed through HaloTiledStep (shared-memory CUDA forward +
discrete adjoint), so the reported design genuinely runs on the halo-tiled
kernel. The final pattern is confirmed with the non-autograd engine (no inverse
crime). GA/PSO are gradient-free forward-only evaluations and are engine
independent, so their earlier numbers remain valid. Wall time and per-iteration
cost are recorded.
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
sys.path.insert(0, os.path.join(ROOT, "code", "engine"))
import pixel_design as px  # noqa: E402
import fdtd_diff_tiled as fdt  # noqa: E402


def tiled_sparams(cfg, device, er_map, n_steps, freqs_hz, empty, chunk=20,
                  use_ckpt=True):
    _, acc = fdt.forward_broadband_tiled(
        cfg, device, er_map, n_steps, freqs_hz, chunk=chunk, use_ckpt=use_ckpt)
    pe, pt = empty[0], empty[1]
    return (acc[0] - pe) / pe, acc[1] / pt


def tiled_objective(theta, base, cell_masks, beta, cfg, device, n_steps,
                    freqs_hz, empty, chunk=20, p_norm=8.0):
    er, p = px.er_map_from(theta, base, cell_masks, beta)
    s11, s21 = tiled_sparams(cfg, device, er, n_steps, freqs_hz, empty, chunk)
    r = torch.abs(s11)
    return torch.mean(r ** p_norm) ** (1.0 / p_norm), s11, s21, p


def tiled_adam(theta0, base, cell_masks, cfg, device, n_steps, freqs_hz, empty,
               iters=25, lr=0.15, chunk=20):
    beta_schedule = [(0, 2.0), (iters // 3, 5.0), (2 * iters // 3, 12.0)]
    theta = theta0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    beta = beta_schedule[0][1]
    hist = []
    for it in range(iters):
        for bstart, bval in beta_schedule:
            if it == bstart:
                beta = bval
        t0 = time.perf_counter()
        opt.zero_grad()
        loss, s11, s21, p = tiled_objective(
            theta, base, cell_masks, beta, cfg, device, n_steps, freqs_hz,
            empty, chunk)
        loss.backward()
        opt.step()
        with torch.no_grad():
            theta.clamp_(0.0, 1.0)
        it_s = time.perf_counter() - t0
        hist.append(dict(it=it, beta=beta, loss=loss.item(),
                         max_S11=s11.abs().max().item(),
                         mean_S11=torch.mean(s11.abs()).item(),
                         iter_s=it_s))
        print("it %2d beta %4.1f max|S11| %.4f mean %.4f (%.1fs)" % (
            it, beta, hist[-1]["max_S11"], hist[-1]["mean_S11"], it_s),
            flush=True)
    bits = px.hard_binary(theta)
    with torch.no_grad():
        er_b = px.binary_er(bits, base, cell_masks)
        s11, s21 = tiled_sparams(cfg, device, er_b, n_steps, freqs_hz, empty,
                                 chunk, use_ckpt=False)
    return dict(theta=theta.detach().cpu().numpy(), bits=bits.cpu().numpy(),
                history=hist, binary_S11=s11.detach().cpu().numpy(),
                binary_S21=s21.detach().cpu().numpy())


def main():
    t_start = time.time()
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
    print("baseline max|S11| %.4f" % np.max(np.abs(bs11)), flush=True)

    adam = []
    for name, theta0 in (("uniform", torch.full((nc,), 0.5, device=device)),
                         ("low", torch.full((nc,), 0.2, device=device))):
        r = tiled_adam(theta0, base, cell_masks, cfg, device, n_steps,
                       freqs_hz, empty, iters=25, lr=0.15)
        r["name"] = name
        adam.append(r)
        print("Adam %s max|S11| %.4f" % (
            name, np.max(np.abs(r["binary_S11"]))), flush=True)

    best = min(adam, key=lambda r: float(np.mean(np.abs(r["binary_S11"]) ** 2)))
    verify = px.independent_verify(
        torch.tensor(best["bits"]), base, cell_masks, record_ns=10.0,
        device=device)
    print("independent max|S11| %.4f resid %.3e" % (
        verify["max_S11"], verify["residual_energy"]), flush=True)

    out = dict(
        engine="halo_tiled",
        baseline_max=float(np.max(np.abs(bs11))),
        adam=[dict(name=r["name"], bits=r["bits"].tolist(),
                   max_S11=float(np.max(np.abs(r["binary_S11"]))),
                   mean_S11=float(np.mean(np.abs(r["binary_S11"]))),
                   history=r["history"]) for r in adam],
        best=dict(name=best["name"], bits=best["bits"].tolist(),
                  max_S11=float(np.max(np.abs(best["binary_S11"]))),
                  mean_S11=float(np.mean(np.abs(best["binary_S11"])))),
        independent_max=verify["max_S11"],
        independent=[dict(frequency=o["frequency"], s11=complex(o["s11"]),
                          s21=complex(o["s21"]), energy=o["energy"])
                     for o in verify["sweep"]],
        total_s=time.time() - t_start)

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

    path = os.path.join(ROOT, "results", "pixel_tiled.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(jsonable(out), fh, indent=2)
    print("saved", path, "total %.0f s" % (time.time() - t_start), flush=True)


if __name__ == "__main__":
    main()
