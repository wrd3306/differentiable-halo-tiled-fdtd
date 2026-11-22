"""Multi-parameter differentiable 3-D anti-reflection design.

The ar_match device has a fixed high-index obstacle (er = 6) that reflects
strongly, flanked by two matching posts whose permittivities (er1, er2) are the
two design variables. The objective is the reflected power |S11|^2. Because the
bare obstacle sets a large baseline reflection and over-dielectric posts
reflect strongly themselves, the matching optimum is an interior point reached
by quarter-wave interference, not a boundary solution.

The design is validated three ways: autograd gradients against central finite
differences; the Adam optimum against a deterministic grid search and multiple
optimizer starts; and the final design against the non-autograd map engine over
a full 10 ns record (no inverse crime). Design parameters are projected onto a
physical interval to keep the lossless update stable.
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
import fdtd_diff3d as f3  # noqa: E402
import run3d  # noqa: E402

DEVICE = "ar_match"
# Matching posts must remain real dielectric structures (er >= 1.5); er = 1
# would mean no post and is the trivial "remove the design" escape.
ER_MIN, ER_MAX = 1.5, 6.0


def _design(cfg, device):
    """Base map (fixed obstacle, posts at er=1) and the two post masks."""
    boxes = d3.DEVICES[DEVICE]
    base = d3.er_map(DEVICE, cfg)
    m1 = np.zeros_like(base)
    m2 = np.zeros_like(base)
    for bi, out in ((0, m1), (1, m2)):
        x0, x1, y0, y1, z0, z1, _ = boxes[bi]
        i0 = max(1, int(round(x0 * 1e-3 / cfg["dx"])))
        i1 = min(cfg["Nx"] - 1, int(round(x1 * 1e-3 / cfg["dx"])))
        j0 = max(1, int(round(y0 * 1e-3 / cfg["dx"])))
        j1 = min(cfg["Ny"] - 1, int(round(y1 * 1e-3 / cfg["dx"])))
        k0 = int(round(z0 * 1e-3 / cfg["dx"]))
        k1 = int(round(z1 * 1e-3 / cfg["dx"]))
        out[i0:i1, j0:j1, k0:k1] = 1.0
    return (torch.tensor(base, device=device),
            torch.tensor(m1, device=device),
            torch.tensor(m2, device=device))


def er_map_from(er1, er2, base, m1, m2):
    return base + (er1 - 1.0) * m1 + (er2 - 1.0) * m2


def _full_gates(cfg, n_steps, device):
    """Full-record (ungated) DFT: flat window over the whole record."""
    z_ref, z_tr = d3.planes(cfg)
    win = torch.ones(n_steps, device=device)
    nn = torch.arange(n_steps, device=device)
    ker = torch.exp(1j * 2 * np.pi * cfg["f0"] * nn * cfg["dt"])
    return {z: (0, n_steps - 1, win, ker) for z in (z_ref, z_tr)}


def _empty_phasors(cfg, n_steps, gates, device, chunk=20):
    er_one = torch.ones((cfg["Nx"], cfg["Ny"], cfg["Nz"]), device=device)
    with torch.no_grad():
        _, _, acc = f3.forward_map(cfg, device, er_one, n_steps, chunk=chunk,
                                   gates=gates)
    return acc


def loaded_sparams(er1, er2, cfg, design, empty, gates, device, chunk=20):
    n_steps = gates[d3.planes(cfg)[0]][1] + 1
    base, m1, m2 = design
    erm = er_map_from(er1, er2, base, m1, m2)
    _, _, acc = f3.forward_map(cfg, device, erm, n_steps, chunk=chunk,
                               gates=gates)
    z_ref, z_tr = d3.planes(cfg)
    pe, pt = empty[z_ref], empty[z_tr]
    return (acc[z_ref] - pe) / pe, acc[z_tr] / pt


def loss_fn(er1, er2, *args, **kwargs):
    s11, s21 = loaded_sparams(er1, er2, *args, **kwargs)
    return abs(s11) ** 2, s11, s21


def _project(er):
    with torch.no_grad():
        er.clamp_(ER_MIN, ER_MAX)


def finite_difference(cfg, n_steps, design, empty, gates, device,
                      base=(3.0, 3.0), h=0.05, chunk=20):
    er1 = torch.tensor(base[0], device=device, requires_grad=True)
    er2 = torch.tensor(base[1], device=device, requires_grad=True)
    L, _, _ = loss_fn(er1, er2, cfg, design, empty, gates, device, chunk)
    L.backward()
    auto = (er1.grad.item(), er2.grad.item())

    def f(a, b):
        with torch.no_grad():
            l, _, _ = loss_fn(torch.tensor(a, device=device),
                              torch.tensor(b, device=device), cfg, design,
                              empty, gates, device, chunk)
        return l.item()

    fd1 = (f(base[0] + h, base[1]) - f(base[0] - h, base[1])) / (2 * h)
    fd2 = (f(base[0], base[1] + h) - f(base[0], base[1] - h)) / (2 * h)
    return dict(autograd=auto, finite_diff=(fd1, fd2),
                rel_err=(abs(auto[0] - fd1) / (abs(fd1) + 1e-12),
                         abs(auto[1] - fd2) / (abs(fd2) + 1e-12)))


def optimize(er1_0, er2_0, cfg, design, empty, gates, device,
             iters=25, lr=0.12, chunk=20):
    er1 = torch.tensor(er1_0, device=device, requires_grad=True)
    er2 = torch.tensor(er2_0, device=device, requires_grad=True)
    opt = torch.optim.Adam([er1, er2], lr=lr)
    hist = []
    for it in range(iters):
        opt.zero_grad()
        L, s11, s21 = loss_fn(er1, er2, cfg, design, empty, gates, device,
                              chunk)
        L.backward()
        opt.step()
        _project(er1)
        _project(er2)
        hist.append(dict(it=it, er1=er1.item(), er2=er2.item(),
                         loss=L.item(), S11_abs=abs(s11).item(),
                         S21_abs=abs(s21).item()))
    return er1.item(), er2.item(), hist


def grid_search(cfg, design, empty, gates, device,
                grid=(1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5), chunk=20):
    best, table = None, []
    for a in grid:
        for b in grid:
            with torch.no_grad():
                l, s11, s21 = loss_fn(torch.tensor(a, device=device),
                                      torch.tensor(b, device=device), cfg,
                                      design, empty, gates, device, chunk)
            table.append(dict(er1=a, er2=b, loss=l.item(),
                              S11_abs=abs(s11).item(),
                              S21_abs=abs(s21).item()))
            if best is None or l.item() < best["loss"]:
                best = dict(er1=a, er2=b, loss=l.item(),
                            S11_abs=abs(s11).item(),
                            S21_abs=abs(s21).item())
    return best, table


def main(record_ns=6.0, chunk=20):
    device = torch.device("cuda")
    cfg = d3.config3d(2)
    n_steps = int(np.ceil(record_ns * 1e-9 / cfg["dt"]))
    design = _design(cfg, device)
    gates = _full_gates(cfg, n_steps, device)
    empty = _empty_phasors(cfg, n_steps, gates, device, chunk)

    # 0. bare-obstacle baseline (posts truly absent, er = 1)
    with torch.no_grad():
        bl, bs11, bs21 = loss_fn(torch.tensor(1.0, device=device),
                                 torch.tensor(1.0, device=device), cfg,
                                 design, empty, gates, device, chunk)
    print("bare obstacle |S11| %.4f |S21| %.4f" % (abs(bs11), abs(bs21)))

    # 1. gradient verification
    grad = finite_difference(cfg, n_steps, design, empty, gates, device,
                             chunk=chunk)
    print("gradient check", json.dumps(grad))

    # 2. optimizer from multiple starts
    starts = [(2.0, 4.0), (4.0, 2.0), (3.0, 3.0)]
    runs = []
    for st in starts:
        a, b, hist = optimize(st[0], st[1], cfg, design, empty, gates, device,
                              chunk=chunk)
        runs.append(dict(start=st, er1=a, er2=b,
                         final_loss=hist[-1]["loss"], history=hist))
        print("optimized from", st, "-> er1 %.4f er2 %.4f |S11| %.4f" % (
            a, b, hist[-1]["S11_abs"]))

    # 3. grid search baseline
    best, table = grid_search(cfg, design, empty, gates, device, chunk=chunk)
    print("grid best", json.dumps(best))

    # 4. independent verification of the best optimizer design with the
    # non-autograd engine over a full 10 ns record
    sol = min(runs, key=lambda r: r["final_loss"])
    override = {0: sol["er1"], 1: sol["er2"]}
    ver = run3d.run_device(DEVICE, scale=2, device=device,
                           er_override=override, physical_record=10e-9)
    out = dict(bare_obstacle=dict(S11_abs=abs(bs11).item(),
                                  S21_abs=abs(bs21).item()),
               gradient_check=grad, optimizer_runs=runs, grid_best=best,
               grid_table=table,
               independent_verification=dict(
                   er1=sol["er1"], er2=sol["er2"],
                   S11_abs=abs(ver["s11"]), S21_abs=abs(ver["s21"]),
                   energy=ver["energy_balance"],
                   residual=ver["residual_energy"]))
    print("independent verification", json.dumps(
        out["independent_verification"]))

    with open(os.path.join(ROOT, "results", "optimize3d.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("saved optimize3d.json")


if __name__ == "__main__":
    main()
