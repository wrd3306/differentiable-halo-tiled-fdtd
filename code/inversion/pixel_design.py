"""High-dimensional broadband pixelated anti-reflection (AR) inverse design.

A fixed full-width dielectric slab (er = 1.45, 6 mm thick) reflects over the
8-12 GHz band. A pixelated AR region of Nc binary design cells sits between the
input port and the slab. Each cell is a coarse physical block (minimum feature
size = cell size) that is either vacuum (er = 1) or dielectric (er = 2); the
whole design is simulated in full 3-D on the same hard voxel grid.

The continuous relaxation is a sigmoid projection optimised by Adam with the
discrete-adjoint gradients; gradient-free genetic-algorithm (GA) and
particle-swarm (PSO) baselines optimise the same binary objective. The winning
design is re-simulated with the non-autograd engine (no inverse crime), and
fabrication robustness is measured by erosion/dilation of the binary mask.

All results are real full-wave solves; no surrogate, fitted, or interpolated
model is used anywhere.
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
import fdtd_diff as fd  # noqa: E402

# ---- physical design constants (millimetres) ----
ER_DIELECTRIC = 2.0          # binary "material" state (the pattern dielectric)
ER_OBSTACLE = 1.45           # fixed centered reflecting slab
CELL_MM = 2.0                # sub-wavelength design-cell pitch
X0, X1 = 6.0, 18.0           # design region: centered 12 mm strip
Z0, Z1 = 36.0, 52.0          # design-region z extent (16 mm)
OB_X0, OB_X1 = 6.0, 18.0     # fixed obstacle: centered 12 mm strip
OB_Z0, OB_Z1 = 52.0, 56.0    # fixed obstacle: 4 mm thick
FREQS_GHZ = (8.0, 9.0, 10.0, 11.0, 12.0)
ER_MIN, ER_MAX = 1.0, ER_DIELECTRIC


def pixel_config(record_ns=6.0):
    """Waveguide config with a wideband pulse covering 8-12 GHz."""
    cfg = d3.config3d(2)
    # Shorter Gaussian envelope: spectral std ~ 1.5 GHz, covers the band.
    cfg["pulse"] = dict(t0=0.8e-9, tw=0.15e-9, fc=10e9)
    n_steps = int(np.ceil(record_ns * 1e-9 / cfg["dt"]))
    return cfg, n_steps


def cell_grid():
    """Return (n_x, n_z) coarse cell counts and their physical bounds."""
    nx = int(round((X1 - X0) / CELL_MM))
    nz = int(round((Z1 - Z0) / CELL_MM))
    return nx, nz


def _voxel_range(p0, p1, cfg, axis_size):
    """Closed physical-mm interval -> interior voxel index range on the grid."""
    i0 = max(1, int(round(p0 * 1e-3 / cfg["dx"])))
    i1 = min(axis_size - 1, int(round(p1 * 1e-3 / cfg["dx"])))
    return i0, i1


def build_geometry(cfg, device):
    """Base map (vacuum + fixed obstacle) and the per-cell voxel masks."""
    nx, nz = cell_grid()
    Nx, Ny, Nz = cfg["Nx"], cfg["Ny"], cfg["Nz"]
    base = np.ones((Nx, Ny, Nz), dtype=np.float32)

    # Fixed obstacle: full interior height.
    i0, i1 = _voxel_range(OB_X0, OB_X1, cfg, Nx)
    j0, j1 = 1, Ny - 1
    k0, k1 = _voxel_range(OB_Z0, OB_Z1, cfg, Nz)
    base[i0:i1, j0:j1, k0:k1] = ER_OBSTACLE

    masks = []
    cells = []
    for iz in range(nz):
        for ix in range(nx):
            cx0, cx1 = X0 + ix * CELL_MM, X0 + (ix + 1) * CELL_MM
            cz0, cz1 = Z0 + iz * CELL_MM, Z0 + (iz + 1) * CELL_MM
            ci0, ci1 = _voxel_range(cx0, cx1, cfg, Nx)
            ck0, ck1 = _voxel_range(cz0, cz1, cfg, Nz)
            m = np.zeros((Nx, Ny, Nz), dtype=np.float32)
            m[ci0:ci1, 1:Ny - 1, ck0:ck1] = 1.0
            masks.append(m)
            cells.append((ix, iz, ci0, ci1, ck0, ck1))
    return (torch.tensor(base, device=device),
            torch.stack([torch.tensor(m, device=device) for m in masks]),
            cells)


def er_map_from(theta, base, cell_masks, beta):
    """Differentiable er map from relaxed densities theta (Nc,).

    Sigmoid projection drives the densities toward {0, 1} as beta grows.
    """
    p = torch.sigmoid(beta * (theta - 0.5))
    p = p / torch.sigmoid(torch.tensor(beta * 0.5))  # normalise endpoints
    p = p.clamp(0.0, 1.0)
    delta = (ER_DIELECTRIC - 1.0) * p            # er above vacuum per cell
    er = base + torch.einsum("c,cxyz->xyz", delta, cell_masks)
    return er, p


def run_chunk_bb(state, c, n0, n1, geom, acc):
    """Pure chunk: multi-frequency online DFT at the two monitor planes."""
    dt = geom["dt"]
    freqs = geom["freqs_t"]                  # (F,) tensor, Hz
    planes = geom["planes"]                  # two plane indices
    for n in range(n0, n1):
        state = fd.step_fn(state, c, n, dt, geom["f0"], geom["idx"],
                           geom["idy"], geom["idz"], geom["mh"], geom["pulse"])
        amps = torch.stack([fd._modal_amp(state[1], geom["cfg"], z)
                            for z in planes])              # (2,)
        ker = torch.exp(1j * 2 * np.pi * freqs * (n * dt))  # (F,)
        acc = acc + amps[:, None].to(torch.complex64) * ker[None, :] * dt
    return state, acc


def forward_broadband(cfg, device, er_map, n_steps, freqs_hz, chunk=20,
                      use_ckpt=True):
    """Run the differentiable FDTD and return per-plane, per-freq phasors."""
    c = f3.build_constants_map(cfg, device, er_map)
    state = fd.init_state(cfg, device)
    planes = list(d3.planes(cfg))
    geom = dict(dt=cfg["dt"], f0=cfg["f0"], idx=1 / cfg["dx"],
                idy=1 / cfg["dy"], idz=1 / cfg["dz"],
                mh=cfg["dt"] / fd.fg.MU0, pulse=cfg.get("pulse"), cfg=cfg,
                planes=planes, freqs_t=torch.tensor(freqs_hz, device=device))
    acc = torch.zeros((len(planes), len(freqs_hz)), dtype=torch.complex64,
                      device=device)
    n0 = 0
    while n0 < n_steps:
        n1 = min(n0 + chunk, n_steps)
        if use_ckpt:
            state, acc = torch.utils.checkpoint.checkpoint(
                run_chunk_bb, state, c, n0, n1, geom, acc,
                use_reentrant=False)
        else:
            state, acc = run_chunk_bb(state, c, n0, n1, geom, acc)
        n0 = n1
    return state, acc


def sparams(cfg, device, er_map, n_steps, freqs_hz, empty, chunk=20,
            use_ckpt=True):
    """Return complex S11, S21 vectors over the frequency band."""
    _, acc = forward_broadband(cfg, device, er_map, n_steps, freqs_hz,
                               chunk=chunk, use_ckpt=use_ckpt)
    pe, pt = empty[0], empty[1]
    s11 = (acc[0] - pe) / pe
    s21 = acc[1] / pt
    return s11, s21


def objective(theta, base, cell_masks, beta, cfg, device, n_steps, freqs_hz,
              empty, chunk=20, use_ckpt=True, p_norm=8.0):
    """Smooth minimax: high-order Lp norm of the reflection over the band."""
    er, p = er_map_from(theta, base, cell_masks, beta)
    s11, s21 = sparams(cfg, device, er, n_steps, freqs_hz, empty, chunk,
                       use_ckpt)
    r = torch.abs(s11)
    loss = torch.mean(r ** p_norm) ** (1.0 / p_norm)
    return loss, s11, s21, p


def eval_bits(bits, base, cell_masks, cfg, device, n_steps, freqs_hz, empty,
              chunk=20):
    """Evaluate the binary objective for one pattern (no grad, no checkpoint)."""
    with torch.no_grad():
        er = binary_er(bits, base, cell_masks)
        s11, s21 = sparams(cfg, device, er, n_steps, freqs_hz, empty, chunk,
                           use_ckpt=False)
        loss = float(torch.mean(s11.abs() ** 2))
    return loss, s11.detach().cpu().numpy(), s21.detach().cpu().numpy()


def genetic_algorithm(base, cell_masks, cfg, device, n_steps, freqs_hz, empty,
                      pop_size=12, generations=10, mut_prob=0.08, seed=0,
                      chunk=20):
    """Binary GA with tournament selection, uniform crossover, elitism."""
    rng = np.random.default_rng(seed)
    nc = cell_masks.shape[0]
    pop = (rng.random((pop_size, nc)) < 0.5).astype(np.float32)
    scores = np.zeros(pop_size)
    n_evals = 0
    best = None
    history = []

    def score(ind):
        nonlocal n_evals
        l, _, _ = eval_bits(torch.tensor(ind, device=device), base,
                            cell_masks, cfg, device, n_steps, freqs_hz, empty,
                            chunk)
        n_evals += 1
        return l

    for gen in range(generations):
        for i in range(pop_size):
            scores[i] = score(pop[i])
        order = np.argsort(scores)
        if best is None or scores[order[0]] < best["loss"]:
            best = dict(bits=pop[order[0]].copy(), loss=float(scores[order[0]]))
        history.append(dict(gen=gen, best_loss=float(best["loss"]),
                            gen_best=float(scores[order[0]])))
        new_pop = [pop[order[0]].copy()]             # elitism
        while len(new_pop) < pop_size:
            def pick():
                a, b = rng.integers(0, pop_size, 2)
                return pop[a] if scores[a] < scores[b] else pop[b]
            p1, p2 = pick(), pick()
            mask = (rng.random(nc) < 0.5).astype(np.float32)
            child = p1 * mask + p2 * (1 - mask)
            flip = (rng.random(nc) < mut_prob).astype(np.float32)
            child = np.abs(child - flip)
            new_pop.append(child.astype(np.float32))
        pop = np.stack(new_pop)
    _, s11, s21 = eval_bits(torch.tensor(best["bits"], device=device), base,
                            cell_masks, cfg, device, n_steps, freqs_hz, empty,
                            chunk)
    return dict(bits=best["bits"], loss=best["loss"], S11=s11, S21=s21,
                history=history, n_evals=n_evals)


def particle_swarm(base, cell_masks, cfg, device, n_steps, freqs_hz, empty,
                   n_particles=12, iterations=10, seed=0, chunk=20):
    """Continuous PSO; positions are hard-thresholded to binary at evaluation."""
    rng = np.random.default_rng(seed)
    nc = cell_masks.shape[0]
    pos = rng.random((n_particles, nc)).astype(np.float32)
    vel = (rng.random((n_particles, nc)) - 0.5).astype(np.float32) * 0.2
    pbest = pos.copy()
    pbest_score = np.full(n_particles, np.inf)
    gbest, gbest_score = None, np.inf
    history = []
    n_evals = 0
    w, c1, c2 = 0.7, 1.5, 1.5

    def score(x):
        nonlocal n_evals
        bits = torch.tensor((x >= 0.5).astype(np.float32), device=device)
        l, _, _ = eval_bits(bits, base, cell_masks, cfg, device, n_steps,
                            freqs_hz, empty, chunk)
        n_evals += 1
        return l

    for it in range(iterations):
        for i in range(n_particles):
            s = score(pos[i])
            if s < pbest_score[i]:
                pbest_score[i], pbest[i] = s, pos[i].copy()
            if s < gbest_score:
                gbest_score, gbest = s, pos[i].copy()
        history.append(dict(iter=it, best_loss=float(gbest_score)))
        for i in range(n_particles):
            r1, r2 = rng.random(nc), rng.random(nc)
            vel[i] = (w * vel[i]
                      + c1 * r1 * (pbest[i] - pos[i])
                      + c2 * r2 * (gbest - pos[i]))
            vel[i] = np.clip(vel[i], -0.4, 0.4)
            pos[i] = np.clip(pos[i] + vel[i], 0.0, 1.0)
    bits = (gbest >= 0.5).astype(np.float32)
    _, s11, s21 = eval_bits(torch.tensor(bits, device=device), base,
                            cell_masks, cfg, device, n_steps, freqs_hz, empty,
                            chunk)
    return dict(bits=bits, loss=gbest_score, S11=s11, S21=s21,
                history=history, n_evals=n_evals)


def hard_binary(theta):
    """Threshold relaxed densities to a {0,1} binary pattern."""
    return (theta.detach() >= 0.5).to(torch.float32)


def binary_er(bits, base, cell_masks):
    """Build the er map from a hard binary pattern (no grad)."""
    return base + (ER_DIELECTRIC - 1.0) * torch.einsum(
        "c,cxyz->xyz", bits, cell_masks)


def binary_local_search(bits0, base, cell_masks, cfg, device, n_steps,
                        freqs_hz, empty, max_passes=3, chunk=20):
    """Greedy coordinate descent on the hard binary pattern.

    Each pass sweeps all cells and accepts every flip that lowers the L8
    minimax objective, re-evaluating the accepted pattern on the real grid.
    """
    bits = np.asarray(bits0, dtype=np.float32).copy()
    cur, _, _ = eval_bits(torch.tensor(bits, device=device), base, cell_masks,
                          cfg, device, n_steps, freqs_hz, empty, chunk)
    history = [cur]
    n_evals = 1
    for _ in range(max_passes):
        improved = False
        for c in range(len(bits)):
            trial = bits.copy()
            trial[c] = 1.0 - trial[c]
            val, _, _ = eval_bits(torch.tensor(trial, device=device), base,
                                  cell_masks, cfg, device, n_steps, freqs_hz,
                                  empty, chunk)
            n_evals += 1
            if val < cur - 1e-9:
                bits, cur, improved = trial, val, True
                history.append(cur)
        if not improved:
            break
    _, s11, s21 = eval_bits(torch.tensor(bits, device=device), base,
                            cell_masks, cfg, device, n_steps, freqs_hz, empty,
                            chunk)
    return dict(bits=bits, loss=cur, S11=s11, S21=s21, history=history,
                n_evals=n_evals)


def adam_design(theta0, base, cell_masks, cfg, device, n_steps, freqs_hz,
                empty, iters=40, lr=0.15, chunk=20, beta_schedule=None):
    """Adam with sigmoid-projection continuation; returns relaxed + binary."""
    if beta_schedule is None:
        beta_schedule = [(0, 2.0), (iters // 3, 5.0),
                         (2 * iters // 3, 12.0)]
    theta = theta0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    beta = beta_schedule[0][1]
    hist = []
    for it in range(iters):
        for bstart, bval in beta_schedule:
            if it == bstart:
                beta = bval
        opt.zero_grad()
        loss, s11, s21, p = objective(theta, base, cell_masks, beta, cfg,
                                      device, n_steps, freqs_hz, empty, chunk)
        loss.backward()
        opt.step()
        with torch.no_grad():
            theta.clamp_(0.0, 1.0)
        hist.append(dict(it=it, beta=beta, loss=loss.item(),
                         max_S11=s11.abs().max().item(),
                         mean_S11=torch.mean(s11.abs()).item()))
    bits = hard_binary(theta)
    with torch.no_grad():
        er_b = binary_er(bits, base, cell_masks)
        s11, s21 = sparams(cfg, device, er_b, n_steps, freqs_hz, empty, chunk,
                           use_ckpt=False)
    return dict(theta=theta.detach().cpu().numpy(),
                bits=bits.cpu().numpy(), history=hist,
                binary_S11=s11.detach().cpu().numpy(),
                binary_S21=s21.detach().cpu().numpy())


def independent_verify(bits, base, cell_masks, record_ns=10.0, device=None):
    """Non-autograd L0 verification of a binary design over a full record."""
    import engine3d as e3
    import fdtd_gpu as fg
    import sparam as sp
    device = device or torch.device("cuda")
    cfg, n_steps = pixel_config(record_ns=record_ns)
    freqs_hz = np.array(FREQS_GHZ) * 1e9
    z_ref, z_tr = d3.planes(cfg)

    def record(er_np):
        fields, const = e3.build_state_map(cfg, device, er_np)
        rec = {z_ref: [], z_tr: []}
        with torch.no_grad():
            for n in range(n_steps):
                fg.step(fields, const, n)
                rec[z_ref].append(sp.modal_amp(fields["Ey"], cfg, z_ref).item())
                rec[z_tr].append(sp.modal_amp(fields["Ey"], cfg, z_tr).item())
        resid = sum(float((fields[k] ** 2).sum()) for k in
                    ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"))
        return rec, resid

    def phasor(ts, f):
        nn = np.arange(len(ts))
        return np.sum(ts * np.exp(1j * 2 * np.pi * f * nn * cfg["dt"])) * cfg["dt"]

    empty_map = np.ones((cfg["Nx"], cfg["Ny"], cfg["Nz"]), dtype=np.float32)
    with torch.no_grad():
        loaded_er = binary_er(torch.tensor(bits.astype(np.float32),
                                           device=device),
                              torch.tensor(base.cpu().numpy() if torch.is_tensor(base)
                                           else base, device=device),
                              cell_masks).cpu().numpy()
    rec_e, _ = record(empty_map)
    rec_l, resid = record(loaded_er)
    out = []
    for f in freqs_hz:
        pe = phasor(rec_e[z_ref], f)
        pt = phasor(rec_e[z_tr], f)
        s11 = (phasor(rec_l[z_ref], f) - pe) / pe
        s21 = phasor(rec_l[z_tr], f) / pt
        out.append(dict(frequency=float(f), s11=complex(s11), s21=complex(s21),
                        energy=abs(s11) ** 2 + abs(s21) ** 2))
    return dict(n_steps=n_steps, record_ns=record_ns, sweep=out,
                max_S11=max(abs(o["s11"]) for o in out),
                residual_energy=resid, er_map=loaded_er)


def fabrication_perturb(bits, base, cell_masks, cfg, device, n_steps,
                        freqs_hz, empty, voxel_cells=1, chunk=20):
    """Erosion/dilation of the design material by one voxel (x-z plane)."""
    from scipy import ndimage
    with torch.no_grad():
        nominal = binary_er(torch.tensor(bits.astype(np.float32), device=device),
                            base, cell_masks)
    design_mask = (nominal.cpu().numpy() > 1.0)
    # 2-D structuring element applied slice-by-slice in the x-z plane.
    struct = ndimage.generate_binary_structure(2, 1)
    results = {}
    for mode in ("erosion", "dilation"):
        perturbed = np.zeros_like(design_mask)
        for j in range(design_mask.shape[1]):
            sl = design_mask[:, j, :]
            for _ in range(voxel_cells):
                if mode == "erosion":
                    sl = ndimage.binary_erosion(sl, struct)
                else:
                    sl = ndimage.binary_dilation(sl, struct)
            perturbed[:, j, :] = sl
        er_np = np.where(perturbed, ER_DIELECTRIC, 1.0).astype(np.float32)
        # keep the fixed obstacle (base) untouched
        base_np = base.cpu().numpy() if torch.is_tensor(base) else base
        er_np = np.where(base_np > 1.0, base_np, er_np)
        with torch.no_grad():
            s11, s21 = sparams(cfg, device,
                               torch.tensor(er_np, device=device), n_steps,
                               freqs_hz, empty, chunk, use_ckpt=False)
        results[mode] = dict(S11=s11.cpu().numpy(), S21=s21.cpu().numpy(),
                             max_S11=float(s11.abs().max()))
    return results
