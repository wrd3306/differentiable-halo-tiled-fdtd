"""Full 3-D vector FDTD engine on GPU (L0 naive baseline).

Faithful port of the production physics (solver_long.py): six interleaved field
components, z-directed CPML with psi auxiliary variables, PEC waveguide walls,
TE10 current-sheet source. The L0 style keeps the naive vectorised structure -
fresh derivative tensors via zeros_like and separate element-wise kernels - so
the cost of temporary global-memory traffic is measured before fusion.

Device-agnostic: pass device=torch.device('cuda'|'cpu').
"""
import numpy as np
import torch

C0 = 299792458.0
EPS0 = 8.854187817e-12
MU0 = 1.2566370614e-6
ETA0 = np.sqrt(MU0 / EPS0)


def cfl_dt(dx, dy, dz, cfl=0.7):
    return cfl * min(dx, dy, dz) / (C0 * np.sqrt(3.0))


def source_value(cfg, n):
    """Scalar source waveform: CW sinusoid, or a Gaussian-modulated pulse."""
    t = n * cfg["dt"]
    if cfg.get("pulse"):
        p = cfg["pulse"]
        env = np.exp(-((t - p["t0"]) / p["tw"]) ** 2)
        return env * np.sin(2 * np.pi * p["fc"] * t)
    return np.sin(2 * np.pi * cfg["f0"] * t)


def build_state(cfg, device, dtype=torch.float32, block=None):
    """Return state dict and precomputed coefficients.

    cfg: dict with Nx,Ny,Nz,dx,dy,dz,dt,f0,src_z,npml
    block: optional ('box', (x0,x1,y0,y1,z0,z1), er, sigma) for lossy material.
    """
    Nx, Ny, Nz = cfg["Nx"], cfg["Ny"], cfg["Nz"]
    dx, dy, dz = cfg["dx"], cfg["dy"], cfg["dz"]
    dt, f0 = cfg["dt"], cfg["f0"]
    n_pml = cfg["npml"]
    sz_shape = (Nx, Ny, Nz)

    # Material coefficients (vacuum default).
    Ce = torch.ones(sz_shape, device=device, dtype=dtype)
    De = torch.full(sz_shape, dt / EPS0, device=device, dtype=dtype)
    if block is not None:
        kind, (x0, x1, y0, y1, z0, z1), er, sig = block
        eps_abs = EPS0 * er
        denom = 2 * eps_abs + sig * dt
        Ce_b = (2 * eps_abs - sig * dt) / denom
        De_b = 2 * dt / denom
        Ce[x0:x1, y0:y1, z0:z1] = Ce_b
        De[x0:x1, y0:y1, z0:z1] = De_b

    # Standard CPML z profiles: cubic sigma (Taflove optimal), kappa, alpha.
    m = 3
    kappa_max = 1.0
    alpha_max = 0.0
    sig_1d = torch.zeros(Nz)
    kap_1d = torch.ones(Nz)
    alp_1d = torch.zeros(Nz)
    for k in range(n_pml):
        d = (n_pml - k) / n_pml  # 1 at outer cell, small at inner edge
        sig = 0.8 * (m + 1) / (ETA0 * dz) * d ** m
        kap = 1 + (kappa_max - 1) * d ** m
        alp = alpha_max * d
        sig_1d[k] = sig
        sig_1d[Nz - 1 - k] = sig
        kap_1d[k] = kap
        kap_1d[Nz - 1 - k] = kap
        alp_1d[k] = alp
        alp_1d[Nz - 1 - k] = alp
    sig3 = sig_1d.view(1, 1, Nz).expand(Nx, Ny, Nz).to(device=device, dtype=dtype)
    kap3 = kap_1d.view(1, 1, Nz).expand(Nx, Ny, Nz).to(device=device, dtype=dtype)
    alp3 = alp_1d.view(1, 1, Nz).expand(Nx, Ny, Nz).to(device=device, dtype=dtype)
    bz = torch.exp(-(sig3 / kap3 + alp3) / EPS0 * dt)
    denom = kap3 * (sig3 + alp3 * kap3)
    az = torch.where(denom > 0,
                     (bz - 1) * sig3 / torch.where(denom > 0, denom,
                                                   torch.ones_like(denom)),
                     torch.zeros_like(bz))

    # PEC wall masks.
    mx = torch.ones(sz_shape, device=device, dtype=dtype)
    mx[0, :, :] = 0
    mx[-1, :, :] = 0
    my = torch.ones(sz_shape, device=device, dtype=dtype)
    my[:, 0, :] = 0
    my[:, -1, :] = 0

    # TE10 current-sheet source profile (precomputed; scalar sv applied per step).
    src_z = cfg["src_z"]
    xi = torch.arange(1, Nx - 1, dtype=dtype)
    j_amp = 1.0  # timing only; amplitude irrelevant to throughput
    jprof = torch.zeros(sz_shape, device=device, dtype=dtype)
    jprof[1:Nx - 1, 1:Ny - 1, src_z] = torch.sin(np.pi * xi * dx / (Nx * dx)).unsqueeze(1)

    fields = {
        "Ex": torch.zeros(sz_shape, device=device, dtype=dtype),
        "Ey": torch.zeros(sz_shape, device=device, dtype=dtype),
        "Ez": torch.zeros(sz_shape, device=device, dtype=dtype),
        "Hx": torch.zeros(sz_shape, device=device, dtype=dtype),
        "Hy": torch.zeros(sz_shape, device=device, dtype=dtype),
        "Hz": torch.zeros(sz_shape, device=device, dtype=dtype),
        # CPML psi variables.
        "pe": torch.zeros(sz_shape, device=device, dtype=dtype),
        "px": torch.zeros(sz_shape, device=device, dtype=dtype),
        "phy": torch.zeros(sz_shape, device=device, dtype=dtype),
        "phx": torch.zeros(sz_shape, device=device, dtype=dtype),
    }
    const = dict(dx=dx, dy=dy, dz=dz, dt=dt, f0=f0, bz=bz, az=az,
                 kap=kap3, mx=mx, my=my, Ce=Ce, De=De, jprof=jprof,
                 j_amp=j_amp, source_value=source_value, pulse=cfg.get("pulse"))
    return fields, const


def step(f, c, n, src_scale=1.0):
    """One naive FDTD time step (L0). src_scale ramps the source smoothly."""
    dt = c["dt"]
    dx, dy, dz = c["dx"], c["dy"], c["dz"]
    bz, az, kap = c["bz"], c["az"], c["kap"]

    # ---- E spatial derivatives (6) ----
    dEydz = torch.zeros_like(f["Ey"])
    dEydz[:, :, :-1] = (f["Ey"][:, :, 1:] - f["Ey"][:, :, :-1]) / dz
    dExdz = torch.zeros_like(f["Ex"])
    dExdz[:, :, :-1] = (f["Ex"][:, :, 1:] - f["Ex"][:, :, :-1]) / dz
    dEzdy = torch.zeros_like(f["Ez"])
    dEzdy[:, :-1, :] = (f["Ez"][:, 1:, :] - f["Ez"][:, :-1, :]) / dy
    dEzdx = torch.zeros_like(f["Ez"])
    dEzdx[:-1, :, :] = (f["Ez"][1:, :, :] - f["Ez"][:-1, :, :]) / dx
    dEydx = torch.zeros_like(f["Ey"])
    dEydx[:-1, :, :] = (f["Ey"][1:, :, :] - f["Ey"][:-1, :, :]) / dx
    dExdy = torch.zeros_like(f["Ex"])
    dExdy[:, :-1, :] = (f["Ex"][:, 1:, :] - f["Ex"][:, :-1, :]) / dy

    # ---- H update with CPML effective z-derivative (dE/kap + psi) ----
    f["pe"] = bz * f["pe"] + az * dEydz
    f["px"] = bz * f["px"] + az * dExdz
    eyz = dEydz / kap + f["pe"]
    exz = dExdz / kap + f["px"]
    f["Hx"] = f["Hx"] - (dt / MU0) * (dEzdy - eyz)
    f["Hy"] = f["Hy"] - (dt / MU0) * (exz - dEzdx)
    f["Hz"] = f["Hz"] - (dt / MU0) * (dEydx - dExdy)

    # ---- H spatial derivatives (6) ----
    dHzdy = torch.zeros_like(f["Hz"])
    dHzdy[:, 1:, :] = (f["Hz"][:, 1:, :] - f["Hz"][:, :-1, :]) / dy
    dHydz = torch.zeros_like(f["Hy"])
    dHydz[:, :, 1:] = (f["Hy"][:, :, 1:] - f["Hy"][:, :, :-1]) / dz
    dHxdz = torch.zeros_like(f["Hx"])
    dHxdz[:, :, 1:] = (f["Hx"][:, :, 1:] - f["Hx"][:, :, :-1]) / dz
    dHzdx = torch.zeros_like(f["Hz"])
    dHzdx[1:, :, :] = (f["Hz"][1:, :, :] - f["Hz"][:-1, :, :]) / dx
    dHydx = torch.zeros_like(f["Hy"])
    dHydx[1:, :, :] = (f["Hy"][1:, :, :] - f["Hy"][:-1, :, :]) / dx
    dHxdy = torch.zeros_like(f["Hx"])
    dHxdy[:, 1:, :] = (f["Hx"][:, 1:, :] - f["Hx"][:, :-1, :]) / dy

    # ---- E update with CPML effective z-derivative (dH/kap + psi) ----
    f["phy"] = bz * f["phy"] + az * dHydz
    f["phx"] = bz * f["phx"] + az * dHxdz
    hyz = dHydz / kap + f["phy"]
    hxz = dHxdz / kap + f["phx"]
    f["Ex"] = c["Ce"] * f["Ex"] + c["De"] * (dHzdy - hyz)
    f["Ey"] = c["Ce"] * f["Ey"] + c["De"] * (hxz - dHzdx)
    f["Ez"] = c["Ce"] * f["Ez"] + c["De"] * (dHydx - dHxdy)

    # ---- source + PEC ----
    sv = c["source_value"](c, n) * src_scale
    f["Ey"] = f["Ey"] - c["De"] * c["j_amp"] * sv * c["jprof"]
    f["Ex"] = f["Ex"] * c["my"]
    f["Ey"] = f["Ey"] * c["mx"]
    f["Ez"] = f["Ez"] * c["mx"] * c["my"]


def run(cfg, device, n_steps, dtype=torch.float32, block=None, warmup=5):
    fields, const = build_state(cfg, device, dtype=dtype, block=block)
    with torch.no_grad():
        for n in range(warmup):
            step(fields, const, n)
        if device.type == "cuda":
            torch.cuda.synchronize()
        import time
        t0 = time.perf_counter()
        for n in range(n_steps):
            step(fields, const, n + warmup)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return elapsed, fields
