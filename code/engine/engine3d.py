"""General-material 3-D FDTD builder.

The only difference from fdtd_gpu.build_state is that the medium is specified
by a full 3-D relative-permittivity map instead of a single box. For the
lossless materials used here (sigma = 0), Ce = 1 everywhere and De =
dt/(EPS0*er) cell by cell. The CPML, PEC walls, source, and the time step are
identical to fdtd_gpu, so fg.step can be reused unchanged and the physics stay
bit-consistent with the reference engine.
"""
import numpy as np
import torch

import fdtd_gpu as fg


def build_state_map(cfg, device, er_map, dtype=torch.float32):
    """Build fields and constants from a full 3-D er map (lossless)."""
    Nx, Ny, Nz = cfg["Nx"], cfg["Ny"], cfg["Nz"]
    dx, dy, dz = cfg["dx"], cfg["dy"], cfg["dz"]
    dt, n_pml = cfg["dt"], cfg["npml"]
    shape = (Nx, Ny, Nz)

    er_t = torch.as_tensor(er_map, device=device, dtype=dtype)
    Ce = torch.ones(shape, device=device, dtype=dtype)          # lossless Ce=1
    De = dt / (fg.EPS0 * er_t)                                  # cellwise

    # CPML (identical to fdtd_gpu): Taflove optimal cubic sigma, kappa=1.
    m = 3
    sig_1d = torch.zeros(Nz)
    kap_1d = torch.ones(Nz)
    for k in range(n_pml):
        d = (n_pml - k) / n_pml
        sig = 0.8 * (m + 1) / (fg.ETA0 * dz) * d ** m
        sig_1d[k] = sig
        sig_1d[Nz - 1 - k] = sig
        kap_1d[k] = 1.0
        kap_1d[Nz - 1 - k] = 1.0
    sig3 = sig_1d.view(1, 1, Nz).expand(Nx, Ny, Nz).to(device=device, dtype=dtype)
    kap3 = kap_1d.view(1, 1, Nz).expand(Nx, Ny, Nz).to(device=device, dtype=dtype)
    bz = torch.exp(-(sig3 / kap3) / fg.EPS0 * dt)
    denom = kap3 * sig3
    az = torch.where(denom > 0, (bz - 1) * sig3 / torch.where(
        denom > 0, denom, torch.ones_like(denom)), torch.zeros_like(bz))

    # PEC walls.
    mx = torch.ones(shape, device=device, dtype=dtype)
    mx[0, :, :] = 0
    mx[-1, :, :] = 0
    my = torch.ones(shape, device=device, dtype=dtype)
    my[:, 0, :] = 0
    my[:, -1, :] = 0

    # TE10 current-sheet source.
    src_z = cfg["src_z"]
    xi = torch.arange(1, Nx - 1, dtype=dtype)
    jprof = torch.zeros(shape, device=device, dtype=dtype)
    jprof[1:Nx - 1, 1:Ny - 1, src_z] = torch.sin(
        np.pi * xi * dx / (Nx * dx)).unsqueeze(1)

    fields = {
        "Ex": torch.zeros(shape, device=device, dtype=dtype),
        "Ey": torch.zeros(shape, device=device, dtype=dtype),
        "Ez": torch.zeros(shape, device=device, dtype=dtype),
        "Hx": torch.zeros(shape, device=device, dtype=dtype),
        "Hy": torch.zeros(shape, device=device, dtype=dtype),
        "Hz": torch.zeros(shape, device=device, dtype=dtype),
        "pe": torch.zeros(shape, device=device, dtype=dtype),
        "px": torch.zeros(shape, device=device, dtype=dtype),
        "phy": torch.zeros(shape, device=device, dtype=dtype),
        "phx": torch.zeros(shape, device=device, dtype=dtype),
    }
    const = dict(dx=dx, dy=dy, dz=dz, dt=dt, f0=cfg["f0"], bz=bz, az=az,
                 kap=kap3, mx=mx, my=my, Ce=Ce, De=De, jprof=jprof,
                 j_amp=1.0, source_value=fg.source_value,
                 pulse=cfg.get("pulse"))
    return fields, const
