"""Differentiable halo-tiled FDTD step.

The forward pass is the genuine shared-memory halo-tiled CUDA kernel
(fused_tiled.cu): a cooperative block loads its (tile + halo) field once and
reuses it across every curl component. The reverse pass is the discrete adjoint
of that same local update. Because the fused kernel is numerically identical to
the functional out-of-place step (verified separately to < 1e-8 over a physical
trajectory), the vector-Jacobian product is evaluated on the functional twin and
is therefore the exact gradient of the fused forward for the discrete update.

Only the ten state tensors and the material coefficient De carry gradients;
the CPML coefficients and PEC masks are fixed by geometry and are passed as
non-differentiated context.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import fdtd_diff as fd  # noqa: E402
from ext_build import get_ext  # noqa: E402

CU_SRC = os.path.join(ROOT, "code", "kernels", "fused_tiled.cu")
BUILD = os.path.join(ROOT, "code", "kernels", "_tiled_build")
_MODULE = None


def module():
    global _MODULE
    if _MODULE is None:
        _MODULE = get_ext("fused_tiled", [CU_SRC], BUILD)
    return _MODULE


def _source_scalar(n, dt, pulse):
    t = n * dt
    env = np.exp(-((t - pulse["t0"]) / pulse["tw"]) ** 2)
    return float(env * np.sin(2 * np.pi * pulse["fc"] * t))


class HaloTiledStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Ex, Ey, Ez, Hx, Hy, Hz, pe, px, phy, phx, De, c, n, geom):
        ctx.save_for_backward(Ex, Ey, Ez, Hx, Hy, Hz, pe, px, phy, phx, De)
        ctx.c = c
        ctx.n = n
        ctx.geom = geom

        mod = module()
        Ex, Ey, Ez, Hx, Hy, Hz, pe, px, phy, phx = [
            t.detach().clone() for t in (Ex, Ey, Ez, Hx, Hy, Hz, pe, px, phy, phx)]
        Nx, Ny, Nz = Ex.shape
        idx, idy, idz, mh, dt = (geom["idx"], geom["idy"], geom["idz"],
                                 geom["mh"], geom["dt"])
        mod.h_launch_tiled(
            Ex, Ey, Ez, c["bz"], c["cz"], Hx, Hy, Hz, pe, px,
            Nx, Ny, Nz, idx, idy, idz, mh)
        sv = _source_scalar(n, dt, geom["pulse"])
        mod.e_launch_tiled(
            Ex, Ey, Ez, Hx, Hy, Hz, c["Ce"], De, c["bz"], c["cz"],
            phy, phx, c["jprof"], c["mx"], c["my"],
            Nx, Ny, Nz, idx, idy, idz, sv)
        return Ex, Ey, Ez, Hx, Hy, Hz, pe, px, phy, phx

    @staticmethod
    def backward(ctx, gEx, gEy, gEz, gHx, gHy, gHz, gpe, gpx, gphy, gphx):
        saved = ctx.saved_tensors
        old = saved[:10]
        De = saved[10]
        x = [t.detach() for t in old]
        de = De.detach()
        c2 = dict(ctx.c)
        geom = ctx.geom
        # Nested autograd inside backward: re-enable grad explicitly.
        with torch.enable_grad():
            x = [t.clone().requires_grad_(True) for t in x]
            de = de.clone().requires_grad_(True)
            c2["De"] = de
            out = fd.step_fn(tuple(x), c2, ctx.n, geom["dt"], geom.get("f0"),
                             geom["idx"], geom["idy"], geom["idz"], geom["mh"],
                             geom["pulse"])
            need = x + [de]
            vjps = torch.autograd.grad(
                out, need,
                grad_outputs=tuple(g.contiguous() for g in (gEx, gEy, gEz, gHx,
                                                           gHy, gHz, gpe, gpx,
                                                           gphy, gphx)),
                retain_graph=False, allow_unused=True)
        g_state = [g if g is not None else torch.zeros_like(t)
                   for g, t in zip(vjps[:10], x)]
        g_de = vjps[10]
        if g_de is None:
            g_de = torch.zeros_like(de)
        # 10 state grads, De grad, then None for (c, n, geom) non-tensor inputs.
        return (*g_state, g_de, None, None, None)


def step(state, c, n, geom):
    """Functional wrapper: runs one differentiable halo-tiled step."""
    return HaloTiledStep.apply(
        state[0], state[1], state[2], state[3], state[4], state[5], state[6],
        state[7], state[8], state[9], c["De"], c, n, geom)
