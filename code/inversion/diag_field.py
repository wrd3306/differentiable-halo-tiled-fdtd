"""Diagnostic: steady-state modal phasor magnitude along z, empty vs loaded."""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.join(HERE, "..", "engine")
sys.path.insert(0, ENG)
sys.path.insert(0, HERE)
import fdtd_gpu as fg  # noqa: E402
import sparam as sp  # noqa: E402


def run_profile(cfg, geom, er, planes, n_steps, dft_start, device):
    block = None if er is None else ("box", geom, er, 0.0)
    fields, const = fg.build_state(cfg, device, block=block)
    omega = 2 * np.pi * cfg["f0"]
    re = {z: 0.0 for z in planes}
    im = {z: 0.0 for z in planes}
    with torch.no_grad():
        for n in range(n_steps):
            fg.step(fields, const, n)
            if n >= dft_start:
                wt = omega * n * cfg["dt"]
                cw, sw = np.cos(wt), np.sin(wt)
                for z in planes:
                    a = sp.modal_amp(fields["Ey"], cfg, z).item()
                    re[z] += a * cw * cfg["dt"]
                    im[z] += a * sw * cfg["dt"]
    return {z: abs(complex(re[z], im[z])) for z in planes}


def main():
    cfg, geom = sp.make_config()
    device = torch.device("cuda")
    planes = list(range(20, 81, 5))
    n_steps, dft_start = 2000, 0
    empty = run_profile(cfg, None, None, planes, n_steps, dft_start, device)
    loaded = run_profile(cfg, geom, 4.0, planes, n_steps, dft_start, device)
    print("z   empty|A|      loaded|A|")
    for z in planes:
        print(f"{z:3d}  {empty[z]:.3e}   {loaded[z]:.3e}")


if __name__ == "__main__":
    main()
