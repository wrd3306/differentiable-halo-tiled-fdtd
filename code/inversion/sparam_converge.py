"""Convergence of FDTD S-parameters vs DFT window (diagnose leakage)."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.join(HERE, "..", "engine")
sys.path.insert(0, ENG)
sys.path.insert(0, HERE)
import sparam as sp  # noqa: E402


def main():
    cfg, geom = sp.make_config()
    device = torch.device("cuda")
    z_ref, z_tr = 25, 70
    a11, a21 = sp.analytic_sparams(4.0, cfg, geom)
    print(f"analytic |S11|={abs(a11):.4f} |S21|={abs(a21):.4f}")
    for dft_start, n_steps in ((380, 700), (600, 1100), (800, 1500), (1000, 2000)):
        empty = sp.run_dft(cfg, None, None, z_ref, z_tr, n_steps, dft_start, device)
        loaded = sp.run_dft(cfg, geom, 4.0, z_ref, z_tr, n_steps, dft_start, device)
        s11 = (loaded[z_ref] - empty[z_ref]) / empty[z_ref]
        s21 = loaded[z_tr] / empty[z_tr]
        energy = abs(s11) ** 2 + abs(s21) ** 2
        print(f"start={dft_start:5d} end={n_steps:5d} | |S11|={abs(s11):.4f} "
              f"|S21|={abs(s21):.4f} energy={energy:.4f}")


if __name__ == "__main__":
    main()
