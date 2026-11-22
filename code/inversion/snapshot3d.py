"""Capture real 3-D field snapshots during full-wave device interaction.

The actual Yee-grid field arrays are saved (no interpolation or fitting) at a
set of timesteps around the peak of the device interaction, together with the
geometry mask. The figure script renders the recorded arrays directly. Both a
genuine 3-D device (an isolated post, whose field is non-uniform in all three
axes) and the full-cross-section slab (uniform in x and y) are captured so the
contrast between the 3-D and the effectively 1-D case is visible.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(HERE, "..", "engine"))
sys.path.insert(0, HERE)
import devices3d as d3  # noqa: E402
import engine3d as e3  # noqa: E402
import fdtd_gpu as fg  # noqa: E402


def capture(name, scale=2, record_ns=4.0, save_fracs=(0.45, 0.55, 0.65)):
    """Run one device and save the real Ey array at selected timesteps.

    save_fracs are fractions of the total run; the timestep nearest the peak
    field inside the device is always included."""
    device = torch.device("cuda")
    cfg = d3.config3d(scale)
    n_steps = int(np.ceil(record_ns * 1e-9 / cfg["dt"]))
    er_map = d3.er_map(name, cfg)
    fields, const = e3.build_state_map(cfg, device, er_map)

    # Track the peak |Ey| inside the device region to locate interaction.
    dev_cells = torch.tensor(er_map > 1.0, device=device)
    peak_val, peak_n = -1.0, 0
    saved = {}
    target_ns = sorted(set(int(round(f * n_steps)) for f in save_fracs))
    with torch.no_grad():
        for n in range(n_steps):
            fg.step(fields, const, n)
            v = float(fields["Ey"][dev_cells].abs().max())
            if v > peak_val:
                peak_val, peak_n = v, n
            if n in target_ns:
                saved[n] = fields["Ey"].cpu().numpy().copy()
    if peak_n not in saved:
        saved[peak_n] = fields["Ey"].cpu().numpy().copy()

    out = os.path.join(ROOT, "results", "snapshot_%s.npz" % name)
    arrays = {"er_map": er_map.astype(np.float32),
              "dx": np.array(cfg["dx"]), "dt": np.array(cfg["dt"]),
              "peak_step": np.array(peak_n), "peak_val": np.array(peak_val)}
    for n, a in sorted(saved.items()):
        arrays["Ey_%d" % n] = a
    np.savez_compressed(out, **arrays)
    print("saved", out, "peak step", peak_n, "value %.4e" % peak_val,
          "snapshots", sorted(saved))
    return out


def main():
    # A genuine 3-D device and the effectively 1-D full slab for contrast.
    capture("centered_post", scale=2)
    capture("full_slab", scale=2)


if __name__ == "__main__":
    main()
