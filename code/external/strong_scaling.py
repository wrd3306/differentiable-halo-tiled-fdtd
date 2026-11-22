"""Strong-scaling / throughput ladder for the production 3-D FDTD engine.

The production device runs use engine3d.build_state_map together with
fdtd_gpu.step (the L0 map engine). This script times that exact code path on a
ladder of grids that preserve the rectangular-waveguide aspect ratio
(a:b:L = 24:12:100), from the 0.5 mm device grid (230 k cells) up to a ~50 M
cell grid. For every grid it records the steady-state throughput, the peak
device memory, the sampled GPU utilization / power / clock, and the full-record
time-to-solution. All numbers are measured; nothing is extrapolated.
"""
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(HERE, "..", "engine"))
import engine3d as e3  # noqa: E402
import fdtd_gpu as fg  # noqa: E402

NVSMI = r"C:\Windows\System32\nvidia-smi.exe"
A_MM, B_MM, L_MM = 24.0, 12.0, 100.0
FULL_STEPS = 14836
TIMED_STEPS = 400
WARMUP = 60

# scale -> (Nx, Ny, Nz). Aspect ratio 2:1:8.333.
SCALES = {
    "scale2_0.50mm": (48, 24, 200),
    "scale4_0.25mm": (96, 48, 400),
    "scale8_0.125mm": (192, 96, 800),
    "scale12_0.083mm": (288, 144, 1200),
}


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._stop_evt = threading.Event()
        self.samples = []

    def run(self):
        q = ("--query-gpu=utilization.gpu,power.draw,clocks.current.graphics,"
             "memory.used")
        while not self._stop_evt.is_set():
            try:
                r = subprocess.run([NVSMI, q, "--format=csv"],
                                   capture_output=True, text=True, timeout=10)
                lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
                out = lines[-1]  # data row, skip header
                vals = [float(part.strip().split()[0]) for part in out.split(",")]
                u, p, clk, mem = vals
                self.samples.append((u, p, clk, mem))
            except Exception:
                pass
            time.sleep(0.1)

    def stop(self):
        self._stop_evt.set()


def make_cfg(shape):
    nx, ny, nz = shape
    dx = A_MM / nx * 1e-3
    dy = B_MM / ny * 1e-3
    dz = L_MM / nz * 1e-3
    dt = fg.cfl_dt(dx, dy, dz, 0.7)
    return {
        "Nx": nx, "Ny": ny, "Nz": nz,
        "dx": dx, "dy": dy, "dz": dz,
        "dt": dt, "f0": 10e9, "src_z": int(round(nz * 0.15)),
        "npml": 10,
    }


def main():
    device = torch.device("cuda")
    rows = []
    for tag, shape in SCALES.items():
        cfg = make_cfg(shape)
        n_cells = int(np.prod(shape))
        er_map = np.ones(shape, dtype=np.float32)
        torch.cuda.reset_peak_memory_stats(device)
        fields, const = e3.build_state_map(cfg, device, er_map)

        with torch.no_grad():
            for n in range(WARMUP):
                fg.step(fields, const, n)
            torch.cuda.synchronize()

            sampler = Sampler()
            sampler.start()
            t0 = time.perf_counter()
            for n in range(TIMED_STEPS):
                fg.step(fields, const, n + WARMUP)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            sampler.stop()
            sampler.join(timeout=2)

        peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1024**3
        mcups = n_cells * TIMED_STEPS / 1e6 / elapsed
        if sampler.samples:
            util = np.mean([s[0] for s in sampler.samples])
            power = np.mean([s[1] for s in sampler.samples])
            clk = np.mean([s[2] for s in sampler.samples])
            mem_used = np.max([s[3] for s in sampler.samples])
        else:
            util = power = clk = mem_used = float("nan")

        # Full-record time-to-solution from the measured steady-state rate.
        tts = n_cells * FULL_STEPS / (mcups * 1e6)
        row = {
            "tag": tag,
            "shape": list(shape),
            "cells": n_cells,
            "dt_s": cfg["dt"],
            "timed_steps": TIMED_STEPS,
            "elapsed_s": round(elapsed, 4),
            "mcells_s": round(mcups, 2),
            "peak_mem_allocated_gb": round(peak_mem_gb, 3),
            "gpu_utilization_pct": round(util, 1),
            "avg_power_w": round(power, 1),
            "avg_graphics_clock_mhz": round(clk, 0),
            "nvsmi_mem_used_mib": round(mem_used, 0),
            "full_record_steps": FULL_STEPS,
            "time_to_solution_s": round(tts, 2),
        }
        rows.append(row)
        print(json.dumps(row, indent=2))
        del fields, const
        torch.cuda.empty_cache()

    out = os.path.join(ROOT, "results", "external", "strong_scaling.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"engine": "production L0 (engine3d + fdtd_gpu.step)",
                   "device": torch.cuda.get_device_name(0),
                   "rows": rows}, fh, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
