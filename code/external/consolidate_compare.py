"""Consolidate the three-way (self GPU / MEEP / openEMS) same-grid comparison.

All three solvers use the identical 3-D device geometry (devices_geom.json),
the same uniform 0.5 mm grid (230,400 cells), and the same 7-12 GHz band with
26 frequency samples. This script merges the per-device S-parameters at 10 GHz,
the energy-balance checks, and the per-step throughput (MCells/s) into a single
JSON + CSV. No fitting, interpolation, or synthetic data is used.
"""
import csv
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "results", "external")

# Self GPU throughput measured repeatedly on the exact 48x24x200 grid
# (cells x steps / elapsed). Four timed runs give the following rates; the
# median is reported as the representative figure with the spread recorded.
SELF_GPU_RUNS = [314.47, 308.57, 323.83, 324.30]
SELF_GPU_MED = statistics.median(SELF_GPU_RUNS)
SELF_GPU_NAME = "NVIDIA GeForce RTX 5060 Ti"


def load(name):
    with open(os.path.join(RES, name), "r") as f:
        return json.load(f)


def index_by_name(rows):
    return {r["name"]: r for r in rows}


def main():
    meep = index_by_name(load("meep_results.json"))
    openems = index_by_name(load("openems_results.json"))
    openems_st = index_by_name(load("openems_single_thread.json"))
    self_ = index_by_name(load(os.path.join(ROOT, "results", "devices3d.json")))

    names = list(self_.keys())
    assert set(meep) == set(names) and set(openems) == set(names)

    rows = []
    for name in names:
        s = self_[name]
        m = meep[name]["at10"]
        o = openems[name]["at10"]
        rows.append({
            "name": name,
            "self_S11": s["S11_abs"],
            "self_S21": s["S21_abs"],
            "self_energy": s["energy_balance"],
            "meep_S11": m["S11_abs"],
            "meep_S21": m["S21_abs"],
            "meep_energy": m["energy"],
            "openems_S11": o["S11_abs"],
            "openems_S21": o["S21_abs"],
            "openems_energy": o["energy"],
        })

    # Per-step throughput (MCells/s) across the nine devices. The controlled
    # single-core openEMS rate is used for the apples-to-apples CPU comparison;
    # the auto-tuned default and the fine-grid multi-thread rate are recorded as
    # context so the thread-count difference is fully transparent.
    meep_rate = [meep[n]["mcells_per_s"] for n in names]
    openems_st_rate = [openems_st[n]["mcells_per_s"] for n in names]
    openems_default_rate = [openems[n]["mcells_per_s"] for n in names]
    openems_fine = index_by_name(load("openems_fine_results.json"))
    openems_fine_rate = [openems_fine[n]["mcells_per_s"] for n in openems_fine]
    throughput = {
        "grid_mm": 0.5,
        "n_cells": 230400,
        "self_GPU": {
            "device": SELF_GPU_NAME,
            "median_MCells_s": SELF_GPU_MED,
            "runs_MCells_s": SELF_GPU_RUNS,
            "min": min(SELF_GPU_RUNS),
            "max": max(SELF_GPU_RUNS),
            "courant": 0.4044,
            "dt_s": 6.74e-13,
        },
        "MEEP_CPU": {
            "version": "1.35.1 nompi (single core)",
            "mean_MCells_s": statistics.mean(meep_rate),
            "min": min(meep_rate),
            "max": max(meep_rate),
            "courant": 0.4044,
            "dt_s": 6.74e-13,
        },
        "openEMS_CPU_single_thread": {
            "version": "0.37.0 MSVC (numThreads=1)",
            "mean_MCells_s": statistics.mean(openems_st_rate),
            "min": min(openems_st_rate),
            "max": max(openems_st_rate),
            "courant": 0.99,
            "dt_s": 9.629166007732353e-13,
        },
        "openEMS_context": {
            "auto_tuned_default_0p5mm": {
                "mean_MCells_s": statistics.mean(openems_default_rate),
                "min": min(openems_default_rate),
                "max": max(openems_default_rate),
                "note": "Small 0.5 mm grid: auto-tuner selects a low thread "
                        "count, so the default rate is close to single core.",
            },
            "multithread_0p25mm": {
                "mean_MCells_s": statistics.mean(openems_fine_rate),
                "min": min(openems_fine_rate),
                "max": max(openems_fine_rate),
                "note": "Larger 1.9 M-cell grid: auto-tuner uses 8-9 threads.",
            },
            "msvc_note": "Upstream docs flag a known performance penalty in "
                         "the MSVC build relative to the GCC build.",
        },
    }
    # Speedup of the self GPU engine over each single-core CPU solver.
    throughput["speedup_vs_MEEP"] = SELF_GPU_MED / statistics.mean(meep_rate)
    throughput["speedup_vs_openEMS_single"] = \
        SELF_GPU_MED / statistics.mean(openems_st_rate)

    out = {
        "grid_mm": 0.5,
        "n_cells": 230400,
        "frequency_hz": 10e9,
        "analytic_full_slab": {"S11": 0.04448, "S21": 0.99901, "energy": 1.0},
        "devices": rows,
        "throughput": throughput,
    }
    with open(os.path.join(RES, "three_way_compare.json"), "w") as f:
        json.dump(out, f, indent=2)

    csv_path = os.path.join(RES, "three_way_compare.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["device",
                    "self|S11|", "self|S21|", "self E",
                    "MEEP|S11|", "MEEP|S21|", "MEEP E",
                    "openEMS|S11|", "openEMS|S21|", "openEMS E"])
        for r in rows:
            w.writerow([r["name"],
                        "%.4f" % r["self_S11"], "%.4f" % r["self_S21"],
                        "%.4f" % r["self_energy"],
                        "%.4f" % r["meep_S11"], "%.4f" % r["meep_S21"],
                        "%.4f" % r["meep_energy"],
                        "%.4f" % r["openems_S11"], "%.4f" % r["openems_S21"],
                        "%.4f" % r["openems_energy"]])

    print("wrote three_way_compare.json / .csv with %d devices" % len(rows))
    print("\n-- S-parameters at 10 GHz (|S11| / |S21| / energy) --")
    for r in rows:
        print("%-16s self %.4f/%.4f/%.4f | MEEP %.4f/%.4f/%.4f | oEMS %.4f/%.4f/%.4f"
              % (r["name"], r["self_S11"], r["self_S21"], r["self_energy"],
                 r["meep_S11"], r["meep_S21"], r["meep_energy"],
                 r["openems_S11"], r["openems_S21"], r["openems_energy"]))
    print("\n-- Throughput (MCells/s) --")
    print("self GPU median: %.2f (%.1f-%.1f)" %
          (SELF_GPU_MED, min(SELF_GPU_RUNS), max(SELF_GPU_RUNS)))
    print("MEEP CPU (1 core) mean:   %.2f (%.1f-%.1f)" %
          (statistics.mean(meep_rate), min(meep_rate), max(meep_rate)))
    print("openEMS (1 thread) mean:  %.2f (%.1f-%.1f)" %
          (statistics.mean(openems_st_rate),
           min(openems_st_rate), max(openems_st_rate)))
    print("openEMS auto default:     %.2f ; 0.25mm multi-thread: %.2f" %
          (statistics.mean(openems_default_rate),
           statistics.mean(openems_fine_rate)))
    print("speedup vs MEEP: %.2fx ; vs openEMS single: %.2fx" %
          (throughput["speedup_vs_MEEP"],
           throughput["speedup_vs_openEMS_single"]))


if __name__ == "__main__":
    main()
