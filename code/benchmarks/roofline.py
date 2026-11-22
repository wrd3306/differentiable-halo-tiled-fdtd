"""Empirical GPU roofline: achievable memory bandwidth and device specs.

Two micro-benchmarks on large fp32 tensors:
  - copy:  dst.copy_(src)            -> 1 read + 1 write = 2*N bytes
  - triad: c = a + b                 -> 2 reads + 1 write = 3*N bytes
Median of repeated, synchronised runs. FDTD is memory-bound, so this bandwidth
is the ceiling against which the FDTD achieved bandwidth is compared.
"""
import json
import time
import torch

OUT = r"C:\Users\wrd\Doubao\chats\2026-09-22\new-chat-1\systolic_gpu_fdtd\results\roofline.json"


def median(xs):
    return float(torch.tensor(xs).median())


def time_copy(n_elem, reps):
    src = torch.randn(n_elem, device="cuda")
    dst = torch.empty_like(src)
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        dst.copy_(src)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return median(ts)


def time_triad(n_elem, reps):
    a = torch.randn(n_elem, device="cuda")
    b = torch.randn(n_elem, device="cuda")
    c = torch.empty_like(a)
    for _ in range(3):
        c = a + b
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        c = a + b
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return median(ts)


def main():
    p = torch.cuda.get_device_properties(0)
    n_elem = 20_000_000
    nbytes = n_elem * 4
    reps = 20

    t_copy = time_copy(n_elem, reps)
    t_triad = time_triad(n_elem, reps)

    bw_copy = (2 * nbytes) / t_copy / 1e9
    bw_triad = (3 * nbytes) / t_triad / 1e9

    info = {
        "device": torch.cuda.get_device_name(0),
        "compute_capability": f"{p.major}.{p.minor}",
        "sms": p.multi_processor_count,
        "total_mem_gb": round(p.total_memory / 1e9, 2),
        "n_elem": n_elem,
        "copy_time_s": round(t_copy, 6),
        "copy_bw_gbs": round(bw_copy, 2),
        "triad_time_s": round(t_triad, 6),
        "triad_bw_gbs": round(bw_triad, 2),
        "peak_achieved_bw_gbs": round(max(bw_copy, bw_triad), 2),
    }
    print(json.dumps(info, indent=2))
    with open(OUT, "w") as f:
        json.dump(info, f, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
