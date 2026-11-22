import json
import torch

p = torch.cuda.get_device_properties(0)
info = {
    "name": p.name,
    "total_memory_gb": round(p.total_memory / 1024**3, 3),
    "multi_processor_count": p.multi_processor_count,
    "compute_capability": f"{p.major}.{p.minor}",
    "max_threads_per_sm": p.max_threads_per_multi_processor,
    "warp_size": p.warp_size,
    "l2_cache_size_mb": round(p.L2_cache_size / 1024**2, 2),
    "memory_bus_width_bits": p.memory_bus_width,
    "memory_clock_rate_mhz": round(p.memory_clock_rate / 1e6, 1),
    "global_memory_bus_width": p.memory_bus_width,
    "max_shared_memory_per_sm_kb": round(p.shared_memory_per_multiprocessor / 1024, 1),
    "max_shared_memory_per_block_kb": round(p.shared_memory_per_block / 1024, 1),
    "torch_version": torch.__version__,
    "cuda_version_torch": torch.version.cuda,
}
# Theoretical bandwidth: DDR means 2 transfers/clock.
bw = 2 * p.memory_bus_width * p.memory_clock_rate / 8 / 1e9
info["theoretical_bw_gbs_ddr"] = round(bw, 1)
print(json.dumps(info, indent=2))
