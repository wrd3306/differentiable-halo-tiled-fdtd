# Differentiable Halo-Tiled GPU FDTD for Waveguide and Photonic Devices

A memory-bounded, reverse-mode differentiable finite-difference time-domain
(FDTD) solver for rectangular-waveguide and photonic components. The forward
engine is a roofline-guided **halo-tiled shared-memory GPU kernel**; the
reverse pass is built from functional out-of-place updates and gradient
checkpointing, so reverse-mode gradients are available under bounded peak
memory without retaining the full per-step autograd graph.

Every reported number is produced by a full-wave solve. No surrogate, fitted,
or interpolated model is used anywhere in the forward or inverse pipeline.

## What is inside

```
code/
  engine/        Core FDTD and differentiable engines
    fdtd_gpu.py            L0 production map engine (PyTorch, in-place)
    engine3d.py            3-D material engine (PEC walls, CPML, current sheet)
    fdtd_diff.py           L5 functional differentiable engine + checkpoints
    fdtd_diff3d.py         3-D constants for the functional engine
    fdtd_diff_tiled.py     Tiled differentiable engine (forward + reverse)
    halo_tiled_step.py     autograd.Function wrapping the fused CUDA kernel
    ext_build.py           Load a prebuilt extension or JIT-build it
    verify_tiled_match.py  Numerical equivalence of the two forward paths
    grad_check.py          Finite-difference gradient checks (functional)
    grad_check_tiled.py    Finite-difference gradient checks (tiled)
    check_stability.py     CFL and bounded-field checks
  kernels/       CUDA C++ sources
    fused_tiled.cu         Halo-tiled shared-memory stencil
    fused_fdtd.cu          Fused single-kernel stencil
    fused_bf16.cu          Mixed-precision variant
  inversion/     Devices, S-parameters, and inverse design
    devices3d.py           3-D device library and rasterizer
    run3d.py               Two-port full-wave extraction
    validate3d.py          Identity, reciprocity, symmetry, convergence
    snapshot3d.py          3-D field snapshots
    optimize3d.py          Multi-start 3-D anti-reflection optimizer
    pixel_design.py        Pixelated broadband design (Adam, GA, PSO, verify)
    run_pixel_tiled.py     Broadband design driven through the tiled kernel
    grin_design.py / grin_lp.py      Graded-index (GRIN) designs
    grin_verify.py / grinlp_verify.py / grin_fine_verify.py
    grin_fab.py            Manufacturing-tolerance study
    sparam.py              S-parameter extraction and analytic slab reference
  external/      Independent reference solvers and cross-checks
    mode_matching.py       Modal mode-matching solver
    fdfd_2d_check.py       Frequency-domain FDFD arbiter
    fem_2d_check.py        Weak-form FEM arbiter
    build_voxel_masks.py   Unified hard voxel masks for all solvers
    strong_scaling.py      Strong-scaling and hardware measurements
    hw_profile.py          Roofline / bandwidth profiling
    meep_*.py / openems_*.py         External solver adapters
  benchmarks/    Roofline and kernel-level benchmarks
data/
  voxels/        Hard voxel masks (npz + CSXCAD h5) shared by every solver
results/         Recorded JSON outputs used in the paper
```

## Requirements

- NVIDIA GPU (developed on an RTX 5060 Ti, Blackwell, 16 GB); CUDA runtime
  compatible with the installed PyTorch.
- Python 3.10 with PyTorch 2.11 (CUDA 12.8 build), NumPy, SciPy.
- To (re)build the CUDA extension: CUDA Toolkit 12.8 and MSVC C++ Build Tools
  on Windows. A prebuilt extension is used directly when present, so the
  toolchain is only needed for a clean rebuild.
- Optional external solvers for the cross-method comparison: gprMax, MEEP,
  openEMS. The core engine and all inverse-design runs do not require them.

## Quick start

```bash
# 1. Build or load the fused halo-tiled extension (prebuilt .pyd is used
#    automatically; otherwise it JIT-compiles via cpp_extension).
python code/engine/verify_tiled_match.py

# 2. Confirm the reverse-mode gradients against finite differences.
python code/engine/grad_check_tiled.py

# 3. Run the 3-D validation suite (identity, reciprocity, symmetry).
python code/inversion/validate3d.py

# 4. Drive a broadband pixelated design through the tiled differentiable path,
#    then confirm it with an independent non-autograd simulation.
python code/inversion/run_pixel_tiled.py
```

## Reproducibility and verification

- The tiled CUDA forward and the functional PyTorch forward advance from the
  same initial state and agree to within floating-point round-off, so the
  reverse pass is exact for the discrete update.
- Gradients are checked against central finite differences.
- Final designs are re-simulated with a separate non-autograd engine over a
  longer record, avoiding an inverse crime.
- The same hard voxel mask is fed to every solver in the cross-method study,
  removing geometry differences as a source of disagreement.

## License

Released under the MIT License. See [LICENSE](LICENSE).
