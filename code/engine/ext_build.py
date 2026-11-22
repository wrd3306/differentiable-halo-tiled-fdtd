"""Centralized, verified cpp_extension build configuration (L3+).

Reusable load() wrapper carrying the exact flags that were proven to work on
this machine: VS2022 Build Tools (v143 14.44) + CUDA 12.8 nvcc + ninja.

Gotchas already handled here:
  - "-DUSE_CUDA" so compiled_autograd.h takes the Windows-safe branch
    (avoids MSVC C2872 'std' ambiguity in its if-constexpr chain);
  - the Build Tools install did not register the Windows SDK, so its UCRT/UM
    include and lib paths are passed explicitly.
"""
import os

from torch.utils.cpp_extension import load

KITS = r"C:\Program Files (x86)\Windows Kits\10"
SDK = "10.0.26100.0"

_CUDA_FLAGS = [
    "-DUSE_CUDA",
    "-O3",
    f"-I{KITS}\\Include\\{SDK}\\shared",
    f"-I{KITS}\\Include\\{SDK}\\ucrt",
    f"-I{KITS}\\Include\\{SDK}\\um",
]


def import_prebuilt(name, build_dir):
    """Import an already-compiled <name>.pyd without invoking the toolchain.

    Returns None when no compiled module is present so callers can fall back to
    a JIT build. This keeps runtime imports working outside a vcvars64 shell.
    """
    pyd = os.path.join(build_dir, name + ".pyd")
    if not os.path.exists(pyd):
        return None
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, pyd)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ext(name, sources, build_dir, extra_cuda_cflags=None, verbose=False):
    os.makedirs(build_dir, exist_ok=True)
    flags = list(_CUDA_FLAGS)
    if extra_cuda_cflags:
        flags.extend(extra_cuda_cflags)
    return load(
        name=name,
        sources=sources,
        extra_cuda_cflags=flags,
        extra_ldflags=[
            f"/LIBPATH:{KITS}\\Lib\\{SDK}\\ucrt\\x64",
            f"/LIBPATH:{KITS}\\Lib\\{SDK}\\um\\x64",
        ],
        build_directory=build_dir,
        verbose=verbose,
    )


def get_ext(name, sources, build_dir, extra_cuda_cflags=None, verbose=False):
    """Use a prebuilt module when available; otherwise JIT-compile it."""
    module = import_prebuilt(name, build_dir)
    if module is not None:
        return module
    return load_ext(name, sources, build_dir, extra_cuda_cflags, verbose)
