"""Export the 3-D device box definitions to a dependency-free JSON file.

MEEP (inside WSL) and openEMS (native) read this file so that every solver uses
the identical physical geometry without importing the PyTorch engine.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "inversion"))

import devices3d as d3  # noqa: E402

out = {
    "waveguide_mm": {"a": 24, "b": 12, "L": 100},
    "grid_mm": 0.5,
    "f0_hz": 10e9,
    "devices": {name: [list(map(float, box)) for box in boxes]
                for name, boxes in d3.DEVICES.items()},
}

out_path = os.path.join(HERE, "devices_geom.json")
with open(out_path, "w") as fh:
    json.dump(out, fh, indent=1)
print("wrote", out_path)
print("devices:", list(out["devices"].keys()))
