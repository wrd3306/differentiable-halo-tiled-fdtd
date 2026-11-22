import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "..", "results", "external", "gprmax_bench")
os.makedirs(OUTDIR, exist_ok=True)

# tag -> (dx, nx, ny, nz)
GRIDS = {
    "scale2": (0.0005, 48, 24, 200),
    "scale4": (0.00025, 96, 48, 400),
    "scale8": (0.000125, 192, 96, 800),
    "scale12": (0.0000833333, 288, 144, 1200),
}

TEMPLATE = """#title: GPU throughput ladder {tag}
#domain: {lx} {ly} {lz}
#dx_dy_dz: {dx} {dx} {dx}
#time_window: 10e-9

#waveform: gaussian 1 10e9 my_wave
#hertzian_dipole: z {sx} {sy} {sz} my_wave
#rx: {rx} {ry} {rz}
"""

for tag, (dx, nx, ny, nz) in GRIDS.items():
    lx, ly, lz = dx * nx, dx * ny, dx * nz
    sx, sy, sz = lx * 0.5, ly * 0.5, lz * 0.15
    rx, ry, rz = lx * 0.5, ly * 0.5, lz * 0.7
    text = TEMPLATE.format(tag=tag, lx=lx, ly=ly, lz=lz, dx=dx,
                           sx=sx, sy=sy, sz=sz, rx=rx, ry=ry, rz=rz)
    path = os.path.join(OUTDIR, f"{tag}.in")
    with open(path, "w") as fh:
        fh.write(text)
    print("wrote", path)
