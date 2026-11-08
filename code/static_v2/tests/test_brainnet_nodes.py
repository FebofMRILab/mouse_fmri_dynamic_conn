#!/usr/bin/env python3
"""
Tests of brainnet_nodes.py (BrainNet Viewer .node files for the static-FC maps).

    python tests/test_brainnet_nodes.py [lab_example.node]      # from static_v2/

  N1  coordinates: 82 ROIs, finite; loading refuses a different ROI order or count
  N2  a written file, read as BrainNet reads it (textscan '%f %f %f %f %f %s' over
      the whitespace-separated stream), gives 82 nodes with the coordinates,
      colour and size written and label '-'
  N3  colour = MATLAB round of the strength (halves away from zero)
  N4  (with a lab .node file as argument) writing that file's own sizes
      reproduces its x, y, z and size columns, and its colour column wherever
      the size is not printed as an exact half
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import brainnet_nodes as bn          # noqa: E402

results = []


def check(name, ok):
    results.append(("PASS" if ok else "FAIL", name))


def brainnet_read(path):
    """BrainNet_LoadFiles.m / BrainNet_MapCfg.m:
    data = textscan(fid,'%f %f %f %f %f %s'); whitespace (tabs, spaces, line
    breaks) separates the fields, and the 6-field format is applied in turn."""
    tokens = open(path).read().split()
    rows = [tokens[i:i + 6] for i in range(0, len(tokens), 6)]
    sphere = np.array([[float(v) for v in r[:5]] for r in rows])
    labels = [r[5] if len(r) == 6 else None for r in rows]
    return sphere, labels


table = pd.read_csv(bn.DEFAULT_COORDS)
names = table.roi_name.tolist()
xyz, _ = bn.load_node_coordinates(names)

# N1
ok = xyz.shape == (82, 3) and np.isfinite(xyz).all() and len(set(names)) == 82
swapped = names.copy()
swapped[0], swapped[1] = swapped[1], swapped[0]
for bad in (swapped, names[:-1]):
    try:
        bn.load_node_coordinates(bad)
        ok = False
    except ValueError:
        pass
check("N1 82 finite coordinates; a different ROI order or count is refused", ok)

# N2
rng = np.random.default_rng(1)
strength = np.r_[0.0, rng.uniform(0, 7, 81)]
with tempfile.TemporaryDirectory() as tmp:
    f = bn.write_brainnet_node(os.path.join(tmp, "t.node"), strength, xyz)
    sphere, labels = brainnet_read(f)
ok = (sphere.shape == (82, 5) and all(lab == "-" for lab in labels)
      and np.allclose(sphere[:, :3], xyz) and np.allclose(sphere[:, 4], strength, rtol=1e-5, atol=1e-6)
      and np.array_equal(sphere[:, 3], bn.matlab_round(strength)))
check("N2 read as BrainNet reads it: 82 nodes, coordinates, colour, size, label '-'", ok)

# N3
v = np.array([0.0, 0.49, 0.5, 1.5, 2.5, 5.5, 6.36, -0.5, -2.5])
check("N3 colour = MATLAB round (halves away from zero)",
      np.array_equal(bn.matlab_round(v), [0, 0, 1, 2, 3, 6, 6, -1, -3]))

# N4
if len(sys.argv) > 1:
    lab = pd.read_csv(sys.argv[1], sep=r"\s+", header=None)
    with tempfile.TemporaryDirectory() as tmp:
        f = bn.write_brainnet_node(os.path.join(tmp, "r.node"), lab[4].to_numpy(float), xyz)
        mine = pd.read_csv(f, sep="\t", header=None)
    half = np.isclose(lab[4] % 1, 0.5)
    ok = (np.allclose(mine[[0, 1, 2, 4]].to_numpy(float), lab[[0, 1, 2, 4]].to_numpy(float))
          and (mine[3][~half] == lab[3][~half]).all())
    check(f"N4 reproduces the lab file {os.path.basename(sys.argv[1])} "
          f"({int(half.sum())} size(s) printed as an exact half not compared for colour)", ok)

for r, name in results:
    print(f"{r}  {name}")
n_fail = sum(r == "FAIL" for r, _ in results)
print(f"\n{len(results) - n_fail} passed, {n_fail} failed")
sys.exit(1 if n_fail else 0)
