#!/usr/bin/env python3
"""
End-to-end test of the BrainNet .node export in
full_analysis_abeta_mice_permutation_transitivity.py, on simulated inputs (no
mouse data).

    python tests/test_brainnet_nodes_end_to_end.py [work_dir]      # from static_v2/

Builds a static/ folder with simulated inputs: 95 subjects in the study's 7
groups, the study's 82 ROI names, and stronger coupling among 10 ROIs in the
CRND8 groups so that NBS finds components. It runs the current script and the
previous one (tests/legacy/full_analysis_abeta_mice_permutation_transitivity_before_nodes.py)
on it, both with permutations reduced (test copies only: 200, and 400 for NBS
so that the Bonferroni-adjusted component p can fall below 0.05), and checks:

  E1  group maps: every BrainNet_<group>_density_0.15.edge has a .node that,
      read as BrainNet reads it, has 82 nodes with label '-', the coordinates
      of brainnet_node_coords_highres.csv, size = row sums of the .edge matrix
      and colour = size rounded
  E2  NBS: every component's and every all-significant t-statistic .edge has
      a .node with size = row sums of |t| (0 outside the component)
  E3  every .edge file has its .node, and the manifest describes them
  E4  every other output (CSV tables, .edge files) is byte-identical to the
      previous script's; the only new files are .node files and trend files
  E5  roi_names_order.csv in another order stops the script before the analysis
  E6  trend maps: a third run with NBS_ALPHA = 0 and NBS_TREND_ALPHA = 1.01 (every
      component a trend; test copy only) writes *_trend_tstats.edge,
      *_trend_binary.edge and *_trend.node for every component, no
      all-significant file, *_trend circular plots only, and the manifest line

The script's own modules (welch_tests_scipy.py, nbs_published.py,
bctpy_nbs_welch.py, brainnet_nodes.py) are imported from static_v2/.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
import brainnet_nodes as bn          # noqa: E402

SCRIPT = "full_analysis_abeta_mice_permutation_transitivity.py"
PREVIOUS = os.path.join(HERE, "legacy", SCRIPT.replace(".py", "_before_nodes.py"))
GROUPS = {"NTG_Y": 19, "NTG_M": 9, "NTG_A": 25, "CRND8_Y": 13, "CRND8_M": 5,
          "5XFAD_Y": 12, "5XFAD_M": 12}
results = []


def check(name, ok):
    results.append(("PASS" if ok else "FAIL", name))


def brainnet_read(path):
    """BrainNet_LoadFiles.m: textscan(fid,'%f %f %f %f %f %s') over the stream."""
    tokens = open(path).read().split()
    rows = [tokens[i:i + 6] for i in range(0, len(tokens), 6)]
    return np.array([[float(v) for v in r[:5]] for r in rows]), [r[-1] for r in rows]


def make_inputs(folder, names):
    rng = np.random.default_rng(7)
    n_roi = len(names)
    os.makedirs(folder, exist_ok=True)
    pd.DataFrame({"roi_index": np.arange(1, n_roi + 1), "roi_name": names}).to_csv(
        os.path.join(folder, "roi_names_order.csv"), index=False)
    labels = [g for g, n in GROUPS.items() for _ in range(n)]
    pd.DataFrame({"index": np.arange(1, len(labels) + 1), "genotype": labels}).to_csv(
        os.path.join(folder, "subject_group_order_mappings.csv"), index=False)
    loadings = rng.normal(0, 0.6, (n_roi, 6))
    header = ",".join(str(i) for i in range(1, n_roi + 1))
    lines = [header]
    for s, g in enumerate(labels):
        T = 150
        X = rng.normal(size=(T, 6)) @ loadings.T + rng.normal(size=(T, n_roi))
        if g.startswith("CRND8"):
            X[:, :10] += 1.2 * rng.normal(size=(T, 1))
        C = np.corrcoef(X, rowvar=False)
        C = (C + C.T) / 2
        np.fill_diagonal(C, 1.0)
        if s:
            lines.append(header)                      # repeated header, as in merged.csv
        lines += [",".join(f"{v:.10f}" for v in row) for row in C]
    open(os.path.join(folder, "merged.csv"), "w").write("\n".join(lines) + "\n")


def test_copy(src, dst, all_trend=False):
    text = open(src).read()
    text, n1 = re.subn(r"n_permutations=10000", "n_permutations=200", text)
    text, n2 = re.subn(r"NBS_PERMUTATIONS = 10000", "NBS_PERMUTATIONS = 400", text)
    assert n1 > 0 and n2 == 1, "permutation settings not found"
    if all_trend:
        text, n3 = re.subn(r"NBS_ALPHA = 0.05\n", "NBS_ALPHA = 0.0\n", text)
        text, n4 = re.subn(r"NBS_TREND_ALPHA = 0.07\n", "NBS_TREND_ALPHA = 1.01\n", text)
        assert n3 == 1 and n4 == 1, "NBS thresholds not found"
    open(dst, "w").write(text)


def run(script, folder):
    env = dict(os.environ, MPLBACKEND="Agg",
               PYTHONPATH=os.pathsep.join([p for p in (os.environ.get("TEST_EXTRA_PATH"), PKG) if p]))
    return subprocess.run([sys.executable, script], cwd=folder, env=env,
                          capture_output=True, text=True)


work = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="static_nodes_")
shutil.rmtree(work, ignore_errors=True)
os.makedirs(work)
coords = pd.read_csv(bn.DEFAULT_COORDS)
names = coords.roi_name.tolist()
xyz = coords[["x", "y", "z"]].to_numpy(float)

runs = {}
for label, src, all_trend in [("new", os.path.join(PKG, SCRIPT), False), ("previous", PREVIOUS, False),
                              ("all_trend", os.path.join(PKG, SCRIPT), True)]:
    folder = os.path.join(work, label)
    make_inputs(folder, names)
    test_copy(src, os.path.join(work, f"{label}_{SCRIPT}"), all_trend)
    p = run(os.path.join(work, f"{label}_{SCRIPT}"), folder)
    open(os.path.join(work, f"{label}.log"), "w").write(p.stdout + p.stderr)
    if p.returncode != 0:
        print(p.stdout[-3000:], p.stderr[-3000:])
        raise SystemExit(f"{label} script failed (log in {work})")
    runs[label] = os.path.join(folder, "fc_analysis_mice")
out = runs["new"]
files = sorted(os.listdir(out))

# E1
ok, n_group = True, 0
for g in GROUPS:
    stem = os.path.join(out, f"BrainNet_{g}_density_0.15")
    if not os.path.isfile(stem + ".node"):
        ok = False
        continue
    n_group += 1
    sphere, lab = brainnet_read(stem + ".node")
    strength = np.loadtxt(stem + ".edge").sum(axis=1)
    ok &= (sphere.shape == (82, 5) and all(x == "-" for x in lab) and np.allclose(sphere[:, :3], xyz)
           and np.allclose(sphere[:, 4], strength, rtol=1e-5, atol=1e-6)
           and np.array_equal(sphere[:, 3], bn.matlab_round(sphere[:, 4])))
check(f"E1 group maps: {n_group} of 7 .node files; coordinates, size = .edge row sums, colour", ok
      and n_group == 7)

# E2
t_edges = [f for f in files if f.startswith("NBS_") and f.endswith("_tstats.edge")]
ok = len(t_edges) > 0
for f in t_edges:
    node = os.path.join(out, f.replace("_tstats.edge", ".node"))
    if not os.path.isfile(node):
        ok = False
        continue
    sphere, lab = brainnet_read(node)
    T = np.loadtxt(os.path.join(out, f))
    ok &= (sphere.shape == (82, 5) and np.allclose(sphere[:, :3], xyz)
           and np.allclose(sphere[:, 4], np.abs(T).sum(axis=1), rtol=1e-5, atol=1e-6)
           and bool((sphere[np.abs(T).sum(axis=1) == 0, 4] == 0).all()))
check(f"E2 NBS: {len(t_edges)} t-statistic .edge files, each with a .node of summed |t|", ok)

# E3
edge_stems = {re.sub(r"_(tstats|binary)$", "", f[:-5]) for f in files if f.endswith(".edge")}
node_stems = {f[:-5] for f in files if f.endswith(".node")}
manifest = open(os.path.join(out, "analysis_manifest.txt")).read()
check(f"E3 every .edge network has its .node ({len(node_stems)}), manifest line present",
      edge_stems == node_stems and "BrainNet .node files" in manifest)

# E4
prev = sorted(os.listdir(runs["previous"]))
compare = [f for f in prev if f.endswith((".csv", ".edge"))]
same = all(open(os.path.join(out, f), "rb").read() == open(os.path.join(runs["previous"], f), "rb").read()
           for f in compare)
new_files = set(files) - set(prev)
check(f"E4 {len(compare)} CSV and .edge outputs identical to the previous script's; "
      f"new files are only .node or trend files ({len(new_files)})",
      same and all(f.endswith(".node") or "_trend" in f for f in new_files))

# E5
folder = os.path.join(work, "wrong_order")
swapped = names.copy()
swapped[0], swapped[41] = swapped[41], swapped[0]
make_inputs(folder, swapped)
p = run(os.path.join(work, f"new_{SCRIPT}"), folder)
check("E5 ROI names in another order: stops before the analysis",
      p.returncode != 0 and "ROI names differ" in p.stderr
      and not os.path.isfile(os.path.join(folder, "fc_analysis_mice", "edgewise_omnibus_ANOVA.csv")))

# E6
tout = runs["all_trend"]
tfiles = sorted(os.listdir(tout))
summary = pd.read_csv(os.path.join(tout, "NBS_component_summary.csv"))
comps = summary[summary.component > 0]
ok = len(comps) > 0 and not any("all_significant" in f for f in tfiles)
for r in comps.itertuples():
    stem = f"NBS_{r.comparison.replace(' vs ', '_vs_')}_component_{r.component}_trend"
    ok &= all(os.path.isfile(os.path.join(tout, stem + x)) for x in ("_tstats.edge", "_binary.edge", ".node"))
    if ok:
        sphere, _ = brainnet_read(os.path.join(tout, stem + ".node"))
        T = np.loadtxt(os.path.join(tout, stem + "_tstats.edge"))
        ok &= np.allclose(sphere[:, 4], np.abs(T).sum(axis=1), rtol=1e-5, atol=1e-6)
circ = [f for f in tfiles if f.startswith("figure10_NBS_circular_")]
ok &= len(circ) > 0 and all(f.endswith("_trend.png") for f in circ)
ok &= "trend components" in open(os.path.join(tout, "analysis_manifest.txt")).read()
check(f"E6 trend maps: {len(comps)} components, each with *_trend .edge/.node files; "
      f"{len(circ)} circular plots, all *_trend; no all-significant files", ok)

for r, name in results:
    print(f"{r}  {name}")
n_fail = sum(r == "FAIL" for r, _ in results)
print(f"\n{len(results) - n_fail} passed, {n_fail} failed. Work folder: {work}")
sys.exit(1 if n_fail else 0)
