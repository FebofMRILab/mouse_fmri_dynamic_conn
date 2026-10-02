#!/usr/bin/env python3
"""
Updated vs previous HMM statistics and figure scripts, on simulated inputs in
the formats the updated MATLAB steps write (tests/make_mock_inputs.py).

    python tests/test_hmm_python_stats.py [work_dir]      # from the hmm_py folder

Must be equal (previous vs updated, same data)
  S1  dynamics stats: Welch F and df of every omnibus test
  S2  dynamics stats: Welch t of every planned contrast
  S3  dynamics stats: paired t of every state comparison, and its p where the
      sign-flip test is exact (2**n <= 10,000, i.e. groups of <= 13 mice)
  S4  dynamics stats: presence chi-square statistic
  S5  graph-metric script: Welch F and df, contrast t, paired t (and exact p)
  S6  figure titles equal the statistics tables (same test, same seed)
Must hold
  S7  every script refuses tables from a different fit
  S8  every script runs end to end on the new formats and writes its outputs
  S9  every script runs end to end for the reported model, K = 4, and reports
      four states
  S10 with no HMM_* settings, the scripts read the reported model (K = 4,
      rank 15) from /blue/febo/share/Abeta_Age/rabies/signals
Reported
  omnibus permutation p (both permutation tests; Monte Carlo difference only)
  contrast p: ordinary Welch t vs the previous permutation of Welch's t
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
LEG = os.path.join(HERE, "legacy")
work = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="hmm_py_test_")
shutil.rmtree(work, ignore_errors=True)
os.makedirs(work)
sig = os.path.join(work, "signals")
model = os.path.join(sig, "hmmmar_simple_k3_hmmpca15")
res = []


def run(script, args=(), env=None, cwd=None, ok=True):
    e = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=os.path.dirname(script))
    e.update(env or {})
    p = subprocess.run([sys.executable, script, *args], env=e, cwd=cwd or work,
                       capture_output=True, text=True)
    if ok and p.returncode != 0:
        print(p.stdout[-3000:], p.stderr[-3000:])
        raise SystemExit(f"{os.path.basename(script)} failed")
    return p


def check(name, value, passed):
    res.append((name, value, "PASS" if passed else "FAIL"))


def report(name, value):
    res.append((name, value, "report"))


def maxdiff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    both = np.isfinite(a) & np.isfinite(b)
    same_nan = np.array_equal(np.isfinite(a), np.isfinite(b))
    return (float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0), same_nan


run(os.path.join(HERE, "make_mock_inputs.py"), [work])
# the previous scripts were written for K = 3, so the comparisons S1-S8 use K = 3
NEW = {"HMM_OUT_BASE": sig, "HMM_K": "3"}

# ---------------- dynamics statistics ----------------
run(os.path.join(PKG, "hmm_dynamic_stats.py"), env=NEW)
run(os.path.join(LEG, "hmm_dynamic_stats_previous.py"), env={"HMM_DATA_DIR": sig})
new_dir = os.path.join(sig, "hmm_dynamics_stats_hmmpca15_k3")
old_dir = os.path.join(sig, "hmm_dynamics_stats_hmmpca50_k3")
on, oo = (pd.read_csv(os.path.join(d, "omnibus_tests.csv")) for d in (new_dir, old_dir))
key = ["outcome", "state"]
m = on.merge(oo, on=key, suffixes=("_new", "_old"))
w = m[m.test_new.str.startswith("Welch")]
dF, nanF = maxdiff(w.statistic_new, w.statistic_old)
d1, _ = maxdiff(w.df1_new, w.df1_old)
d2, _ = maxdiff(w.df2_Welch_new, w.df2_Welch_old)
check("S1 Welch F, updated vs previous (max diff)", dF, dF < 1e-10 and nanF)
check("S1 Welch df (statsmodels vs previous formula, max diff)", max(d1, d2), max(d1, d2) < 1e-9)
dp, _ = maxdiff(w.p_permutation_new, w.p_permutation_old)
report("omnibus permutation p, scipy vs previous loop (max |dp|)", dp)
pr = m[m.test_new.str.contains("chi")]
dc, nanc = maxdiff(pr.statistic_new, pr.statistic_old)
check("S4 presence chi-square statistic (max diff)", dc, dc < 1e-10 and nanc)

pn, po = (pd.read_csv(os.path.join(d, "planned_contrasts.csv")) for d in (new_dir, old_dir))
m = pn.merge(po, on=["outcome", "state", "comparison"], suffixes=("_new", "_old"))
dt, nant = maxdiff(m.t_new, m.t_old)
check("S2 contrast Welch t, updated vs previous (max diff)", dt, dt < 1e-10 and nant)
dpc = (m.p_welch - m.p_permutation).abs()
report("contrast p: ordinary Welch t vs permutation of Welch t (median |dp|)",
       float(dpc.median()))
report("contrast p: largest |dp|", float(dpc.max()))

sn, so = (pd.read_csv(os.path.join(d, "posthoc_state_comparisons_by_group.csv"))
          for d in (new_dir, old_dir))
m = sn.merge(so, on=["outcome", "group", "comparison"], suffixes=("_new", "_old"))
dt, nant = maxdiff(m.t_paired_new, m.t_paired_old)
exact = m.n_paired_animals_new <= 13
dpe, _ = maxdiff(m.p_permutation_new[exact], m.p_permutation_old[exact])
check("S3 paired t, updated vs previous (max diff)", dt, dt < 1e-10 and nant)
check(f"S3 exact sign-flip p (groups <= 13 mice; {int(exact.sum())} tests, max diff)",
      dpe, dpe < 1e-12)
dpm, _ = maxdiff(m.p_permutation_new[~exact], m.p_permutation_old[~exact])
report("sign-flip p, Monte Carlo groups (max |dp|)", dpm)

# ---------------- graph-metric figures and tests ----------------
gm_csv = os.path.join(model, "hmm_raw_posthoc_graph_metrics",
                      "graph_metrics_per_subject_per_state_density15.csv")
run(os.path.join(PKG, "brainnetwork_plotter_Rmigration_bin_transitivity.py"), env=NEW)
old_cwd = os.path.join(work, "old_graph")
os.makedirs(old_cwd)
shutil.copy(gm_csv, old_cwd)
run(os.path.join(LEG, "brainnetwork_plotter_Rmigration_bin_transitivity_previous.py"),
    cwd=old_cwd)
gn = os.path.join(model, "hmm_graph_figures")
go = os.path.join(old_cwd, "hmm_graph_figures")
a, b = (pd.read_csv(os.path.join(d, "group_tests_by_state.csv")) for d in (gn, go))
m = a.merge(b, on=["state", "metric"], suffixes=("_new", "_old"))
dF, nanF = maxdiff(m.F_new, m.F_old)
dd, _ = maxdiff(m.df2_Welch_new, m.df2_Welch_old)
check("S5 graph metrics: Welch F and df (max diff)", max(dF, dd), max(dF, dd) < 1e-9 and nanF)
a, b = (pd.read_csv(os.path.join(d, "posthoc_group_comparisons_by_state.csv")) for d in (gn, go))
m = a.merge(b, on=["metric", "state", "comparison"], suffixes=("_new", "_old"))
dt, nant = maxdiff(m.t_Welch_new, m.t_Welch_old)
check("S5 graph metrics: contrast Welch t (max diff)", dt, dt < 1e-10 and nant)
a, b = (pd.read_csv(os.path.join(d, "posthoc_state_comparisons_by_group.csv")) for d in (gn, go))
m = a.merge(b, on=["metric", "group", "comparison"], suffixes=("_new", "_old"))
dt, nant = maxdiff(m.t_paired_new, m.t_paired_old)
exact = m.n_paired_animals_new <= 13
dpe, _ = maxdiff(m.p_permutation_new[exact], m.p_permutation_old[exact])
check("S5 graph metrics: paired t, and exact sign-flip p (max diff)", max(dt, dpe),
      max(dt, dpe) < 1e-10 and nant)

# ---------------- dynamics figures: titles = tables ----------------
run(os.path.join(PKG, "dynamic_fc_plotting_RtoPythonmigration.py"), env=NEW)
fig_dir = os.path.join(sig, "hmm_dynamics_figures_hmmpca15_k3")
ft = pd.read_csv(os.path.join(fig_dir, "group_tests_shown_in_figures.csv"))
on["panel"] = on.state.astype(str)
pairs = [("switching_rate", "switching_rate", "all subjects", "all"),
         ("max_FO", "max_FO", "all subjects", "all")]
diffs = []
for outcome, stat_outcome, panel, state in pairs:
    p_fig = ft[(ft.outcome == outcome) & (ft.panel == panel)].p_raw.iloc[0]
    p_tab = on[(on.outcome == stat_outcome) & (on.state.astype(str) == state)].p_permutation.iloc[0]
    diffs.append(abs(p_fig - p_tab))
for outcome, stat_outcome in (("fractional_occupancy", "fractional_occupancy"),
                              ("mean_dwell_seconds", "dwell_time_log10")):
    f = ft[ft.outcome == outcome].drop_duplicates("panel")
    t = on[on.outcome == stat_outcome].sort_values("state")
    diffs.append(float(np.nanmax(np.abs(f.p_raw.to_numpy() - t.p_permutation.to_numpy()))))
check("S6 figure-title p equals the statistics table (max diff)", max(diffs), max(diffs) == 0)

# ---------------- bootstrap supplement ----------------
p = run(os.path.join(PKG, "hmm_bootstrap_stability_plotter.py"), env=NEW)
bdir = os.path.join(model, "bootstrap_supplement")
p_old = run(os.path.join(LEG, "hmm_bootstrap_stability_plotter_previous.py"),
            ["--csv", os.path.join(model, "hmm_bootstrap_stability",
                                   "bootstrap_stability_subject_state.csv"),
             "--out_dir", os.path.join(work, "old_boot")], ok=False)
report("previous bootstrap plotter on the updated run5B CSV: exit code "
       "(error on missing mean_dwell_volumes expected)", p_old.returncode)
outs = [os.path.join(model, "hmm_graph_figures", "combined_graph_metrics_by_group_state.png"),
        os.path.join(fig_dir, "figure5_mean_dwell_by_group_state.png"),
        os.path.join(bdir, "FigS_bootstrap_attrition.png"),
        os.path.join(bdir, "TableS_bootstrap_stability.docx")]
missing = [o for o in outs if not os.path.isfile(o)]
check(f"S8 all scripts ran; key outputs present ({len(outs) - len(missing)}/{len(outs)})",
      len(missing), not missing)

# ---------------- S7: a table from another fit is refused ----------------
dwell = os.path.join(sig, "hmm_dwell_times_all_K_all_lowrank.csv")
backup = dwell + ".bak"
shutil.copy(dwell, backup)
d = pd.read_csv(dwell)
d["fit_id"] = "b" * 64
d.to_csv(dwell, index=False)
r1 = run(os.path.join(PKG, "hmm_dynamic_stats.py"), env=NEW, ok=False)
r2 = run(os.path.join(PKG, "dynamic_fc_plotting_RtoPythonmigration.py"), env=NEW, ok=False)
shutil.move(backup, dwell)
g = pd.read_csv(gm_csv)
g.loc[0, "fit_id"] = "b" * 64
bad = os.path.join(work, "graph_mixed.csv")
g.to_csv(bad, index=False)
r3 = run(os.path.join(PKG, "brainnetwork_plotter_Rmigration_bin_transitivity.py"), [bad],
         env=NEW, ok=False)
codes = [r1.returncode, r2.returncode, r3.returncode]
check("S7 tables from another fit refused (stats, dynamics figures, graph figures)",
      sum(c != 0 for c in codes), all(c != 0 for c in codes))

# ---------------- S9: the reported model, K = 4 ----------------
work4 = os.path.join(work, "k4")
run(os.path.join(HERE, "make_mock_inputs.py"), [work4, "4"])
sig4 = os.path.join(work4, "signals")
model4 = os.path.join(sig4, "hmmmar_simple_k4_hmmpca15")
K4 = {"HMM_OUT_BASE": sig4, "HMM_K": "4"}
for s in ("hmm_dynamic_stats.py", "dynamic_fc_plotting_RtoPythonmigration.py",
          "brainnetwork_plotter_Rmigration_bin_transitivity.py",
          "hmm_bootstrap_stability_plotter.py"):
    run(os.path.join(PKG, s), env=K4)
om4 = pd.read_csv(os.path.join(sig4, "hmm_dynamics_stats_hmmpca15_k4", "omnibus_tests.csv"))
gt4 = pd.read_csv(os.path.join(model4, "hmm_graph_figures", "group_tests_by_state.csv"))
states4 = (sorted(set(pd.to_numeric(om4.state, errors="coerce").dropna().astype(int))),
           sorted(set(gt4.state.astype(int))))
outs4 = [os.path.join(sig4, "hmm_dynamics_figures_hmmpca15_k4", "figure5_mean_dwell_by_group_state.png"),
         os.path.join(model4, "hmm_graph_figures", "combined_graph_metrics_by_group_state.png"),
         os.path.join(model4, "bootstrap_supplement", "FigS_bootstrap_attrition.png"),
         os.path.join(model4, "bootstrap_supplement", "TableS_bootstrap_stability.docx")]
n4 = sum(os.path.isfile(o) for o in outs4)
check(f"S9 K = 4: all scripts ran, states {states4[0]} / {states4[1]}, outputs {n4}/{len(outs4)}",
      n4, n4 == len(outs4) and states4[0] == [1, 2, 3, 4] and states4[1] == [1, 2, 3, 4])

# ---------------- S10: defaults ----------------
clean = {k: v for k, v in os.environ.items() if not k.startswith("HMM_")}
p = subprocess.run([sys.executable, "-c",
                    "import hmm_model_selection as s; print(s.K, s.LOWRANK, s.PCA, s.TAG, s.MODEL_DIR)"],
                   cwd=PKG, env=clean, capture_output=True, text=True)
expect = ("4 15 0 hmmpca15_k4 "
          "/blue/febo/share/Abeta_Age/rabies/signals/hmmmar_simple_k4_hmmpca15")
check("S10 defaults: K = 4, rank 15, /blue/febo/share/Abeta_Age/rabies/signals",
      float(p.stdout.strip() == expect), p.stdout.strip() == expect)

w_ = max(len(r[0]) for r in res)
print(f"\n{'Check':<{w_}}  {'Value':>10}  Result")
for name, v, r in res:
    print(f"{name:<{w_}}  {v:>10.4g}  {r}")
n_fail = sum(r[2] == "FAIL" for r in res)
print(f"\n{sum(r[2] == 'PASS' for r in res)} passed, {n_fail} failed. Work folder: {work}")
sys.exit(1 if n_fail else 0)
