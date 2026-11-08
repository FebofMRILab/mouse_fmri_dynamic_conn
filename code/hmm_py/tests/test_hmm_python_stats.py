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
  S9  every script runs end to end for both reported models (primary:
      Gaussian HMM on 25 PCs, K = 3, PCA tables; secondary: HMM-PCA rank 15,
      K = 2) and reports their states
  S10 with no HMM_* settings, the scripts read the primary model (K = 3,
      options.pca = 25, hmm_v2/MODEL_PRIMARY.env) from
      /blue/febo/share/Abeta_Age/rabies/signals
  S11 fit-selection rule (inference_scope.csv; K = 4, rank 15, a model that is
      not reported, so the scope mechanism alone is tested): only the listed states and
      whole-fit outcomes are tested; their raw p equal the unrestricted run's;
      Holm is over the listed states only; descriptive tables keep every state
  S12 an inference_scope.csv from another fit is refused by every script
  S13 with nothing to test, every script finishes without tests, and no
      table or figure with tests from the earlier run is left behind
  S14 either reported model without inference_scope.csv is refused by every
      script unless HMM_ALLOW_NO_SCOPE=1
  S15 K = 2 (secondary): occupancy is one test (FO of state 2 = 1 - FO of
      state 1; E50). The test of state 2 computed on its own equals state 1's;
      the table runs it once (state 2 row: same_test_as = 1, same p), Holm
      counts it once (p_Holm = p), the planned contrasts of state 2 have t = -t
      of state 1 and state 1's q, BH pools the state 1 contrasts only, and both
      figure panels show the one p
  S16 one tested state (secondary model, scope state 1 only): all four
      scripts finish; the graph script's state-comparison table is written
      empty with its columns (it used to stop with KeyError 'metric'; E52)
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
NEW = {"HMM_OUT_BASE": sig, "HMM_K": "3", "HMM_LOWRANK": "15", "HMM_PCA": "0"}

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

# ---------------- S9: both reported models ----------------
scripts3 = ("hmm_dynamic_stats.py", "dynamic_fc_plotting_RtoPythonmigration.py",
            "brainnetwork_plotter_Rmigration_bin_transitivity.py",
            "hmm_bootstrap_stability_plotter.py")
workp = os.path.join(work, "primary")
run(os.path.join(HERE, "make_mock_inputs.py"), [workp, "3", "pca25"])
sigp = os.path.join(workp, "signals")
modelp = os.path.join(sigp, "hmmmar_simple_k3_pca25")
PRI = {"HMM_OUT_BASE": sigp, "HMM_K": "3", "HMM_LOWRANK": "0", "HMM_PCA": "25",
       "HMM_ALLOW_NO_SCOPE": "1"}
work2 = os.path.join(work, "secondary")
run(os.path.join(HERE, "make_mock_inputs.py"), [work2, "2"])
sig2 = os.path.join(work2, "signals")
model2 = os.path.join(sig2, "hmmmar_simple_k2_hmmpca15")
SEC = {"HMM_OUT_BASE": sig2, "HMM_K": "2", "HMM_LOWRANK": "15", "HMM_PCA": "0",
       "HMM_ALLOW_NO_SCOPE": "1"}
summ = []
ok9 = True
for env, sg, md, tag, want in ((PRI, sigp, modelp, "pca25_k3", [1, 2, 3]),
                               (SEC, sig2, model2, "hmmpca15_k2", [1, 2])):
    for sc in scripts3:
        run(os.path.join(PKG, sc), env=env)
    om = pd.read_csv(os.path.join(sg, f"hmm_dynamics_stats_{tag}", "omnibus_tests.csv"))
    gt = pd.read_csv(os.path.join(md, "hmm_graph_figures", "group_tests_by_state.csv"))
    st = (sorted(set(pd.to_numeric(om.state, errors="coerce").dropna().astype(int))),
          sorted(set(gt.state.astype(int))))
    outs_m = [os.path.join(sg, f"hmm_dynamics_figures_{tag}", "figure5_mean_dwell_by_group_state.png"),
              os.path.join(md, "hmm_graph_figures", "combined_graph_metrics_by_group_state.png"),
              os.path.join(md, "bootstrap_supplement", "FigS_bootstrap_attrition.png"),
              os.path.join(md, "bootstrap_supplement", "TableS_bootstrap_stability.docx")]
    n = sum(os.path.isfile(o) for o in outs_m)
    ok9 = ok9 and n == len(outs_m) and st[0] == want and st[1] == want
    summ.append(f"{tag}: states {st[0]}, outputs {n}/{len(outs_m)}")
check("S9 both reported models end to end (" + "; ".join(summ) + ")", float(ok9), ok9)

# ---------------- S10: defaults ----------------
clean = {k: v for k, v in os.environ.items() if not k.startswith("HMM_")}
p = subprocess.run([sys.executable, "-c",
                    "import hmm_model_selection as s; print(s.K, s.LOWRANK, s.PCA, s.TAG, s.MODEL_DIR, s.ROLE)"],
                   cwd=PKG, env=clean, capture_output=True, text=True)
expect = ("3 0 25 pca25_k3 "
          "/blue/febo/share/Abeta_Age/rabies/signals/hmmmar_simple_k3_pca25 primary")
check("S10 defaults: primary model, K = 3, pca 25, /blue/febo/share/Abeta_Age/rabies/signals",
      float(p.stdout.strip() == expect), p.stdout.strip() == expect)

# ---------------- K = 4 (not a reported model) for the scope tests ----------------
work4 = os.path.join(work, "k4")
run(os.path.join(HERE, "make_mock_inputs.py"), [work4, "4"])
sig4 = os.path.join(work4, "signals")
model4 = os.path.join(sig4, "hmmmar_simple_k4_hmmpca15")
K4 = {"HMM_OUT_BASE": sig4, "HMM_K": "4", "HMM_LOWRANK": "15", "HMM_PCA": "0"}
for sc in scripts3:
    run(os.path.join(PKG, sc), env=K4)
om4 = pd.read_csv(os.path.join(sig4, "hmm_dynamics_stats_hmmpca15_k4", "omnibus_tests.csv"))
gt4 = pd.read_csv(os.path.join(model4, "hmm_graph_figures", "group_tests_by_state.csv"))

# ---------------- S11-S13: fit-selection rule (inference_scope.csv) ----------------
from statsmodels.stats.multitest import multipletests  # noqa: E402

FIT4 = open(os.path.join(model4, "fit_id.txt")).read().strip()


def holm_finite(pv):
    pv = np.asarray(pv, float)
    out = np.full_like(pv, np.nan)
    ok = np.isfinite(pv)
    if ok.any():
        out[ok] = multipletests(pv[ok], method="holm")[1]
    return out


def write_scope(states, sw, mfo, fit_id=FIT4):
    rows = [("state", str(k), int(k in states)) for k in range(1, 5)]
    rows += [("whole_fit", "switching_rate", int(sw)), ("whole_fit", "max_FO", int(mfo))]
    pd.DataFrame([dict(kind=k, item=i, tested=t, n_pass=0, n_other_fits=9, r_min=0.9,
                       fit_id=fit_id, rule_sha256="0" * 64) for k, i, t in rows]).to_csv(
        os.path.join(model4, "inference_scope.csv"), index=False)


write_scope({2, 4}, True, False)
for s in scripts3:
    run(os.path.join(PKG, s), env=K4)
om_r = pd.read_csv(os.path.join(sig4, "hmm_dynamics_stats_hmmpca15_k4", "omnibus_tests.csv"))
pl_r = pd.read_csv(os.path.join(sig4, "hmm_dynamics_stats_hmmpca15_k4", "planned_contrasts.csv"))
ds_r = pd.read_csv(os.path.join(sig4, "hmm_dynamics_stats_hmmpca15_k4",
                                "dynamic_metrics_median_range_by_state.csv"))
gt_r = pd.read_csv(os.path.join(model4, "hmm_graph_figures", "group_tests_by_state.csv"))
fig_r = pd.read_csv(os.path.join(sig4, "hmm_dynamics_figures_hmmpca15_k4", "group_tests_shown_in_figures.csv"))
st_om = set(om_r.loc[om_r.state != "all", "state"].astype(int))
outc = set(om_r.loc[om_r.state == "all", "outcome"])
same_raw = om_r.loc[om_r.state != "all"].merge(om4, on=["outcome", "state"], suffixes=("", "_all"))
d_raw = float(np.nanmax(np.abs(same_raw.p_permutation - same_raw.p_permutation_all)))
fo = om_r[om_r.outcome == "fractional_occupancy"].sort_values("state")
d_holm = float(np.nanmax(np.abs(fo.p_Holm - holm_finite(fo.p_permutation))))
gt_m = gt_r.merge(gt4, on=["state", "metric"], suffixes=("", "_all"))
d_graw = float(np.nanmax(np.abs(gt_m.p_raw - gt_m.p_raw_all)))
d_gholm = float(np.nanmax(np.abs(gt_r.p_Holm - holm_finite(gt_r.p_raw))))
panels = set(fig_r.panel)
ok11 = (st_om == {2, 4} and outc == {"switching_rate"} and d_raw < 1e-12 and d_holm < 1e-12
        and set(gt_r.state.astype(int)) == {2, 4} and d_graw < 1e-12 and d_gholm < 1e-12
        and set(pl_r.state.astype(str)) == {"all", "2", "4"} and "max_FO" not in set(pl_r.outcome)
        and panels == {"all subjects", "State 2", "State 4"} and len(ds_r) == 4)
check(f"S11 scope states 2, 4 + switching rate: tests on {sorted(st_om)} / "
      f"{sorted(set(gt_r.state.astype(int)))}, raw p diff {max(d_raw, d_graw):.1g}, Holm diff",
      max(d_holm, d_gholm), ok11)

write_scope({2, 4}, True, False, fit_id="b" * 64)
codes = [run(os.path.join(PKG, s), env=K4, ok=False).returncode for s in scripts3]
check("S12 scope file from another fit refused (4 scripts)", sum(c != 0 for c in codes),
      all(c != 0 for c in codes))

write_scope({2, 4}, True, False)                      # leave outputs with tests behind
for s in scripts3:
    run(os.path.join(PKG, s), env=K4)
write_scope(set(), False, False)
outs = [run(os.path.join(PKG, s), env=K4, ok=False) for s in scripts3]
fig_n = pd.read_csv(os.path.join(sig4, "hmm_dynamics_figures_hmmpca15_k4", "group_tests_shown_in_figures.csv"))
stale = [os.path.join(sig4, "hmm_dynamics_stats_hmmpca15_k4", "omnibus_tests.csv"),
         os.path.join(sig4, "hmm_dynamics_stats_hmmpca15_k4", "planned_contrasts.csv"),
         os.path.join(model4, "hmm_graph_figures", "group_tests_by_state.csv"),
         os.path.join(sig4, "hmm_dynamics_figures_hmmpca15_k4", "figure3_fractional_occupancy_by_group_state.png"),
         os.path.join(model4, "bootstrap_supplement", "TableS_E.csv")]
n_stale = sum(os.path.isfile(f) for f in stale)
ok13 = (all(o.returncode == 0 for o in outs) and "Nothing to test" in outs[0].stdout
        and "No recurring state" in outs[2].stdout and len(fig_n) == 0 and n_stale == 0)
check("S13 nothing to test: scripts finish, no tests recorded, no stale test outputs",
      float(len(fig_n) + n_stale), ok13)
os.remove(os.path.join(model4, "inference_scope.csv"))

codes = []
for env in (PRI, SEC):
    strict = {k: v for k, v in env.items() if k != "HMM_ALLOW_NO_SCOPE"}
    codes += [run(os.path.join(PKG, sc), env=strict, ok=False).returncode for sc in scripts3]
check("S14 reported models without inference_scope.csv refused (4 scripts x 2 models)",
      sum(c != 0 for c in codes), all(c != 0 for c in codes))

# ---------------- S15: K = 2 occupancy is one test ----------------
sys.path.insert(0, PKG)
from welch_tests_scipy import welch_anova_permutation  # noqa: E402
import hmm_dynamic_stats as hds  # noqa: E402  (for GROUP_ORDER, N_PERMUTATIONS, seed)

dwell2 = pd.read_csv(os.path.join(sig2, "hmm_dwell_times_all_K_all_lowrank.csv"))
fo_w = dwell2.pivot_table(index="subject_id", columns="state", values="fractional_occupancy")
d_sum = float(np.max(np.abs(fo_w[1] + fo_w[2] - 1)))
grp = dwell2.drop_duplicates("subject_id").set_index("subject_id").group


def fo_samples(k):
    return [fo_w.loc[grp[grp == g].index, k].to_numpy(float) for g in hds.GROUP_ORDER]


n_perm = int(os.environ.get("HMM_N_PERM", 10000))
F1, p1 = welch_anova_permutation(fo_samples(1), n_permutations=n_perm, random_seed=hds.RANDOM_SEED)
F2, p2 = welch_anova_permutation(fo_samples(2), n_permutations=n_perm, random_seed=hds.RANDOM_SEED)
st2 = os.path.join(sig2, "hmm_dynamics_stats_hmmpca15_k2")
txt = {"state": str, "same_test_as": str}
om2 = pd.read_csv(os.path.join(st2, "omnibus_tests.csv"), dtype=txt)
pl2 = pd.read_csv(os.path.join(st2, "planned_contrasts.csv"), dtype=txt)
om2["same_test_as"] = om2.same_test_as.fillna("")
pl2["same_test_as"] = pl2.same_test_as.fillna("")
fo2 = om2[om2.outcome == "fractional_occupancy"].set_index("state")
r1, r2 = fo2.loc["1"], fo2.loc["2"]
pf = pl2[pl2.outcome == "fractional_occupancy"]
a = pf[pf.state == "1"].set_index("comparison")
b = pf[pf.state == "2"].set_index("comparison").loc[a.index]
own_fo_q = hds.bh(pl2.loc[(pl2.outcome == "fractional_occupancy") & (pl2.same_test_as == ""), "p_welch"])
fig2 = pd.read_csv(os.path.join(sig2, "hmm_dynamics_figures_hmmpca15_k2", "group_tests_shown_in_figures.csv"))
ff = fig2[fig2.outcome == "fractional_occupancy"].set_index("panel")
def same(x, y):          # p written to CSV and read back (last digit may differ)
    return bool(np.isclose(float(x), float(y), rtol=1e-12, atol=0))


ok15 = (d_sum < 1e-12 and abs(F1 - F2) < 1e-9 and p1 == p2
        and r1.same_test_as == "" and r2.same_test_as == "1"
        and same(r1.p_permutation, p1) and same(r2.p_permutation, p1)
        and same(r1.p_Holm, p1) and same(r2.p_Holm, p1)
        and float(np.max(np.abs(a.t + b.t))) < 1e-9 and np.allclose(a.q_BH_pooled, b.q_BH_pooled)
        and np.allclose(np.sort(a.q_BH_pooled.to_numpy()), np.sort(own_fo_q))
        and len(own_fo_q) == len(a)
        and same(ff.loc["State 1", "p_raw"], p1) and same(ff.loc["State 2", "p_raw"], p1)
        and same(ff.loc["State 1", "p_Holm_within_figure"], p1)
        and same(ff.loc["State 2", "p_Holm_within_figure"], p1)
        and "counted once" in ff.loc["State 2", "test"])
check(f"S15 K = 2 occupancy one test: own-test p {p1:.4g} = {p2:.4g}, Holm = p, t(2) = -t(1), "
      f"q shared, figure panels equal (FO sum dev)", d_sum, ok15)

# ---------------- S16: one tested state ----------------
FIT2 = open(os.path.join(model2, "fit_id.txt")).read().strip()
rows = [("state", "1", 1), ("state", "2", 0), ("whole_fit", "switching_rate", 1), ("whole_fit", "max_FO", 0)]
pd.DataFrame([dict(kind=k, item=i, tested=t, n_pass=0, n_other_fits=9, r_min=0.9, fit_id=FIT2,
                   rule_sha256="0" * 64) for k, i, t in rows]).to_csv(
    os.path.join(model2, "inference_scope.csv"), index=False)
SEC1 = {k: v for k, v in SEC.items() if k != "HMM_ALLOW_NO_SCOPE"}
codes = [run(os.path.join(PKG, sc), env=SEC1, ok=False).returncode for sc in scripts3]
sp = pd.read_csv(os.path.join(model2, "hmm_graph_figures", "posthoc_state_comparisons_by_group.csv"))
gp = pd.read_csv(os.path.join(model2, "hmm_graph_figures", "posthoc_group_comparisons_by_state.csv"))
ok16 = (all(c == 0 for c in codes) and len(sp) == 0 and {"metric", "comparison", "q_BH_pooled"} <= set(sp.columns)
        and set(gp.state.astype(int)) == {1})
check(f"S16 one tested state: 4 scripts exit {codes}; empty state-comparison table with columns",
      float(sum(codes)), ok16)
os.remove(os.path.join(model2, "inference_scope.csv"))

w_ = max(len(r[0]) for r in res)
print(f"\n{'Check':<{w_}}  {'Value':>10}  Result")
for name, v, r in res:
    print(f"{name:<{w_}}  {v:>10.4g}  {r}")
n_fail = sum(r[2] == "FAIL" for r in res)
print(f"\n{sum(r[2] == 'PASS' for r in res)} passed, {n_fail} failed. Work folder: {work}")
sys.exit(1 if n_fail else 0)
