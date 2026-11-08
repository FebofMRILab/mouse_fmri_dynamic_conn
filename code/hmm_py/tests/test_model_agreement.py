#!/usr/bin/env python3
"""
Test of hmm_model_agreement.py (MODEL_SELECTION_RULE.md amendment 2) on
simulated outputs of both reported models whose answer is known.

    python tests/test_model_agreement.py [work_dir]      # from the hmm_py folder

Setup: mock run4d/run5 tables for the primary (Gaussian PCA-25, K = 3) and the
secondary (HMM-PCA 15, K = 2) models (tests/make_mock_inputs.py), the real
statistics scripts run on them under inference_scope.csv files, state FC built
so that primary states 1 and 3 resemble secondary state 1 and primary state 2
resembles secondary state 2.
  primary tests states 1, 2, 3, switching rate and max FO;
  secondary tests state 1 and switching rate only.

  A1  pairing: 1 -> 1, 2 -> 2, 3 -> 1; every r equals numpy's corrcoef
  A2  every classified row equals an independent classification of the two
      models' planned_contrasts.csv / posthoc_group_comparisons_by_state.csv
      rows (same contrast, paired state, q < 0.05, sign of t)
  A3  primary state 2 (partner not tested): every row "not assessable, no
      tested counterpart"; max FO (not tested in the secondary): every row
      "not assessable"; switching rate classified
  A4  outcomes and contrasts covered: 5 per-state outcomes x 2 assessable
      primary states x 8 contrasts, + 2 whole-fit outcomes x 8
  A5  classify(): robust, opposite, primary only, secondary only, neither,
      not assessable on constructed values
  A6  a table from another fit is refused, and the pairing was written before
      any group table was read (state_pairing.csv present, no contrast table)
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
sys.path.insert(0, PKG)
import hmm_model_agreement as agr          # noqa: E402

work = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="hmm_agree_")
shutil.rmtree(work, ignore_errors=True)
os.makedirs(work)
root = os.path.join(work, "signals_root")
res = []
FIT = "a3f1" * 16


def check(name, value, ok):
    res.append((name, value, "PASS" if ok else "FAIL"))


def run(script, args=(), env=None, ok=True):
    e = {k: v for k, v in os.environ.items() if not k.startswith("HMM_")}
    e.update(MPLBACKEND="Agg", **(env or {}))
    p = subprocess.run([sys.executable, script, *args], env=e, capture_output=True, text=True)
    if ok and p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-2000:])
        raise SystemExit(f"{os.path.basename(script)} failed")
    return p


def scope(mdir, states, K, sw, mfo):
    rows = [("state", str(k), int(k in states)) for k in range(1, K + 1)]
    rows += [("whole_fit", "switching_rate", int(sw)), ("whole_fit", "max_FO", int(mfo))]
    pd.DataFrame([dict(kind=k, item=i, tested=t, n_pass=0, n_other_fits=9, r_min=0.9,
                       fit_id=FIT, rule_sha256="0" * 64) for k, i, t in rows]).to_csv(
        os.path.join(mdir, "inference_scope.csv"), index=False)


rng = np.random.default_rng(5)
n_roi = 20
iu = np.triu_indices(n_roi, 1)
A_pat, B_pat = rng.normal(0.3, 0.2, len(iu[0])), rng.normal(0.3, 0.2, len(iu[0]))
fc = {"primary": [A_pat + rng.normal(0, 0.05, A_pat.size), B_pat + rng.normal(0, 0.05, A_pat.size),
                  0.7 * A_pat + 0.3 * B_pat + rng.normal(0, 0.05, A_pat.size)],
      "secondary": [A_pat, B_pat]}
spec = {"primary": dict(K=3, model="pca25", folder="hmmmar_simple_k3_pca25", tag="pca25_k3",
                        fits="hmm_k3_pca25_fits", env={"HMM_K": "3", "HMM_LOWRANK": "0", "HMM_PCA": "25"},
                        states={1, 2, 3}, sw=True, mfo=True),
        "secondary": dict(K=2, model="lowrank15", folder="hmmmar_simple_k2_hmmpca15", tag="hmmpca15_k2",
                          fits="hmm_k2_hmmpca15_fits", env={"HMM_K": "2", "HMM_LOWRANK": "15", "HMM_PCA": "0"},
                          states={1}, sw=True, mfo=False)}
for role, s in spec.items():
    base_dir = os.path.join(work, role)
    run(os.path.join(HERE, "make_mock_inputs.py"), [base_dir, str(s["K"]), s["model"]])
    base = os.path.join(base_dir, "signals")                      # plays the chosen fit_NN/
    mdir = os.path.join(base, s["folder"])
    scope(mdir, s["states"], s["K"], s["sw"], s["mfo"])
    for sc in ("hmm_dynamic_stats.py", "brainnetwork_plotter_Rmigration_bin_transitivity.py"):
        run(os.path.join(PKG, sc), env={"HMM_OUT_BASE": base, "HMM_N_PERM": "2000", **s["env"]})
    os.makedirs(os.path.join(mdir, "state_descriptors"), exist_ok=True)
    d = pd.DataFrame({"roi_i": iu[0] + 1, "roi_j": iu[1] + 1})
    for k, v in enumerate(fc[role], start=1):
        d[f"state_{k}"] = v
    d["fit_id"] = FIT
    d.to_csv(os.path.join(mdir, "state_descriptors", "state_fc_pooled_upper.csv"), index=False)
    sdir = os.path.join(root, s["fits"], "selection")
    os.makedirs(sdir)
    with open(os.path.join(sdir, "selected_fit.txt"), "w") as f:
        f.write(f"reported_fit: fit_04\nselected_base: {base}\nmodel_dir: {mdir}\nfit_id: {FIT}\n")
    s["base"], s["mdir"] = base, mdir

out = os.path.join(work, "agreement")
run(os.path.join(PKG, "hmm_model_agreement.py"), ["--signals", root, "--out_dir", out])
pair = pd.read_csv(os.path.join(out, "state_pairing.csv"))
A = pd.read_csv(os.path.join(out, "agreement_contrasts.csv"), dtype={"primary_state": str,
                                                                     "secondary_state": str})

# A1
rr = [[float(np.corrcoef(fc["primary"][i], fc["secondary"][j])[0, 1]) for j in range(2)] for i in range(3)]
d_r = max(abs(pair.loc[i, f"r_secondary_state_{j + 1}"] - rr[i][j]) for i in range(3) for j in range(2))
ok1 = list(pair.secondary_state) == [1, 2, 1] and d_r < 1e-12
check(f"A1 pairing {list(pair.primary_state)} -> {list(pair.secondary_state)}; r vs numpy (max diff)",
      d_r, ok1)


# A2: independent classification
def tables(s):
    dyn = pd.read_csv(os.path.join(s["base"], f"hmm_dynamics_stats_{s['tag']}", "planned_contrasts.csv"),
                      dtype={"state": str})
    gr = pd.read_csv(os.path.join(s["mdir"], "hmm_graph_figures", "posthoc_group_comparisons_by_state.csv"),
                     dtype={"state": str}).rename(columns={"metric": "outcome", "t_Welch": "t"})
    return pd.concat([dyn[["outcome", "state", "comparison", "t", "q_BH_pooled"]],
                      gr[["outcome", "state", "comparison", "t", "q_BH_pooled"]]])


TP, TS = tables(spec["primary"]), tables(spec["secondary"])
mism, n_cl = 0, 0
for _, r in A[A["class"] != "not assessable"].iterrows():
    a = TP[(TP.outcome == r.outcome) & (TP.state == r.primary_state) & (TP.comparison == r.comparison)].iloc[0]
    b = TS[(TS.outcome == r.outcome) & (TS.state == r.secondary_state) & (TS.comparison == r.comparison)].iloc[0]
    sa, sb = a.q_BH_pooled < 0.05, b.q_BH_pooled < 0.05
    want = (("robust" if np.sign(a.t) == np.sign(b.t) else "opposite") if sa and sb else
            "primary only" if sa else "secondary only" if sb else "neither")
    mism += want != r["class"] or abs(a.t - r.t_primary) > 1e-12 or abs(b.t - r.t_secondary) > 1e-12
    n_cl += 1
check(f"A2 independent classification of {n_cl} rows (mismatches); classes seen "
      f"{sorted(A['class'].unique())}", mism, mism == 0 and n_cl > 0)

# A3
p2 = A[A.primary_state == "2"]
mf = A[A.outcome == "max_FO"]
sw = A[A.outcome == "switching_rate"]
ok3 = (len(p2) > 0 and (p2["class"] == "not assessable").all()
       and p2.reason.str.startswith("no tested counterpart").all()
       and len(mf) == 8 and (mf["class"] == "not assessable").all()
       and len(sw) == 8 and (sw["class"] != "not assessable").all())
check("A3 no tested counterpart (primary 2), max FO not assessable, switching rate classified",
      float(ok3), ok3)

# A4
n_expect = (2 + 3) * 3 * 8 + 2 * 8        # every primary state is tested: 3 x 8 per outcome
ok4 = len(A) == n_expect and set(A.outcome) == {"switching_rate", "max_FO", "fractional_occupancy",
                                                "dwell_time_log10", "strength_mean",
                                                "transitivity_bin", "Eglob_bin"}
check(f"A4 rows {len(A)} (expected {n_expect}: 5 state outcomes x 3 states x 8 + 2 x 8)",
      len(A), ok4)

# A5
cases = [((2.0, 0.01, 3.0, 0.02), "robust"), ((2.0, 0.01, -3.0, 0.02), "opposite"),
         ((2.0, 0.01, 3.0, 0.2), "primary only"), ((2.0, 0.3, -3.0, 0.02), "secondary only"),
         ((2.0, 0.3, 3.0, 0.2), "neither"), ((np.nan, 0.01, 3.0, 0.02), "not assessable")]
bad5 = sum(agr.classify(*v)[0] != want for v, want in cases)
check("A5 classify() on constructed values (mismatches of 6)", bad5, bad5 == 0)

# A6
pc = os.path.join(spec["secondary"]["base"], "hmm_dynamics_stats_hmmpca15_k2", "planned_contrasts.csv")
t = pd.read_csv(pc)
t["fit_id"] = "b" * 64
t.to_csv(pc, index=False)
out6 = os.path.join(work, "agreement_refused")
p = run(os.path.join(PKG, "hmm_model_agreement.py"), ["--signals", root, "--out_dir", out6], ok=False)
ok6 = (p.returncode != 0 and "is not from the reported fit" in (p.stdout + p.stderr)
       and os.path.isfile(os.path.join(out6, "state_pairing.csv"))
       and not os.path.isfile(os.path.join(out6, "agreement_contrasts.csv")))
check("A6 table from another fit refused; pairing already written, no contrast table", float(ok6), ok6)

w_ = max(len(r[0]) for r in res)
print(f"{'Check':<{w_}}  {'Value':>10}  Result")
for name, v, r in res:
    print(f"{name:<{w_}}  {v:>10.4g}  {r}")
n_fail = sum(r[2] == "FAIL" for r in res)
print(f"\n{sum(r[2] == 'PASS' for r in res)} passed, {n_fail} failed. Work folder: {work}")
sys.exit(1 if n_fail else 0)
