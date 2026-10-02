#!/usr/bin/env python3
"""
Simulated inputs in the formats the updated MATLAB steps write, for testing
the Python statistics and figures without the real data:

  <base>/signals/hmmmar_simple_k<K>_hmmpca15/fit_id.txt
  <base>/signals/hmm_{switching_rates,maxFO,dwell_times}_all_K_all_lowrank.csv   (run4d)
  <base>/signals/hmmmar_simple_k<K>_hmmpca15/hmm_raw_posthoc_graph_metrics/
        graph_metrics_per_subject_per_state_density15.csv                      (run5)
  <base>/signals/hmmmar_simple_k<K>_hmmpca15/hmm_bootstrap_stability/
        bootstrap_stability_subject_state.csv                                  (run5B)

and the same run4d data in the previous formats, for the previous scripts:
  <base>/signals/{hmm,pcahmm}_{...}_all_K_all_PCA.csv with pca_components = 50

    python tests/make_mock_inputs.py <base> [K]      (K = 3 by default; the
    previous scripts were written for K = 3)
"""
import os
import sys

import numpy as np
import pandas as pd

GROUPS = {"NTG_Y": 19, "NTG_M": 9, "NTG_A": 25, "CRND8_Y": 13, "CRND8_M": 5,
          "5XFAD_Y": 12, "5XFAD_M": 12}
T, TR, P = 570, 2.0, 82
FIT_ID = "a3f1" * 16

base = sys.argv[1] if len(sys.argv) > 1 else "mock"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 3
sig = os.path.join(base, "signals")
model = os.path.join(sig, f"hmmmar_simple_k{K}_hmmpca15")
for d in (os.path.join(model, "hmm_raw_posthoc_graph_metrics"),
          os.path.join(model, "hmm_bootstrap_stability")):
    os.makedirs(d, exist_ok=True)
with open(os.path.join(model, "fit_id.txt"), "w") as f:
    f.write(FIT_ID + "\n")

rng = np.random.default_rng(2026)
subjects = []
sid = 0
for g, n in GROUPS.items():
    for _ in range(n):
        sid += 1
        subjects.append((sid, g))

sw, mf, dw, gm, bs = [], [], [], [], []
for sid, g in subjects:
    shift = 0.6 if g.startswith("5XFAD") else 0.0
    fo = rng.dirichlet(np.linspace(3.0, 1.0, K) + shift)   # K = 3: [3, 2, 1]
    scale = 3.0 if g == "CRND8_M" else 1.0
    sw.append(dict(subject_id=sid, group=g, K_requested=K, K_returned=K,
                   switching_rate=abs(rng.normal(0.08, 0.02 * scale)), fit_id=FIT_ID))
    mf.append(dict(subject_id=sid, group=g, K_requested=K, K_returned=K,
                   max_FO=fo.max(), fit_id=FIT_ID))
    n_trans = 0
    for k in range(K):
        count = int(round(fo[k] * T))
        episodes = 0 if count < 3 else int(rng.integers(1, max(2, count // 8) + 1))
        mean_dwell = count / episodes if episodes else np.nan
        n_trans += episodes
        dw.append(dict(subject_id=sid, group=g, K_requested=K, K_returned=K, state=k + 1,
                       fractional_occupancy=fo[k], map_TR_count=count,
                       n_dwell_episodes=episodes, mean_dwell_TR=mean_dwell,
                       median_dwell_TR=mean_dwell * 0.8 if episodes else np.nan,
                       mean_dwell_seconds=mean_dwell * TR if episodes else np.nan,
                       median_dwell_seconds=mean_dwell * 0.8 * TR if episodes else np.nan,
                       state_path="group_viterbi", fit_id=FIT_ID))
        neff = fo[k] * T * rng.uniform(0.5, 0.9)
        ok = neff >= 83
        val = (lambda m, s: rng.normal(m + shift * 0.1, s * scale)) if ok else (lambda m, s: np.nan)
        gm.append(dict(subject_id=sid, state=k + 1, group=g, FO=fo[k], effective_n=neff,
                       map_TR_count=count, strength_mean=val(9.0, 1.0),
                       transitivity_bin=val(0.45, 0.04), Eglob_bin=val(0.55, 0.03),
                       fit_id=FIT_ID, min_effective_n=83))
        boot_ok = ok and rng.uniform() > 0.1
        jac = rng.uniform(0.4, 0.8) if ok else np.nan
        bs.append(dict(
            subject_id=sid, state=k + 1, group=g, n_timepoints=T, block_length=9, FO=fo[k],
            map_count=count, n_visits=episodes, mean_dwell_TR=mean_dwell,
            subject_n_transitions=0, subject_max_FO=fo.max(), near_static=int(fo[k] >= 0.85),
            Neff=neff, posterior_mass=fo[k] * T, p_over_Neff=P / neff, passes_Neff=int(ok),
            covariance_rank=P, covariance_effective_rank=rng.uniform(15, 30),
            valid_original_network=1, original_positive_edges=498,
            original_graph_density=0.15,
            split_half_Neff_1=neff / 2, split_half_Neff_2=neff / 2,
            split_half_valid=int(ok and neff > 166),
            split_half_edge_jaccard=jac - 0.15 if ok and neff > 166 else np.nan,
            split_half_nodal_rank_correlation=0.5 if ok and neff > 166 else np.nan,
            bootstrap_success_count=9500 if boot_ok else (5000 if ok else 0),
            bootstrap_success_fraction=0.95 if boot_ok else (0.5 if ok else 0.0),
            n_fail_boot_Neff=300 if ok else 0, n_fail_boot_network=0,
            median_boot_Neff=neff, median_edge_jaccard=jac, mean_edge_jaccard=jac,
            edge_jaccard_CI_low=jac - 0.1 if ok else np.nan,
            edge_jaccard_CI_high=jac + 0.1 if ok else np.nan,
            median_nodal_strength_rank_correlation=jac + 0.1 if ok else np.nan,
            mean_nodal_strength_rank_correlation=jac + 0.1 if ok else np.nan,
            nodal_rank_correlation_CI_low=jac if ok else np.nan,
            nodal_rank_correlation_CI_high=jac + 0.2 if ok else np.nan,
            n_unfiltered=9800 if ok else 0,
            median_edge_jaccard_unfiltered=jac - 0.02 if ok else np.nan,
            median_nodal_rank_unfiltered=jac + 0.08 if ok else np.nan,
            n_resample_pairs=500 if ok else 0,
            median_edge_jaccard_rr=jac - 0.1 if ok else np.nan,
            edge_jaccard_rr_CI_low=jac - 0.2 if ok else np.nan,
            edge_jaccard_rr_CI_high=jac if ok else np.nan,
            median_nodal_rank_rr=jac if ok else np.nan,
            nodal_rank_rr_CI_low=jac - 0.1 if ok else np.nan,
            nodal_rank_rr_CI_high=jac + 0.1 if ok else np.nan,
            analyzable=int(boot_ok), fit_id=FIT_ID))
    for row in bs[-K:]:
        row["subject_n_transitions"] = n_trans

pd.DataFrame(gm).to_csv(os.path.join(model, "hmm_raw_posthoc_graph_metrics",
                                     "graph_metrics_per_subject_per_state_density15.csv"),
                        index=False)
pd.DataFrame(bs).to_csv(os.path.join(model, "hmm_bootstrap_stability",
                                     "bootstrap_stability_subject_state.csv"), index=False)

for kind, rows in (("switching_rates", sw), ("maxFO", mf), ("dwell_times", dw)):
    d = pd.DataFrame(rows)
    d.assign(lowrank_components=15, model_type="lowrank").to_csv(
        os.path.join(sig, f"hmm_{kind}_all_K_all_lowrank.csv"), index=False)
    old = d.drop(columns=["fit_id"]).assign(pca_components=50)      # previous format
    for prefix in ("hmm", "pcahmm"):
        old.to_csv(os.path.join(sig, f"{prefix}_{kind}_all_K_all_PCA.csv"), index=False)
print(f"Mock inputs written under {base} (fit {FIT_ID[:12]})")
