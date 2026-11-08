#!/usr/bin/env python3
"""
hmm_model_agreement.py
-------------------------------------------------------------------
Agreement between the two reported HMMs, applied mechanically as
hmm_null/MODEL_SELECTION_RULE.md amendment 2 (27 Oct 2026; provenance D35)
defines it. Run after both pipelines (run_selected_fit.sh primary and
secondary). No choice is made by hand.

  1  compared: the 8 planned Welch t contrasts of every outcome tested in both
     models (inference_scope.csv of each reported fit): switching rate and
     max FO (whole fit); fractional occupancy, dwell time (log10) and the
     state-network graph metrics (strength_mean, transitivity_bin, Eglob_bin)
     per state. State presence and within-group state comparisons are not
     classified.
  2  whole-fit outcomes: the same contrast in both models.
  3  state pairing: each primary state is paired with the secondary state whose
     pooled state FC (R.state_fc, upper triangle; state_fc_pooled_upper.csv from
     hmm_state_descriptors.m) has the highest Pearson r, among all secondary
     states. Computed and written (state_pairing.csv) before any group table is
     read. A primary state whose partner is not tested in the secondary model
     has no tested counterpart.
  4  per contrast, with each model's own BH q and the sign of its Welch t:
     robust (q < 0.05 in both, same sign), opposite (both, opposite signs),
     primary only, secondary only, neither, not assessable.
  5  K = 2 occupancy: the secondary state 2 rows carry state 1's test (sign
     reversed, state 1's q; E50); the paired state's own row is used.

Inputs (per model, from <signals>/<fits folder>/selection/selected_fit.txt):
  <model dir>/fit_id.txt, inference_scope.csv,
  <model dir>/state_descriptors/state_fc_pooled_upper.csv,
  <model dir>/hmm_graph_figures/posthoc_group_comparisons_by_state.csv,
  <selected base>/hmm_dynamics_stats_<tag>/planned_contrasts.csv
Every file must carry the reported fit's ID.

Outputs in <signals>/hmm_model_agreement/:
  state_pairing.csv, agreement_contrasts.csv, agreement_counts.csv,
  agreement_report.txt (with the rule's SHA-256 and both fit IDs)

Usage (from code/)
    python hmm_py/hmm_model_agreement.py
    python hmm_py/hmm_model_agreement.py --signals /path/to/signals
"""
import argparse
import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hmm_model_selection as sel                         # noqa: E402

ALPHA = 0.05
DYNAMIC_OUTCOMES = {"whole": ["switching_rate", "max_FO"],
                    "state": ["fractional_occupancy", "dwell_time_log10"]}
GRAPH_METRICS = ["strength_mean", "transitivity_bin", "Eglob_bin"]
CLASSES = ["robust", "opposite", "primary only", "secondary only", "neither", "not assessable"]


def model_names(k, r, p):
    if r > 0:
        folder, tag = f"hmmmar_simple_k{k}_hmmpca{r}", f"hmmpca{r}_k{k}"
    else:
        folder, tag = f"hmmmar_simple_k{k}_pca{p}", f"pca{p}_k{k}"
    return folder, tag, folder.replace("hmmmar_simple_", "hmm_") + "_fits"


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_fit_id_table(path, fit_id, **kw):
    t = pd.read_csv(path, **kw)
    ids = t["fit_id"].astype(str).str.strip().unique()
    if len(ids) != 1 or ids[0] != fit_id:
        raise SystemExit(f"{path} is not from the reported fit {fit_id[:12]}.")
    return t


def load_model(role, signals):
    k, r, p = sel.REPORTED_MODELS[role]
    folder, tag, fits = model_names(k, r, p)
    sf = Path(signals) / fits / "selection" / "selected_fit.txt"
    if not sf.is_file():
        raise SystemExit(f"No {sf}: run the {role} model's selection and pipeline first.")
    kv = dict(line.split(": ", 1) for line in sf.read_text().splitlines() if ": " in line)
    base, mdir, fit_id = Path(kv["selected_base"]), Path(kv["model_dir"]), kv["fit_id"].strip()
    if mdir.name != folder:
        raise SystemExit(f"{sf} names {mdir}, not a fit of the {role} model ({folder}).")
    fid_file = mdir / "fit_id.txt"
    if not fid_file.is_file() or fid_file.read_text().strip() != fit_id:
        raise SystemExit(f"{fid_file} does not match {sf}.")
    scope = read_fit_id_table(mdir / "inference_scope.csv", fit_id, dtype={"item": str})
    st = scope[scope["kind"] == "state"]
    whole = scope[scope["kind"] == "whole_fit"].set_index("item")["tested"]
    return dict(role=role, K=k, folder=folder, tag=tag, base=base, mdir=mdir, fit_id=fit_id,
                tested=sorted(int(x) for x in st.loc[st["tested"] == 1, "item"]),
                whole={o: bool(whole.get(o, 0) == 1) for o in DYNAMIC_OUTCOMES["whole"]},
                label=f"{role}: {folder} ({base.name}, fit {fit_id[:12]})")


def state_fc(m):
    t = read_fit_id_table(m["mdir"] / "state_descriptors" / "state_fc_pooled_upper.csv", m["fit_id"])
    cols = [f"state_{k}" for k in range(1, m["K"] + 1)]
    missing = set(cols) - set(t.columns)
    if missing:
        raise SystemExit(f"state_fc_pooled_upper.csv of the {m['role']} model lacks {sorted(missing)}.")
    return t[["roi_i", "roi_j"] + cols]


def pairing(P, S):
    """Rule item 3: every primary state with the secondary state of highest r."""
    a, b = state_fc(P), state_fc(S)
    if not (a[["roi_i", "roi_j"]].to_numpy() == b[["roi_i", "roi_j"]].to_numpy()).all():
        raise SystemExit("The two models' state FC files do not list the same ROI pairs.")
    rows = []
    for kp in range(1, P["K"] + 1):
        x = a[f"state_{kp}"].to_numpy(float)
        rs = [float(np.corrcoef(x, b[f"state_{ks}"].to_numpy(float))[0, 1])
              for ks in range(1, S["K"] + 1)]
        best = int(np.argmax(rs)) + 1                  # ties: the lower state number
        row = dict(primary_state=kp, primary_tested=kp in P["tested"], secondary_state=best,
                   r=rs[best - 1], secondary_tested=best in S["tested"])
        row.update({f"r_secondary_state_{ks}": rs[ks - 1] for ks in range(1, S["K"] + 1)})
        rows.append(row)
    return pd.DataFrame(rows)


def contrast_tables(m):
    dyn = read_fit_id_table(m["base"] / f"hmm_dynamics_stats_{m['tag']}" / "planned_contrasts.csv",
                            m["fit_id"], dtype={"state": str})
    dyn = dyn.rename(columns={"outcome": "outcome"})[["outcome", "state", "comparison", "t", "q_BH_pooled"]]
    gr = read_fit_id_table(m["mdir"] / "hmm_graph_figures" / "posthoc_group_comparisons_by_state.csv",
                           m["fit_id"], dtype={"state": str})
    gr = gr.rename(columns={"metric": "outcome", "t_Welch": "t"})[
        ["outcome", "state", "comparison", "t", "q_BH_pooled"]]
    out = pd.concat([dyn, gr], ignore_index=True)
    out["state"] = out["state"].astype(str).str.strip()
    return out.set_index(["outcome", "state", "comparison"])


def classify(tp, qp, ts, qs):
    if not all(np.isfinite([tp, qp, ts, qs])):
        return "not assessable", "untestable in one model (no t or q)"
    sp, ss = qp < ALPHA, qs < ALPHA
    if sp and ss:
        return ("robust", "") if np.sign(tp) == np.sign(ts) else ("opposite", "")
    if sp:
        return "primary only", ""
    if ss:
        return "secondary only", ""
    return "neither", ""


def lookup(table, outcome, state, comparison):
    key = (outcome, str(state), comparison)
    if key not in table.index:
        return np.nan, np.nan
    r = table.loc[key]
    if isinstance(r, pd.DataFrame):
        raise SystemExit(f"Duplicate rows for {key}.")
    return float(r["t"]), float(r["q_BH_pooled"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--signals", default=os.environ.get("HMM_OUT_BASE",
                                                         "/blue/febo/share/Abeta_Age/rabies/signals"))
    ap.add_argument("--rule", default=str(Path(__file__).resolve().parent.parent / "hmm_null"
                                          / "MODEL_SELECTION_RULE.md"))
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args()
    out = Path(a.out_dir or Path(a.signals) / "hmm_model_agreement")
    sel.fresh_output_dir(out, ["*.csv", "agreement_report.txt"])

    P, S = load_model("primary", a.signals), load_model("secondary", a.signals)

    # ---- item 3, before any group table is read ----
    pair = pairing(P, S)
    pair.to_csv(out / "state_pairing.csv", index=False)

    # ---- items 1, 2, 4, 5 ----
    tp, ts = contrast_tables(P), contrast_tables(S)
    comps = sorted(set(c for _, _, c in tp.index) | set(c for _, _, c in ts.index))
    rows = []
    for outcome in DYNAMIC_OUTCOMES["whole"]:
        for c in comps:
            base = dict(outcome=outcome, primary_state="all", secondary_state="all", pair_r=np.nan,
                        comparison=c)
            if not (P["whole"][outcome] and S["whole"][outcome]):
                who = [m["role"] for m in (P, S) if not m["whole"][outcome]]
                rows.append({**base, "class": "not assessable",
                             "reason": f"not tested in the {' and '.join(who)} model (rule item 7)"})
                continue
            t1, q1 = lookup(tp, outcome, "all", c)
            t2, q2 = lookup(ts, outcome, "all", c)
            cl, why = classify(t1, q1, t2, q2)
            rows.append({**base, "t_primary": t1, "q_primary": q1, "t_secondary": t2,
                         "q_secondary": q2, "class": cl, "reason": why})
    for outcome in DYNAMIC_OUTCOMES["state"] + GRAPH_METRICS:
        for _, pr in pair[pair.primary_tested].iterrows():
            kp, ks = int(pr.primary_state), int(pr.secondary_state)
            for c in comps:
                base = dict(outcome=outcome, primary_state=str(kp), secondary_state=str(ks),
                            pair_r=pr.r, comparison=c)
                if not pr.secondary_tested:
                    rows.append({**base, "class": "not assessable",
                                 "reason": f"no tested counterpart (secondary state {ks} not tested)"})
                    continue
                t1, q1 = lookup(tp, outcome, kp, c)
                t2, q2 = lookup(ts, outcome, ks, c)
                cl, why = classify(t1, q1, t2, q2)
                rows.append({**base, "t_primary": t1, "q_primary": q1, "t_secondary": t2,
                             "q_secondary": q2, "class": cl, "reason": why})
    cols = ["outcome", "primary_state", "secondary_state", "pair_r", "comparison", "t_primary",
            "q_primary", "t_secondary", "q_secondary", "class", "reason"]
    A = pd.DataFrame(rows).reindex(columns=cols)
    A["fit_id_primary"], A["fit_id_secondary"] = P["fit_id"], S["fit_id"]
    A.to_csv(out / "agreement_contrasts.csv", index=False)
    counts = (A.groupby(["outcome", "class"]).size().unstack(fill_value=0)
              .reindex(columns=CLASSES, fill_value=0))
    counts.to_csv(out / "agreement_counts.csv")

    rule_hash = sha256(a.rule) if os.path.isfile(a.rule) else "rule file not found"
    lines = [f"Agreement between the reported HMMs (MODEL_SELECTION_RULE.md amendment 2), "
             f"{datetime.now().isoformat(timespec='seconds')}",
             f"rule      {a.rule}  sha256 {rule_hash}",
             f"primary   {P['label']}; tested states {P['tested'] or 'none'}; whole fit "
             f"{[o for o, v in P['whole'].items() if v] or 'none'}",
             f"secondary {S['label']}; tested states {S['tested'] or 'none'}; whole fit "
             f"{[o for o, v in S['whole'].items() if v] or 'none'}", "",
             "State pairing (highest pooled state-FC r; written before the group tables were read):"]
    for _, pr in pair.iterrows():
        lines.append(f"  primary state {pr.primary_state} -> secondary state {pr.secondary_state} "
                     f"(r = {pr.r:.3f}); primary tested: {pr.primary_tested}; secondary tested: "
                     f"{pr.secondary_tested}")
    lines += ["", "Contrasts by class:", counts.to_string(), ""]
    rob = A[A["class"] == "robust"]
    lines.append(f"ROBUST effects (q < 0.05 in both models, same sign): {len(rob)}")
    for _, r in rob.iterrows():
        lines.append(f"  {r.outcome}, primary state {r.primary_state} / secondary state "
                     f"{r.secondary_state}: {r.comparison} (t {r.t_primary:.2f} / {r.t_secondary:.2f}; "
                     f"q {r.q_primary:.3g} / {r.q_secondary:.3g})")
    opp = A[A["class"] == "opposite"]
    if len(opp):
        lines.append(f"OPPOSITE signs, both q < 0.05: {len(opp)} (see agreement_contrasts.csv)")
    (out / "agreement_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
