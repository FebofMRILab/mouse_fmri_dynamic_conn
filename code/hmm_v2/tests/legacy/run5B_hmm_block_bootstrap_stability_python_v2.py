#!/usr/bin/env python3
"""
run5B_hmm_block_bootstrap_stability.py   (Python port of the MATLAB v1
script, with v2 diagnostics)
-------------------------------------------------------------------
Block-bootstrap stability of subject-specific, state-specific HMM
functional-connectivity networks.

Outputs (in --out_dir)
    bootstrap_stability_subject_state.csv   one row per subject x state
    bootstrap_stability_by_state.csv        state summary
    bootstrap_settings.json                 run settings

Primary outcomes (same definitions as the MATLAB v1 script)
    Neff (Kish), p/Neff, covariance rank and entropy effective rank,
    edge-set Jaccard and nodal-strength Spearman rho between each
    resample's network and the full-data network, bootstrap success
    fraction, analyzable classification.

v2 diagnostics
    (1) Similarity to the full-data network is optimistic, because each
        resample shares ~63% of its unique blocks with the full data.
        Two less optimistic metrics are added:
          - resample-vs-resample: disjoint pairs of independent
            successful resamples;
          - split-half: first vs second half of the run (no shared time
            points; the posteriors still come from an HMM fit to all
            data).
    (2) Posterior mass (sum of weights = T*FO) is recorded next to Neff;
        --min_posterior_mass optionally adds it to the occupancy criterion.
        Failure accounting per resample, and an unfiltered sensitivity
        summary that keeps resamples whose resampled Neff < min_neff.
    (3) Switching descriptors from the MAP path: subject transitions,
        subject max FO, state visits, mean dwell, and a near-static flag
        (FO >= near_static_fo).

Reproducibility
    Each subject gets its own child of numpy.random.SeedSequence(seed),
    so results do not depend on --n_jobs. MATLAB's twister/randi draws
    cannot be replayed in numpy: deterministic outputs (FO, Neff,
    covariance, original network, split-half) match MATLAB v1, while
    bootstrap summaries differ only by Monte Carlo error.

Input
    raw_hmm_results.mat with struct R holding Gamma (cell, one T x K
    matrix per subject), subject_ids, groups, K, n_roi. Both pre-v7.3
    (scipy) and v7.3/HDF5 (h5py) files are read. If `groups` cannot be
    decoded (e.g. a MATLAB string array), pass --groups_csv with columns
    subject_id, group; the v1 bootstrap CSV works for this.

Presets (--dataset) reproduce the settings of the two MATLAB runs;
any explicit flag overrides the preset.
    mouse : comp 50, ../signals/hmmmar_simple_k{K}_hmmpca{comp},
            time series ../signals/hmm_signals, density 0.15, Neff >= 50
    human : comp 15, ../signals/hmmmar_simple_k{K}_hmm{comp},
            time series hmm_signals, density 0.20, Neff >= 40

Usage (HiPerGator; run from the folder the MATLAB scripts ran from)
    python run5B_hmm_block_bootstrap_stability.py --dataset mouse --n_jobs 8
    python run5B_hmm_block_bootstrap_stability.py --dataset human --n_jobs 8
"""

PRESETS = {
    "mouse": dict(comp=50,
                  dir_template="../signals/hmmmar_simple_k{K}_hmmpca{comp}",
                  data_dir="../signals/hmm_signals",
                  graph_density=0.15, min_effective_n=50),
    "human": dict(comp=15,
                  dir_template="../signals/hmmmar_simple_k{K}_hmm{comp}",
                  data_dir="hmm_signals",
                  graph_density=0.20, min_effective_n=50),
}

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import rankdata

EPS = np.finfo(float).eps


# ==================================================================
# Loading
# ==================================================================
def _h5_to_str(f, obj):
    arr = np.asarray(f[obj][()]).ravel()
    return "".join(chr(int(c)) for c in arr)


def _load_mat_v73(path):
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        R = f["R"]
        refs = np.asarray(R["Gamma"][()]).ravel()
        out["Gamma"] = [np.asarray(f[r][()]).T for r in refs]
        out["subject_ids"] = np.asarray(R["subject_ids"][()]).ravel()
        out["K"] = int(np.asarray(R["K"][()]).ravel()[0])
        out["n_roi"] = int(np.asarray(R["n_roi"][()]).ravel()[0])
        groups = None
        if "groups" in R:
            g = R["groups"]
            try:
                if g.dtype == h5py.ref_dtype or g.dtype.kind == "O":
                    groups = [_h5_to_str(f, r)
                              for r in np.asarray(g[()]).ravel()]
                elif g.attrs.get("MATLAB_class", b"") == b"char":
                    arr = np.asarray(g[()])
                    groups = ["".join(chr(int(c)) for c in row)
                              for row in arr.T]
                elif g.attrs.get("MATLAB_class", b"") in (
                        b"double", b"single", b"int32", b"int64",
                        b"uint8", b"int8", b"int16", b"uint16"):
                    groups = [f"{x:g}" for x in
                              np.asarray(g[()], float).ravel()]
            except Exception:
                groups = None
        out["groups"] = groups
        out["groups_raw_type"] = ("HDF5 (v7.3), MATLAB_class="
                                  + str(R["groups"].attrs.get(
                                      "MATLAB_class", b"?"))
                                  if "groups" in R else "missing")
    return out


def _load_mat_legacy(path):
    from scipy.io import loadmat
    R = loadmat(path, simplify_cells=True)["R"]
    out = {
        "Gamma": [np.asarray(g, dtype=float) for g in np.atleast_1d(
            np.asarray(R["Gamma"], dtype=object))],
        "subject_ids": np.asarray(R["subject_ids"]).ravel(),
        "K": int(np.asarray(R["K"]).ravel()[0]),
        "n_roi": int(np.asarray(R["n_roi"]).ravel()[0]),
    }
    raw = R.get("groups")
    out["groups_raw_type"] = type(raw).__name__
    try:
        out["groups"] = [str(g) for g in np.atleast_1d(raw)]
    except Exception:
        out["groups"] = None
    return out


def _groups_look_valid(groups, S):
    """Reject lists that are really undecoded MATLAB objects (string or
    categorical arrays load as opaque MCOS data)."""
    if groups is None or len(groups) != S:
        return False
    bad = ("MCOS", "opaque", "array(", "<", "None", "b'")
    return all(isinstance(g, str) and 0 < len(g) < 64
               and not any(b in g for b in bad) for g in groups)


def _groups_from_csv(csv_path, subject_ids):
    g = pd.read_csv(csv_path)
    if not {"subject_id", "group"} <= set(g.columns):
        raise ValueError(f"{csv_path} needs columns subject_id and group.")
    g = g[["subject_id", "group"]].drop_duplicates()
    dup = g.subject_id[g.subject_id.duplicated()]
    if len(dup):
        raise ValueError(f"{csv_path}: subjects with more than one group: "
                         f"{sorted(dup.unique())[:10]}")
    g = g.set_index("subject_id")["group"]
    missing = [int(s) for s in subject_ids if int(s) not in g.index]
    if missing:
        raise ValueError(f"{csv_path} lacks group labels for subject_id(s) "
                         f"{missing[:10]}{' ...' if len(missing) > 10 else ''}")
    return [str(g.loc[int(s)]) for s in subject_ids]


def load_hmm_results(path, groups_csv=None, fallback_csvs=()):
    """Load R. Group labels come from, in order: --groups_csv; R.groups if
    it decodes to plain labels; the first existing fallback CSV (e.g. the
    MATLAB v1 bootstrap_stability_subject_state.csv)."""
    try:
        R = _load_mat_legacy(path)
    except (NotImplementedError, ValueError):
        R = _load_mat_v73(path)
        R.setdefault("groups_raw_type", "HDF5 (v7.3)")

    S = len(R["subject_ids"])
    if len(R["Gamma"]) != S:
        raise ValueError("Number of Gamma matrices does not match "
                         "number of subjects.")

    if groups_csv is not None:
        R["groups"] = _groups_from_csv(groups_csv, R["subject_ids"])
        print(f"Group labels read from {groups_csv}")
        return R
    if _groups_look_valid(R.get("groups"), S):
        return R
    for c in fallback_csvs:
        if c and os.path.exists(c):
            R["groups"] = _groups_from_csv(c, R["subject_ids"])
            print(f"R.groups could not be decoded "
                  f"({R.get('groups_raw_type')}); group labels read from {c}")
            return R
    raise ValueError(
        f"Could not decode R.groups (loaded as {R.get('groups_raw_type')}; "
        "MATLAB string and categorical arrays cannot be read from Python) "
        "and no fallback CSV was found. Either pass --groups_csv <file> with "
        "columns subject_id, group, or re-save the labels in MATLAB as a "
        "cell array:  R.groups = cellstr(string(R.groups)); "
        "save(path, 'R', '-v7.3')  or export them:  "
        "writetable(table(R.subject_ids(:), string(R.groups(:)), "
        "'VariableNames', {'subject_id','group'}), 'groups.csv')")


def read_timeseries(path):
    try:
        return np.loadtxt(path, dtype=float, ndmin=2)
    except ValueError:
        return pd.read_csv(path, sep=None, header=None,
                           engine="python").to_numpy(dtype=float)


# ==================================================================
# Core computations (mirror the MATLAB local functions)
# ==================================================================
def effective_n(w):
    w = np.clip(np.asarray(w, float).ravel(), 0, None)
    s1, s2 = w.sum(), (w ** 2).sum()
    if s1 <= EPS or s2 <= EPS:
        return 0.0
    return s1 ** 2 / s2


def weighted_covariance(X, w):
    """Reliability-weighted unbiased covariance and correlation."""
    w = np.clip(np.asarray(w, float).ravel(), 0, None)
    tw = w.sum()
    if tw <= EPS:
        return None, None
    denom = tw - (w ** 2).sum() / tw
    if denom <= EPS:
        return None, None
    mu = (X.T @ w) / tw
    Xc = X - mu
    C = (Xc.T @ (Xc * w[:, None])) / denom
    C = (C + C.T) / 2
    v = np.diag(C)
    if not np.all(np.isfinite(v)) or np.any(v <= EPS):
        return None, None
    sd = np.sqrt(v)
    Rm = np.clip(C / np.outer(sd, sd), -1, 1)
    Rm = (Rm + Rm.T) / 2
    np.fill_diagonal(Rm, 1.0)
    if not (np.all(np.isfinite(C)) and np.all(np.isfinite(Rm))):
        return None, None
    return C, Rm


def covariance_rank(C):
    ev = np.linalg.eigvalsh((C + C.T) / 2)
    mx = ev.max() if ev.size else np.nan
    if not np.isfinite(mx) or mx <= EPS:
        return np.nan
    return int(np.sum(ev > C.shape[0] * np.spacing(mx)))


def effective_rank(C):
    ev = np.clip(np.linalg.eigvalsh((C + C.T) / 2), 0, None)
    tot = ev.sum()
    if tot <= EPS:
        return np.nan
    q = ev[ev > 0] / tot
    return float(np.exp(-np.sum(q * np.log(q))))


class Graph:
    """Positive-edge proportional-threshold graph builder for fixed p."""

    def __init__(self, p, density):
        self.p = p
        self.iu, self.ju = np.triu_indices(p, 1)
        self.n_upper = self.iu.size
        self.n_keep = max(1, int(round(density * self.n_upper)))
        self.pos = np.arange(self.n_upper)

    def build(self, Rm):
        """Return (edge mask over upper triangle, nodal strength) or None.
        Ties are broken by upper-triangle position, as in MATLAB."""
        if Rm is None or not np.all(np.isfinite(Rm)):
            return None
        uw = np.maximum(((Rm + Rm.T) / 2)[self.iu, self.ju], 0)
        positive = uw > 0
        n_pos = int(positive.sum())
        if n_pos == 0:
            return None
        n_keep = min(self.n_keep, n_pos)
        order = np.lexsort((self.pos, -uw))[:n_keep]
        mask = np.zeros(self.n_upper, bool)
        mask[order] = True
        strength = np.zeros(self.p)
        sel_w = uw[order]
        np.add.at(strength, self.iu[order], sel_w)
        np.add.at(strength, self.ju[order], sel_w)
        return mask, strength


def jaccard(a, b):
    u = np.count_nonzero(a | b)
    return np.nan if u == 0 else np.count_nonzero(a & b) / u


def spearman(x, y):
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return np.nan
    rx, ry = rankdata(x), rankdata(y)
    if rx.std(ddof=1) <= EPS or ry.std(ddof=1) <= EPS:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def prctile(x, q):
    """MATLAB prctile equivalent (Hazen plotting positions)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.full(np.size(q), np.nan)
    return np.percentile(x, q, method="hazen")


def nanmedian(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else np.nan


def naniqr(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    q = prctile(x, [25, 75])
    return float(q[1] - q[0])


def circular_block_indices(T, L, rng):
    nb = int(np.ceil(T / L))
    starts = rng.integers(0, T, size=nb)
    idx = (starts[:, None] + np.arange(L)[None, :]) % T
    return idx.ravel()[:T]


# ==================================================================
# Per-subject worker
# ==================================================================
def process_subject(args):
    (s, sid, group, X, G, cfg, seed_seq) = args
    boot_ss, pair_ss = seed_seq.spawn(2)
    rng = np.random.default_rng(boot_ss)
    pair_rng = np.random.default_rng(pair_ss)

    K = G.shape[1]
    T, p = X.shape
    graph = Graph(p, cfg["graph_density"])
    min_neff = cfg["min_effective_n"]
    n_boot = cfg["n_bootstrap"]

    L = (max(2, int(np.ceil(T ** (1 / 3))))
         if cfg["block_length"] is None else min(T, cfg["block_length"]))

    map_path = np.argmax(G, axis=1)
    run_starts = np.r_[True, np.diff(map_path) != 0]
    run_labels = map_path[run_starts]
    n_trans = int(np.count_nonzero(np.diff(map_path)))
    max_fo = float(G.mean(axis=0).max())

    rows = []
    for k in range(K):
        w = G[:, k]
        r = dict(subject_id=sid, state=k + 1, group=group,
                 n_timepoints=T, block_length=L)
        r["FO"] = float(w.mean())
        r["map_count"] = int(np.sum(map_path == k))
        r["n_visits"] = int(np.sum(run_labels == k))
        r["mean_dwell_volumes"] = (r["map_count"] / r["n_visits"]
                                   if r["n_visits"] > 0 else np.nan)
        r["subject_n_transitions"] = n_trans
        r["subject_max_FO"] = max_fo
        r["near_static"] = r["FO"] >= cfg["near_static_fo"]

        neff = effective_n(w)
        r["Neff"] = neff
        r["posterior_mass"] = float(w.sum())
        r["p_over_Neff"] = p / max(neff, EPS)
        # Kish Neff >= posterior mass; with diffuse posteriors Neff can
        # approach T for a state the subject never enters, so an optional
        # mass criterion can be added (off by default = v1 behaviour).
        r["passes_Neff"] = bool(
            neff >= min_neff and
            (cfg["min_posterior_mass"] is None
             or r["posterior_mass"] >= cfg["min_posterior_mass"]))
        r["valid_original_network"] = False
        r["analyzable"] = False
        r["bootstrap_success_count"] = 0
        r["bootstrap_success_fraction"] = 0.0
        r["split_half_valid"] = False
        r["n_resample_pairs"] = 0
        r["n_fail_boot_Neff"] = 0
        r["n_fail_boot_network"] = 0
        r["n_unfiltered"] = 0
        rows.append(r)

        C0, R0 = weighted_covariance(X, w)
        if C0 is None:
            continue
        r["covariance_rank"] = covariance_rank(C0)
        r["covariance_effective_rank"] = effective_rank(C0)

        g0 = graph.build(R0)
        if g0 is None:
            continue
        mask0, str0 = g0
        r["original_positive_edges"] = int(mask0.sum())
        r["original_graph_density"] = mask0.sum() / graph.n_upper
        r["valid_original_network"] = True

        if not r["passes_Neff"]:
            continue

        # ---------------- split-half ----------------
        if cfg["run_split_half"]:
            half = T // 2
            parts = [slice(0, half), slice(half, T)]
            res = []
            for h, sl in enumerate(parts):
                wh = G[sl, k]
                nh = effective_n(wh)
                r[f"split_half_Neff_{h + 1}"] = nh
                if nh < min_neff:
                    res.append(None)
                    continue
                _, Rh = weighted_covariance(X[sl], wh)
                res.append(graph.build(Rh))
            if all(x is not None for x in res):
                r["split_half_valid"] = True
                r["split_half_edge_jaccard"] = jaccard(res[0][0],
                                                       res[1][0])
                r["split_half_nodal_rank_correlation"] = spearman(
                    res[0][1], res[1][1])

        # ---------------- bootstrap ----------------
        J_f = np.full(n_boot, np.nan)      # filtered (primary)
        rho_f = np.full(n_boot, np.nan)
        J_a = np.full(n_boot, np.nan)      # unfiltered
        rho_a = np.full(n_boot, np.nan)
        neff_b = np.full(n_boot, np.nan)
        masks = np.zeros((n_boot, graph.n_upper), bool)
        strengths = np.full((n_boot, p), np.nan)

        for b in range(n_boot):
            idx = circular_block_indices(T, L, rng)
            Xb = X[idx]
            Gb = G[idx]
            Gb = Gb / Gb.sum(axis=1, keepdims=True)
            wb = Gb[:, k]
            nb = effective_n(wb)
            neff_b[b] = nb
            _, Rb = weighted_covariance(Xb, wb)
            gb = graph.build(Rb)
            if gb is None:
                continue
            mb, sb = gb
            Jb = jaccard(mask0, mb)
            rb = spearman(str0, sb)
            J_a[b], rho_a[b] = Jb, rb
            masks[b] = mb
            strengths[b] = sb
            if nb >= min_neff:
                J_f[b], rho_f[b] = Jb, rb

        ok = np.isfinite(J_f) & np.isfinite(rho_f)
        n_ok = int(ok.sum())
        r["bootstrap_success_count"] = n_ok
        r["bootstrap_success_fraction"] = n_ok / n_boot

        fail_neff = neff_b < min_neff
        r["n_fail_boot_Neff"] = int(fail_neff.sum())
        r["n_fail_boot_network"] = int(np.sum(~fail_neff & ~ok))
        r["median_boot_Neff"] = nanmedian(neff_b)

        ok_all = np.isfinite(J_a) & np.isfinite(rho_a)
        r["n_unfiltered"] = int(ok_all.sum())
        if ok_all.any():
            r["median_edge_jaccard_unfiltered"] = float(
                np.median(J_a[ok_all]))
            r["median_nodal_rank_unfiltered"] = float(
                np.median(rho_a[ok_all]))

        if n_ok > 0:
            vj, vr = J_f[ok], rho_f[ok]
            r["median_edge_jaccard"] = float(np.median(vj))
            r["mean_edge_jaccard"] = float(np.mean(vj))
            r["edge_jaccard_CI_low"], r["edge_jaccard_CI_high"] = \
                prctile(vj, [2.5, 97.5])
            r["median_nodal_strength_rank_correlation"] = float(
                np.median(vr))
            r["mean_nodal_strength_rank_correlation"] = float(np.mean(vr))
            (r["nodal_rank_correlation_CI_low"],
             r["nodal_rank_correlation_CI_high"]) = prctile(vr,
                                                            [2.5, 97.5])

        # ------------- resample-vs-resample -------------
        succ = np.flatnonzero(ok)
        if succ.size >= 2:
            succ = pair_rng.permutation(succ)
            n_pairs = min(cfg["max_resample_pairs"], succ.size // 2)
            a, c = succ[0:2 * n_pairs:2], succ[1:2 * n_pairs:2]
            inter = np.count_nonzero(masks[a] & masks[c], axis=1)
            union = np.count_nonzero(masks[a] | masks[c], axis=1)
            J_rr = inter / np.maximum(union, 1)
            rho_rr = np.array([spearman(strengths[i], strengths[j])
                               for i, j in zip(a, c)])
            r["n_resample_pairs"] = int(n_pairs)
            r["median_edge_jaccard_rr"] = nanmedian(J_rr)
            r["edge_jaccard_rr_CI_low"], r["edge_jaccard_rr_CI_high"] = \
                prctile(J_rr, [2.5, 97.5])
            r["median_nodal_rank_rr"] = nanmedian(rho_rr)
            r["nodal_rank_rr_CI_low"], r["nodal_rank_rr_CI_high"] = \
                prctile(rho_rr, [2.5, 97.5])

        r["analyzable"] = bool(
            r["passes_Neff"] and r["valid_original_network"]
            and r["bootstrap_success_fraction"]
            >= cfg["min_bootstrap_success_fraction"])

    return s, rows


# ==================================================================
# Output tables
# ==================================================================
SUBJECT_COLUMNS = [
    "subject_id", "state", "group", "n_timepoints", "block_length", "FO",
    "map_count", "Neff", "posterior_mass", "p_over_Neff", "covariance_rank",
    "covariance_effective_rank", "original_positive_edges",
    "original_graph_density", "median_edge_jaccard", "mean_edge_jaccard",
    "edge_jaccard_CI_low", "edge_jaccard_CI_high",
    "median_nodal_strength_rank_correlation",
    "mean_nodal_strength_rank_correlation",
    "nodal_rank_correlation_CI_low", "nodal_rank_correlation_CI_high",
    "bootstrap_success_count", "bootstrap_success_fraction",
    "passes_Neff", "valid_original_network", "analyzable",
    # v2
    "n_resample_pairs", "median_edge_jaccard_rr", "edge_jaccard_rr_CI_low",
    "edge_jaccard_rr_CI_high", "median_nodal_rank_rr",
    "nodal_rank_rr_CI_low", "nodal_rank_rr_CI_high",
    "split_half_Neff_1", "split_half_Neff_2", "split_half_valid",
    "split_half_edge_jaccard", "split_half_nodal_rank_correlation",
    "n_fail_boot_Neff", "n_fail_boot_network", "median_boot_Neff",
    "n_unfiltered", "median_edge_jaccard_unfiltered",
    "median_nodal_rank_unfiltered", "n_visits", "mean_dwell_volumes",
    "subject_n_transitions", "subject_max_FO", "near_static",
]
BOOL_COLUMNS = ["passes_Neff", "valid_original_network", "analyzable",
                "split_half_valid", "near_static"]


def state_summary(df):
    out = []
    for k in sorted(df.state.unique()):
        s = df[df.state == k]
        a = s[s.analyzable]
        sh = a[a.split_half_valid]
        out.append({
            "state": k,
            "n_subject_states_total": len(s),
            "n_passing_Neff": int(s.passes_Neff.sum()),
            "n_analyzable": int(s.analyzable.sum()),
            "n_near_static": int(s.near_static.sum()),
            "median_Neff": nanmedian(s.Neff),
            "IQR_Neff": naniqr(s.Neff),
            "median_p_over_Neff": nanmedian(s.p_over_Neff),
            "median_covariance_rank": nanmedian(s.covariance_rank),
            "median_covariance_effective_rank":
                nanmedian(s.covariance_effective_rank),
            "median_edge_jaccard": nanmedian(a.median_edge_jaccard),
            "IQR_edge_jaccard": naniqr(a.median_edge_jaccard),
            "median_nodal_strength_rank_correlation":
                nanmedian(a.median_nodal_strength_rank_correlation),
            "IQR_nodal_strength_rank_correlation":
                naniqr(a.median_nodal_strength_rank_correlation),
            "median_edge_jaccard_rr": nanmedian(a.median_edge_jaccard_rr),
            "median_nodal_rank_rr": nanmedian(a.median_nodal_rank_rr),
            "n_split_half_valid": len(sh),
            "median_split_half_edge_jaccard":
                nanmedian(sh.split_half_edge_jaccard),
            "median_split_half_nodal_rank":
                nanmedian(sh.split_half_nodal_rank_correlation),
            "median_edge_jaccard_unfiltered":
                nanmedian(a.median_edge_jaccard_unfiltered),
            "median_nodal_rank_unfiltered":
                nanmedian(a.median_nodal_rank_unfiltered),
            "percentage_passing_Neff": 100 * s.passes_Neff.mean(),
            "percentage_analyzable_subject_states":
                100 * s.analyzable.mean(),
        })
    return pd.DataFrame(out)


# ==================================================================
# Main
# ==================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(PRESETS), default="mouse",
                    help="settings preset (see header); flags override it")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--comp", type=int, default=None)
    ap.add_argument("--hmm_dir", default=None,
                    help="HMM results folder; default from the preset "
                         "template")
    ap.add_argument("--hmm_results", default=None,
                    help="default <hmm_dir>/raw_hmm_results.mat")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--file_pattern", default="signal{}.txt")
    ap.add_argument("--out_dir", default=None,
                    help="default <hmm_dir>/hmm_bootstrap_stability_py")
    ap.add_argument("--groups_csv", default=None)
    ap.add_argument("--n_bootstrap", type=int, default=1000)
    ap.add_argument("--graph_density", type=float, default=None)
    ap.add_argument("--min_effective_n", type=float, default=None)
    ap.add_argument("--min_bootstrap_success_fraction", type=float,
                    default=0.80)
    ap.add_argument("--block_length", type=int, default=None,
                    help="default ceil(T^(1/3)) per subject")
    ap.add_argument("--max_resample_pairs", type=int, default=500)
    ap.add_argument("--no_split_half", action="store_true")
    ap.add_argument("--min_posterior_mass", type=float, default=None,
                    help="optional: also require sum of posterior weights "
                         ">= this (e.g. 50); off by default (v1 behaviour)")
    ap.add_argument("--near_static_fo", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--n_jobs", type=int, default=1)
    args = ap.parse_args()

    pre = PRESETS[args.dataset]
    for key in ("comp", "data_dir", "graph_density", "min_effective_n"):
        if getattr(args, key) is None:
            setattr(args, key, pre[key])

    hmm_dir = args.hmm_dir or pre["dir_template"].format(K=args.K,
                                                         comp=args.comp)
    hmm_path = args.hmm_results or os.path.join(hmm_dir,
                                                "raw_hmm_results.mat")
    out_dir = args.out_dir or os.path.join(hmm_dir,
                                           "hmm_bootstrap_stability_py")
    os.makedirs(out_dir, exist_ok=True)

    cfg = dict(
        n_bootstrap=args.n_bootstrap,
        graph_density=args.graph_density,
        min_effective_n=args.min_effective_n,
        min_bootstrap_success_fraction=args.min_bootstrap_success_fraction,
        block_length=args.block_length,
        max_resample_pairs=args.max_resample_pairs,
        run_split_half=not args.no_split_half,
        near_static_fo=args.near_static_fo,
        min_posterior_mass=args.min_posterior_mass,
        seed=args.seed,
    )

    print("Loading HMM results...")
    fallback = [os.path.join(hmm_dir, "hmm_bootstrap_stability",
                             "bootstrap_stability_subject_state.csv"),
                os.path.join(out_dir, "bootstrap_stability_subject_state.csv")]
    R = load_hmm_results(hmm_path, args.groups_csv, fallback)
    K, p = R["K"], R["n_roi"]
    sids = [int(x) for x in R["subject_ids"]]
    S = len(sids)
    print(f"Dataset preset: {args.dataset}  HMM folder: {hmm_dir}")
    print(f"Subjects: {S}  States: {K}  ROIs: {p}  "
          f"Bootstraps: {args.n_bootstrap}  Density: {args.graph_density}  "
          f"Min Neff: {args.min_effective_n}")

    jobs = []
    children = np.random.SeedSequence(args.seed).spawn(S)
    for s, sid in enumerate(sids):
        fn = os.path.join(args.data_dir, args.file_pattern.format(sid))
        if not os.path.exists(fn):
            sys.exit(f"Missing time-series file: {fn}")
        X = read_timeseries(fn)
        G = np.asarray(R["Gamma"][s], float)
        if G.ndim != 2:
            sys.exit(f"Gamma for subject {sid} is not 2-D.")
        if X.shape[0] != G.shape[0]:
            sys.exit(f"Time/Gamma mismatch for subject {sid}: "
                     f"{X.shape[0]} vs {G.shape[0]}.")
        if X.shape[1] != p:
            sys.exit(f"ROI mismatch for subject {sid}: expected {p}, "
                     f"found {X.shape[1]}.")
        if G.shape[1] != K:
            sys.exit(f"State mismatch for subject {sid}.")
        if not np.all(np.isfinite(X)):
            sys.exit(f"Nonfinite time-series value for subject {sid}.")
        if not np.all(np.isfinite(G)) or np.any(G < -1e-10):
            sys.exit(f"Invalid Gamma values for subject {sid}.")
        G = np.clip(G, 0, None)
        rs = G.sum(axis=1, keepdims=True)
        if np.any(rs <= EPS):
            sys.exit(f"Zero-mass Gamma row for subject {sid}.")
        jobs.append((s, sid, R["groups"][s], X, G / rs, cfg, children[s]))
    print("All time-series dimensions validated.")

    t0 = time.time()
    results = [None] * S
    if args.n_jobs > 1:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            for n_done, (s, rows) in enumerate(
                    ex.map(process_subject, jobs), 1):
                results[s] = rows
                print(f"  subject {n_done}/{S} (ID={sids[s]}) done "
                      f"[{time.time() - t0:.0f} s]", flush=True)
    else:
        for job in jobs:
            s, rows = process_subject(job)
            results[s] = rows
            print(f"  subject {s + 1}/{S} (ID={sids[s]}) done "
                  f"[{time.time() - t0:.0f} s]", flush=True)

    df = pd.DataFrame([r for rows in results for r in rows])
    df = df.reindex(columns=SUBJECT_COLUMNS)
    for c in BOOL_COLUMNS:
        df[c] = df[c].fillna(False).astype(bool)
    summary = state_summary(df)

    for c in BOOL_COLUMNS:          # write logicals as 0/1, as MATLAB does
        df[c] = df[c].astype(int)
    p1 = os.path.join(out_dir, "bootstrap_stability_subject_state.csv")
    p2 = os.path.join(out_dir, "bootstrap_stability_by_state.csv")
    df.to_csv(p1, index=False)
    summary.to_csv(p2, index=False)
    with open(os.path.join(out_dir, "bootstrap_settings.json"), "w") as f:
        json.dump({**cfg, "dataset": args.dataset, "n_roi": p, "K": K,
                   "n_subjects": S,
                   "hmm_results": hmm_path, "version": "python-v2"},
                  f, indent=2)

    print("\n" + "=" * 52)
    print("Block-bootstrap stability analysis complete.")
    print(f"Subject-state table: {p1}\nState summary table: {p2}")
    print("=" * 52)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary.T)


if __name__ == "__main__":
    main()