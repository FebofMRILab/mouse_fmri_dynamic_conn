"""
hmm_model_selection.py
-------------------------------------------------------------------
Which HMM fit the Python statistics and figures read, and the check that every
table they read comes from that one fit. Same defaults, environment variables
and folder names as hmm_config.m (MATLAB), so both sides point at the same fit:

    HMM_K          number of states                  (default 4)
    HMM_LOWRANK    HMM-PCA rank, options.lowrank     (default 15; 0 = off)
    HMM_PCA        PCA before the HMM, options.pca   (default 0)
    HMM_OUT_BASE   folder with run4d's tables        (default
                   /blue/febo/share/Abeta_Age/rabies/signals)
    HMM_MODEL_DIR  the fit's folder                  (default
                   <HMM_OUT_BASE>/hmmmar_simple_k<K>_hmmpca<r>, or _pca<p>)

Reported model: HMM-PCA, rank 15, K = 4 (see hmm_v2/CHANGES.md, "Reported model").
"""

import os
import warnings
from pathlib import Path

import pandas as pd

K = int(os.environ.get("HMM_K", "4"))
LOWRANK = int(os.environ.get("HMM_LOWRANK", "15"))
PCA = int(os.environ.get("HMM_PCA", "0"))
if LOWRANK > 0 and PCA > 0:
    raise SystemExit("Set either HMM_LOWRANK (HMM-PCA) or HMM_PCA, not both.")

OUT_BASE = Path(os.environ.get("HMM_OUT_BASE", "/blue/febo/share/Abeta_Age/rabies/signals"))
if LOWRANK > 0:
    MODEL_TYPE, N_COMPONENTS = "lowrank", LOWRANK
    FOLDER = f"hmmmar_simple_k{K}_hmmpca{LOWRANK}"
    TABLE_SUFFIX, COMPONENT_COLUMN = "all_K_all_lowrank", "lowrank_components"
    LABEL = f"HMM-PCA (lowrank {LOWRANK}), K = {K}"
    TAG = f"hmmpca{LOWRANK}_k{K}"
else:
    MODEL_TYPE, N_COMPONENTS = "pca", PCA
    FOLDER = f"hmmmar_simple_k{K}_pca{PCA}"
    TABLE_SUFFIX, COMPONENT_COLUMN = "all_K_all_PCA", "pca_components"
    LABEL = f"PCA {PCA} + HMM, K = {K}"
    TAG = f"pca{PCA}_k{K}"
MODEL_DIR = Path(os.environ.get("HMM_MODEL_DIR") or OUT_BASE / FOLDER)

GRAPH_METRICS_CSV = (MODEL_DIR / "hmm_raw_posthoc_graph_metrics"
                     / "graph_metrics_per_subject_per_state_density15.csv")
BOOTSTRAP_CSV = MODEL_DIR / "hmm_bootstrap_stability" / "bootstrap_stability_subject_state.csv"


def table_path(kind):
    """run4d table for this model family: kind = 'switching_rates', 'maxFO',
    'dwell_times', 'subject_state_FO' or 'subject_model_metrics'."""
    return OUT_BASE / f"hmm_{kind}_{TABLE_SUFFIX}.csv"


def select_model(frame, name):
    """Rows of a run4d table for the selected K and model dimension."""
    k_col = "K_requested" if "K_requested" in frame.columns else "K"
    missing = {k_col, COMPONENT_COLUMN} - set(frame.columns)
    if missing:
        raise ValueError(f"{name} lacks columns {sorted(missing)}; is it a run4d "
                         f"'{TABLE_SUFFIX}' table?")
    out = frame[(frame[k_col] == K) & (frame[COMPONENT_COLUMN] == N_COMPONENTS)].copy()
    if out.empty:
        raise ValueError(f"No rows in {name} for {LABEL}.")
    return out


def check_fit_id(frame, name, fit_ids=None):
    """Every row must carry one fit ID, the one in <MODEL_DIR>/fit_id.txt.
    Returns that ID; `fit_ids` (a set) collects the IDs seen across tables."""
    if "fit_id" not in frame.columns:
        warnings.warn(f"{name} has no fit_id column (written before the fit ID "
                      "was added); its fit cannot be checked.")
        return None
    ids = frame["fit_id"].dropna().astype(str).str.strip().unique()
    if len(ids) != 1:
        raise SystemExit(f"{name} mixes {len(ids)} fits: {list(ids)}")
    fid = ids[0]
    id_file = MODEL_DIR / "fit_id.txt"
    if id_file.is_file() and id_file.read_text().strip() != fid:
        raise SystemExit(f"{name} is not from the fit in {id_file}.")
    if fit_ids is not None:
        fit_ids.add(fid)
        if len(fit_ids) > 1:
            raise SystemExit(f"The tables come from different fits: {sorted(fit_ids)}")
    return fid
