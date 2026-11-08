"""
hmm_model_selection.py
-------------------------------------------------------------------
Which HMM fit the Python statistics and figures read, and the check that every
table they read comes from that one fit. Same defaults, environment variables
and folder names as hmm_config.m (MATLAB), so both sides point at the same fit:

    HMM_K          number of states                  (default: MODEL_PRIMARY.env)
    HMM_LOWRANK    HMM-PCA rank, options.lowrank     (default: MODEL_PRIMARY.env; 0 = off)
    HMM_PCA        PCA before the HMM, options.pca   (default: MODEL_PRIMARY.env; 0 = off)
    HMM_OUT_BASE   folder with run4d's tables        (default
                   /blue/febo/share/Abeta_Age/rabies/signals)
    HMM_MODEL_DIR  the fit's folder                  (default
                   <HMM_OUT_BASE>/hmmmar_simple_k<K>_hmmpca<r>, or _pca<p>)

Reported models (hmm_v2/FIT_SELECTION_RULE.md amendment 1, 27 Oct 2026; D33),
each set in one file in hmm_v2/:
    primary    MODEL_PRIMARY.env    Gaussian HMM on 25 PCs (options.pca = 25), K = 3
    secondary  MODEL_SECONDARY.env  HMM-PCA (options.lowrank = 15), K = 2
The job scripts select the secondary by exporting HMM_K, HMM_LOWRANK and
HMM_PCA from its file (hipergator_tests/reported_model.sh secondary).

What the group tests cover (inference_scope): hmm_v2/select_reported_fit.m
writes <model folder>/inference_scope.csv by the pre-specified rule in
hmm_v2/FIT_SELECTION_RULE.md: tests only for states that recur across
independent fits, and for switching rate and max FO only if their per-mouse
values agree across fits. Without that file either reported model is refused
(HMM_ALLOW_NO_SCOPE=1 overrides, for tests); other models are tested in full,
with a warning.

K = 2 (fo_test_plan): each mouse's two occupancies sum to 1, so the group test
of state 2's occupancy is the test of state 1's (E50); it is run and counted once.
"""

import os
import re
import warnings
from pathlib import Path

import pandas as pd

MODEL_FILE_DIR = Path(os.environ.get("HMM_MODEL_FILE_DIR")
                      or Path(__file__).resolve().parent.parent / "hmm_v2")


def read_model_file(path):
    """NAME=integer lines (HMM_K, HMM_LOWRANK, HMM_PCA); '#' starts a comment
    line. Same format and checks as hmm_config.m's read_model_file."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"No model file {path} (the reported models are set in hmm_v2/; "
                         "HMM_MODEL_FILE_DIR points elsewhere).")
    out = {}
    for line in path.read_text().splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        m = re.fullmatch(r"(HMM_K|HMM_LOWRANK|HMM_PCA)=(\d+)", t)
        if not m:
            raise SystemExit(f"Unreadable line in {path}: {t!r}")
        out[m.group(1)] = int(m.group(2))
    if set(out) != {"HMM_K", "HMM_LOWRANK", "HMM_PCA"}:
        raise SystemExit(f"{path} must set HMM_K, HMM_LOWRANK and HMM_PCA.")
    return (out["HMM_K"], out["HMM_LOWRANK"], out["HMM_PCA"])


REPORTED_MODELS = {role: read_model_file(MODEL_FILE_DIR / f"MODEL_{role.upper()}.env")
                   for role in ("primary", "secondary")}
_default = REPORTED_MODELS["primary"]
K = int(os.environ.get("HMM_K", _default[0]))
LOWRANK = int(os.environ.get("HMM_LOWRANK", _default[1]))
PCA = int(os.environ.get("HMM_PCA", _default[2]))
if LOWRANK > 0 and PCA > 0:
    raise SystemExit("Set either HMM_LOWRANK (HMM-PCA) or HMM_PCA, not both.")
ROLE = next((r for r, m in REPORTED_MODELS.items() if m == (K, LOWRANK, PCA)), None)

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
    LABEL = (f"Gaussian HMM on {PCA} PCs (options.pca), K = {K}" if PCA > 0
             else f"Gaussian HMM, all ROIs, K = {K}")
    TAG = f"pca{PCA}_k{K}"
if ROLE:
    LABEL = f"{LABEL} [{ROLE} model]"
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


SCOPE_FILE = MODEL_DIR / "inference_scope.csv"


def inference_scope():
    """States and whole-fit outcomes the group tests may cover.

    Returns a dict: 'states' (sorted state numbers to test, or None = every
    state), 'switching_rate' and 'max_FO' (True = test it), 'source' (text for
    logs). The file must come from the fit in <MODEL_DIR>/fit_id.txt.
    """
    if not SCOPE_FILE.is_file():
        if ROLE and os.environ.get("HMM_ALLOW_NO_SCOPE") != "1":
            raise SystemExit(
                f"No {SCOPE_FILE}. Group tests on the {ROLE} reported model ({LABEL}) are "
                "limited by the fit-selection rule (hmm_v2/FIT_SELECTION_RULE.md): run "
                "select_reported_fit.m and use the chosen fit (hipergator_tests/run_selected_fit.sh "
                f"{ROLE}). HMM_ALLOW_NO_SCOPE=1 tests every state anyway (tests only, not for the paper).")
        warnings.warn(f"No {SCOPE_FILE}: this fit was not chosen by the fit-selection rule "
                      "(hmm_v2/FIT_SELECTION_RULE.md); every state is tested.")
        return {"states": None, "switching_rate": True, "max_FO": True,
                "source": "no inference_scope.csv (fit not chosen by the rule): every state tested"}
    scope = pd.read_csv(SCOPE_FILE, dtype={"item": str})
    ids = scope["fit_id"].astype(str).str.strip().unique()
    id_file = MODEL_DIR / "fit_id.txt"
    if len(ids) != 1 or not id_file.is_file() or id_file.read_text().strip() != ids[0]:
        raise SystemExit(f"{SCOPE_FILE} is not from the fit in {id_file}.")
    st = scope[scope["kind"] == "state"]
    states = sorted(int(x) for x in st.loc[st["tested"] == 1, "item"])
    whole = scope[scope["kind"] == "whole_fit"].set_index("item")["tested"]
    out = {"states": states,
           "switching_rate": bool(whole.get("switching_rate", 0) == 1),
           "max_FO": bool(whole.get("max_FO", 0) == 1)}
    out["source"] = (f"{SCOPE_FILE.name}: states tested {states or 'none'}; switching rate "
                     f"{'tested' if out['switching_rate'] else 'not tested'}; max FO "
                     f"{'tested' if out['max_FO'] else 'not tested'}")
    return out


def tested_states(state_order, scope):
    """The states in state_order that the group tests cover. State labels may be
    numbers, '1' or 'State 1'; the number is what counts."""
    if scope["states"] is None:
        return list(state_order)
    keep = set(scope["states"])
    return [s for s in state_order if state_number(s) in keep]


def state_number(label):
    """1 from 1, '1' or 'State 1'."""
    return int(str(label).strip().lower().replace("state", "").strip())


def fo_test_plan(states, n_states=None, fo=None):
    """Which states' fractional occupancy gets its own group test.

    Each mouse's occupancies sum to 1 over the K states. With K = 2, FO of one
    state is 1 minus FO of the other, so Welch's ANOVA, the Welch t contrasts
    (sign reversed) and the permutation p are the same test for both states
    (FIT_SELECTION_RULE.md amendment 1; E50). That test is run once, for the
    first tested state, and counted once in the Holm family; the other state
    takes its result. With K > 2 no state's FO is a function of one other
    state's, and every state is tested.

    states    tested state labels, in order (numbers, '1' or 'State 1')
    n_states  K of the fit (default: this model's K)
    fo        optional table with columns subject_id, state, fractional_occupancy:
              the identity is checked on it (every mouse's two FO sum to 1
              within 1e-6); if it does not hold, both states are tested, with
              a warning
    Returns {label: label of the state whose test it uses}; a state maps to
    itself when it is tested on its own.
    """
    n_states = K if n_states is None else n_states
    states = list(states)
    plan = {s: s for s in states}
    if n_states == 2 and len(states) == 2:
        if fo is not None:
            w = fo.pivot_table(index="subject_id", columns="state",
                               values="fractional_occupancy", observed=False)
            dev = (w[states[0]] + w[states[1]] - 1).abs().max()
            if not dev < 1e-6:
                warnings.warn(f"K = 2 but the two occupancies do not sum to 1 (max deviation "
                              f"{dev:.3g}); both states are tested.")
                return plan
        plan[states[1]] = states[0]
    return plan


def fresh_output_dir(path, patterns):
    """Delete the files matching `patterns` that an earlier run of the same
    script left in its output folder, so no table or figure made under another
    scope survives next to the new ones. Creates the folder."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for f in path.glob(pattern):
            if f.is_file():
                f.unlink()

