#!/usr/bin/env python3
"""
make_slice_timing_pairs.py  (helper for slice_timing_lag_check.py)

Builds slice_timing_pairs.csv: one row per mouse with its ROI signal file, its
RABIES native-space label map and its group.

Input: a copy of hmm_signals/meta_data.csv with an added bids_id column
(subject_id = signal file number, group, bids_id = BIDS subject label, with or
without the 'sub-' prefix).

It reports every problem at once instead of stopping at the first: rows with
no bids_id, IDs with no label file, IDs with more than one, and missing signal
files. 'sub-1' never matches 'sub-10'. Pairs are written only when every row
with a signal file is matched, unless --allow_missing is given.

Usage (in signals/):
    python make_slice_timing_pairs.py --meta meta_data_with_bids.csv \
        --labels_dir /blue/febo/share/Abeta_Age/rabies/<preprocess_outputs>/bold_datasink/native_labels
"""
import argparse
import os
import re
import sys

import pandas as pd


def clean_id(v):
    s = str(v).strip()
    if s.lower() in ("", "nan", "none"):
        return None
    s = re.sub(r"^sub-", "", s)
    s = re.sub(r"\.0$", "", s)          # 7 read as 7.0
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="meta_data_with_bids.csv")
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--signals_dir", default="hmm_signals")
    ap.add_argument("--file_pattern", default="signal{}.txt")
    ap.add_argument("--out", default="slice_timing_pairs.csv")
    ap.add_argument("--allow_missing", action="store_true",
                    help="write the pairs that matched even if some rows did not")
    a = ap.parse_args()

    m = pd.read_csv(a.meta, dtype=str, encoding="utf-8-sig", sep=None, engine="python")
    m.columns = [re.sub(r"[^a-z0-9_]", "", c.strip().lower()) for c in m.columns]
    need = {"subject_id", "group", "bids_id"}
    if not need <= set(m.columns):
        sys.exit(f"{a.meta} needs columns {sorted(need)}; found {list(m.columns)}")
    m = m[m.subject_id.notna() & (m.subject_id.str.strip() != "")].copy()
    m["subject_id"] = m.subject_id.map(clean_id)
    m["bids_id"] = m.bids_id.map(clean_id)

    if not os.path.isdir(a.labels_dir):
        sys.exit(f"labels_dir not found: {a.labels_dir}")
    files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(a.labels_dir)
                   for f in fs if f.endswith((".nii", ".nii.gz")))
    print(f"{len(files)} NIfTI files under {a.labels_dir}")
    for f in files[:3]:
        print("  e.g.", os.path.relpath(f, a.labels_dir))

    rows, problems = [], []
    for _, r in m.iterrows():
        sig = os.path.join(a.signals_dir, a.file_pattern.format(r.subject_id))
        if not os.path.exists(sig):
            problems.append(f"subject_id {r.subject_id}: signal file {sig} not found")
            continue
        bid = r.bids_id if isinstance(r.bids_id, str) else None   # pandas turns None into NaN
        if bid is None:
            problems.append(f"subject_id {r.subject_id} (group {r.group}): bids_id is empty")
            continue
        pat = re.compile(r"sub-" + re.escape(bid) + r"(?![A-Za-z0-9])")
        hit = [f for f in files if pat.search(os.path.relpath(f, a.labels_dir))]
        if len(hit) != 1:
            problems.append(f"subject_id {r.subject_id}, bids_id {bid}: "
                            f"{len(hit)} label files" + (": " + "; ".join(hit[:4]) if hit else ""))
            continue
        rows.append(dict(signal_file=sig, label_file=hit[0], group=r.group,
                         subject_id=r.subject_id, bids_id=bid))

    print(f"{len(rows)} of {len(m)} rows matched")
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print("  " + p)
    dup = pd.DataFrame(rows).label_file.duplicated(keep=False) if rows else []
    if len(rows) and dup.any():
        print("ERROR: the same label file is paired with more than one subject_id:")
        print(pd.DataFrame(rows)[dup].to_string(index=False))
        sys.exit(1)
    if problems and not a.allow_missing:
        sys.exit("Nothing written. Fix the rows above, or rerun with --allow_missing.")
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
