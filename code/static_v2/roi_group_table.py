#!/usr/bin/env python3
"""
roi_group_table.py
-------------------------------------------------------------------
ROI-by-ROI group comparison tables (Python port of statistical_table.r).

Same input, layout and outputs as the R script (mean ± SD by group per ROI,
test statistic with df, adjusted p-values with stars; CSV, Word and multipage
landscape PDF), with the study's group tests from welch_tests_scipy.py
(SciPy >= 1.16) in place of Kruskal-Wallis and Dunn's tests:

  omnibus per ROI   Welch ANOVA across the 7 groups, p from 10,000 label
                    permutations (scipy permutation_test of
                    f_oneway(equal_var=False)); df from statsmodels
                    anova_oneway; BH-FDR across ROIs (statsmodels)
  contrasts         the 8 planned contrasts used in every other analysis, as
                    ordinary two-sided Welch t tests (scipy ttest_ind,
                    equal_var=False); BH-FDR pooled across contrasts x ROIs,
                    as for the edgewise contrasts

Input
    a wide CSV: first column subject index, second column group, then one
    column per ROI (default mice_NodePosStrength_PCorr.csv, as in the R script)

Outputs (in --out_dir, prefix --prefix, default "Strength")
    <prefix>_summary_table.{csv,docx,pdf}             omnibus table
    <prefix>_planned_contrasts_table.{csv,docx,pdf}   contrast table
    <prefix>_roi_tests_numeric.csv                    unformatted results

Usage
    python roi_group_table.py [input.csv] [--prefix Strength] [--out_dir .]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from welch_tests_scipy import (welch_anova_df, welch_anova_permutation,  # noqa: E402
                               welch_t_test_df)

# Group order and display names as in the R script (underscores removed).
GROUP_ORDER = ["NTGY", "NTGM", "NTGA", "CRND8Y", "CRND8M", "5XFADY", "5XFADM"]
# The same 8 planned contrasts as every other analysis in the study.
PAIRWISE_COMPARISONS = [
    ("NTGY", "NTGM"), ("NTGY", "NTGA"), ("NTGY", "CRND8Y"),
    ("NTGM", "CRND8M"), ("NTGY", "5XFADY"), ("NTGM", "5XFADM"),
    ("CRND8Y", "CRND8M"), ("5XFADY", "5XFADM"),
]
N_PERMUTATIONS = 10000
RANDOM_SEED = 20260913          # as the HMM graph-metric script


# ------------------------------------------------------------------
# Formatting (as in the R script)
# ------------------------------------------------------------------
def mean_sd(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return "NA"
    if x.size == 1:
        return f"{x.mean():.2f} ± NA"
    return f"{x.mean():.2f} ± {x.std(ddof=1):.2f}"


def format_p(p):
    if not np.isfinite(p):
        return "NA"
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    return ("<0.001" if p < 0.001 else f"{p:.3f}") + stars


def bh(p):
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return out


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
def load(path):
    dat = pd.read_csv(path)
    dat = dat.rename(columns={dat.columns[0]: "index", dat.columns[1]: "group"})
    dat["group"] = dat["group"].astype(str).str.strip().str.replace("_", "", regex=False)
    unexpected = sorted(set(dat["group"]) - set(GROUP_ORDER))
    if unexpected:
        raise SystemExit(f"Unexpected group names found: {', '.join(unexpected)}")
    rois = list(dat.columns[2:])
    dat[rois] = dat[rois].apply(pd.to_numeric, errors="coerce")
    return dat, rois


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------
def roi_tests(dat, rois, n_perm):
    omni, cons = [], []
    for roi in rois:
        samples = [dat.loc[dat.group == g, roi].to_numpy(float) for g in GROUP_ORDER]
        samples = [x[np.isfinite(x)] for x in samples]
        testable = all(x.size >= 2 and np.var(x, ddof=1) > 0 for x in samples)
        if testable:
            F, p = welch_anova_permutation(samples, n_permutations=n_perm,
                                           random_seed=RANDOM_SEED)
            df1, df2 = welch_anova_df(samples)
        else:
            F = p = df1 = df2 = np.nan
        omni.append(dict(ROI=roi, F=F, df1=df1, df2=df2, p_raw=p))
        by = dict(zip(GROUP_ORDER, samples))
        for a, b in PAIRWISE_COMPARISONS:
            xa, xb = by[a], by[b]
            if xa.size < 2 or xb.size < 2 or (np.var(xa, ddof=1) == 0 and np.var(xb, ddof=1) == 0):
                t = df = pc = np.nan
            else:
                t, df, pc = welch_t_test_df(xa, xb)
            cons.append(dict(ROI=roi, Contrast=f"{a} vs {b}", n_a=xa.size, n_b=xb.size,
                             mean_difference=(xa.mean() - xb.mean()) if xa.size and xb.size
                             else np.nan, t=t, df=df, p_raw=pc))
    omni = pd.DataFrame(omni)
    omni["p_adjusted"] = bh(omni.p_raw)                 # BH across ROIs
    cons = pd.DataFrame(cons)
    cons["p_adjusted"] = bh(cons.p_raw)                 # BH across contrasts x ROIs
    return omni, cons


# ------------------------------------------------------------------
# Output tables
# ------------------------------------------------------------------
def summary_table(dat, rois, omni):
    t = pd.DataFrame({"ROI": rois})
    for g in GROUP_ORDER:
        t[g] = [mean_sd(dat.loc[dat.group == g, r]) for r in rois]
    o = omni.set_index("ROI").loc[rois]
    t["Welch F (df1, df2)"] = [
        "NA" if not np.isfinite(F) else f"{F:.2f} ({a:.0f}, {b:.1f})"
        for F, a, b in zip(o.F, o.df1, o.df2)]
    t["Adjusted p-value"] = [format_p(v) for v in o.p_adjusted]
    return t


def contrast_table(cons):
    return pd.DataFrame({
        "ROI": cons.ROI,
        "Contrast": cons.Contrast,
        "Welch t (df)": ["NA" if not np.isfinite(t) else f"{t:.2f} ({d:.1f})"
                         for t, d in zip(cons.t, cons.df)],
        "BH-adjusted p-value": [format_p(v) for v in cons.p_adjusted],
    })


def write_docx(table, path, heading, caption, note):
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt

    doc = Document()
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = Inches(11), Inches(8.5)
    for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, m, Inches(0.6))
    doc.add_heading(heading, level=1)
    cap = doc.add_paragraph()
    cap.add_run(caption).italic = True

    tbl = doc.add_table(rows=1, cols=table.shape[1])
    for j, c in enumerate(table.columns):
        cell = tbl.rows[0].cells[j]
        run = cell.paragraphs[0].add_run(str(c))
        run.bold = True
        run.font.size = Pt(9)
    for _, row in table.iterrows():
        cells = tbl.add_row().cells
        for j, v in enumerate(row):
            run = cells[j].paragraphs[0].add_run(str(v))
            run.font.size = Pt(9)
    left = {"ROI", "Contrast"}
    for row in tbl.rows:
        for j, cell in enumerate(row.cells):
            cell.paragraphs[0].alignment = (WD_ALIGN_PARAGRAPH.LEFT
                                            if table.columns[j] in left
                                            else WD_ALIGN_PARAGRAPH.CENTER)
    # booktabs: rules above and below the table and under the header only
    borders = OxmlElement("w:tblBorders")
    for edge, val in (("top", "single"), ("bottom", "single"), ("left", "nil"),
                      ("right", "nil"), ("insideH", "nil"), ("insideV", "nil")):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), val)
        if val == "single":
            e.set(qn("w:sz"), "8")
            e.set(qn("w:color"), "000000")
        borders.append(e)
    tbl._tbl.tblPr.append(borders)
    for cell in tbl.rows[0].cells:
        tcb = OxmlElement("w:tcBorders")
        e = OxmlElement("w:bottom")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), "6")
        e.set(qn("w:color"), "000000")
        tcb.append(e)
        cell._tc.get_or_add_tcPr().append(tcb)
    doc.add_paragraph(note)
    doc.save(path)


def write_pdf(table, path, title, note, rows_per_page, size):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    n_pages = max(1, int(np.ceil(len(table) / rows_per_page)))
    with PdfPages(path) as pdf:
        for page in range(n_pages):
            part = table.iloc[page * rows_per_page:(page + 1) * rows_per_page]
            fig = plt.figure(figsize=size)
            fig.text(0.5, 0.96, title, ha="center", va="top", fontsize=14, weight="bold")
            ax = fig.add_axes([0.02, 0.07, 0.96, 0.84])
            ax.axis("off")
            tab = ax.table(cellText=part.values, colLabels=list(part.columns),
                           loc="upper center", cellLoc="center")
            tab.auto_set_font_size(False)
            tab.set_fontsize(8)
            tab.auto_set_column_width(list(range(part.shape[1])))
            tab.scale(1.15, 36.0 / rows_per_page)   # fills the page without reaching the note
            for (r, _c), cell in tab.get_celld().items():
                cell.set_edgecolor("white")
                if r == 0:
                    cell.set_text_props(weight="bold")
            fig.text(0.5, 0.025, note, ha="center", fontsize=7)
            pdf.savefig(fig)
            plt.close(fig)


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("csv", nargs="?", default="mice_NodePosStrength_PCorr.csv")
    ap.add_argument("--prefix", default="Strength")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--n_perm", type=int, default=N_PERMUTATIONS)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out = lambda name: os.path.join(args.out_dir, f"{args.prefix}_{name}")  # noqa: E731

    dat, rois = load(args.csv)
    print(f"{len(dat)} subjects, {len(rois)} ROIs; groups: "
          + ", ".join(f"{g} {int((dat.group == g).sum())}" for g in GROUP_ORDER))
    omni, cons = roi_tests(dat, rois, args.n_perm)
    omni.to_csv(out("roi_tests_numeric.csv"), index=False)

    stars = "* p < 0.05; ** p < 0.01; *** p < 0.001."
    summ = summary_table(dat, rois, omni)
    summ.to_csv(out("summary_table.csv"), index=False)
    note_s = ("Values are mean ± SD. Welch ANOVA across groups; p from "
              f"{args.n_perm} permutations of group labels, adjusted across ROIs with "
              "the Benjamini-Hochberg false-discovery-rate procedure. " + stars)
    write_docx(summ, out("summary_table.docx"), "ROI Group Comparison",
               "ROI values by group and Welch ANOVA results", note_s)
    write_pdf(summ, out("summary_table.pdf"), "ROI Group Comparison",
              "Mean ± SD; Welch F (df1, df2); permutation p, BH-adjusted across ROIs. "
              + stars, 20, (14, 8.5))

    ctab = contrast_table(cons)
    ctab.to_csv(out("planned_contrasts_table.csv"), index=False)
    cons.to_csv(out("planned_contrasts_numeric.csv"), index=False)
    note_c = ("Planned contrasts (the 8 used throughout the study): ordinary two-sided "
              "Welch t tests (Welch-Satterthwaite df); p-values adjusted with the "
              "Benjamini-Hochberg procedure pooled across contrasts and ROIs. " + stars)
    write_docx(ctab, out("planned_contrasts_table.docx"), "ROI Planned Contrasts",
               "Planned group contrasts by ROI (Welch t tests)", note_c)
    write_pdf(ctab, out("planned_contrasts_table.pdf"), "ROI Planned Contrasts",
              "Welch t (df); BH-adjusted across contrasts x ROIs. " + stars, 25, (11, 8.5))

    print(f"ROIs with adjusted omnibus p < 0.05: {int((omni.p_adjusted < 0.05).sum())} "
          f"of {omni.p_raw.notna().sum()} tested")
    print(f"Contrasts with BH-adjusted p < 0.05: {int((cons.p_adjusted < 0.05).sum())} "
          f"of {cons.p_raw.notna().sum()} tested")
    print(f"Written to {os.path.abspath(args.out_dir)} (prefix {args.prefix})")


if __name__ == "__main__":
    main()
