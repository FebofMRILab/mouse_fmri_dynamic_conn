# %% [markdown]
# # Nodal graph metrics: publication tables (mouse)
#
# Metrics: node strength, nodal clustering coefficient, nodal efficiency.
# Statistics match `full_analysis_abeta_mice.py`:
#
# * **Omnibus test per ROI:** Welch ANOVA across the 7 groups, BH-FDR across ROIs.
# * **Planned comparisons:** the same 8 Welch t-tests as the edgewise/NBS analyses,
#   with one pooled BH-FDR family across all comparisons × ROIs (not gated on the omnibus).
# * **Correction family:** each metric is corrected separately.
# * **Input:** `node_metrics_by_density.csv` from the main script, at the Figure 7 density.
# * **Exports:** formatted tables (CSV/DOCX/PDF) list only significant results:
#   - summary table: ROIs with q < 0.05 in the omnibus test *or* in at least one planned contrast
#   - contrasts table: planned contrasts with q < 0.05
#   Metrics with no significant result get no formatted tables. Complete numeric results
#   are always saved as CSV (use these as supplementary data).

# %% Imports
import os
import textwrap
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Inches
from welch_tests_scipy import welch_anova_edgewise_permutation, welch_t_edgewise
from welch_tests_scipy import welch_t_test_df

# %% Configuration
INPUT_CSV = "fc_analysis_mice/node_metrics_by_density.csv"
DENSITY = 0.15                   # must match strength_density used for Figure 7
OUT = "fc_analysis_mice/tables"
ALPHA = 0.05
EXPORT_FULL_TABLES = False       # True also writes formatted tables with every ROI/contrast

N_PERMUTATIONS = 10000
RANDOM_SEED = 20260911

# column in INPUT_CSV -> (file prefix, label for captions, decimals)
METRICS = {
    "node_strength":   ("Strength",   "node strength",                 2),
    "node_clustering": ("Clustering", "nodal clustering coefficient",  3),
    "node_efficiency": ("Efficiency", "nodal efficiency",              3),
}

GROUP_ORDER = ["NTG_Y", "NTG_M", "NTG_A", "CRND8_Y", "CRND8_M", "5XFAD_Y", "5XFAD_M"]

PAIRWISE_COMPARISONS = [
    ("NTG_Y", "NTG_M"),
    ("NTG_Y", "NTG_A"),
    ("NTG_Y", "CRND8_Y"),
    ("NTG_M", "CRND8_M"),
    ("NTG_Y", "5XFAD_Y"),
    ("NTG_M", "5XFAD_M"),
    ("CRND8_Y", "CRND8_M"),
    ("5XFAD_Y", "5XFAD_M"),
]

os.makedirs(OUT, exist_ok=True)

# %% Load data at the chosen density
long_df = pd.read_csv(INPUT_CSV)
long_df["genotype"] = long_df["genotype"].astype(str).str.strip()

missing_cols = [m for m in METRICS if m not in long_df.columns]
if missing_cols:
    raise ValueError(f"{INPUT_CSV} is missing metric columns: {missing_cols}")

selected = long_df.loc[np.isclose(long_df["density"], DENSITY)].copy()
if selected.empty:
    raise ValueError(
        f"Density {DENSITY} not in {INPUT_CSV}; "
        f"available: {sorted(long_df['density'].unique())}"
    )

unexpected = sorted(set(selected["genotype"]) - set(GROUP_ORDER))
if unexpected:
    raise ValueError(f"Unexpected group labels: {unexpected}")

if selected.duplicated(["subject_index", "roi_index"]).any():
    raise ValueError("Duplicate subject x ROI rows at this density.")

roi_order = (
    selected.drop_duplicates("roi_index")
    .sort_values("roi_index")["roi_name"]
    .tolist()
)

subject_groups = (
    selected.drop_duplicates("subject_index")
    .set_index("subject_index")["genotype"]
    .sort_index()
)
group_n = subject_groups.value_counts().reindex(GROUP_ORDER, fill_value=0)

N_ROI = len(roi_order)
print(f"Density: {DENSITY}; subjects: {len(subject_groups)}; ROIs: {N_ROI}")
print(group_n.to_string())

# Isolated nodes (degree 0) get clustering = 0 and efficiency = 0 by convention.
# Report how common they are so group differences are not driven by them.
if "node_degree" in selected.columns:
    isolated = (
        selected.assign(isolated=selected["node_degree"] == 0)
        .groupby("genotype", observed=False)["isolated"].mean()
        .reindex(GROUP_ORDER)
    )
    print("\nFraction of isolated nodes by group:")
    print(isolated.round(4).to_string())


def metric_matrix(metric):
    """Subjects x ROIs matrix for one metric, in ROI order."""
    wide = selected.pivot(index="subject_index", columns="roi_name", values=metric)
    wide = wide[roi_order].reindex(subject_groups.index)
    if wide.isna().any().any():
        raise ValueError(f"Missing {metric} values at this density.")
    return wide


# %% Statistical helpers
def welch_contrast(a, b):
    """
    Welch t-test for a - b, with the Welch 95% CI of the mean difference
    and Hedges' g (pooled-SD standardizer, small-sample corrected).

    The p_raw returned here is from the t distribution; analyze_metric
    overwrites it with the permutation p-value. The t, df, CI and g are kept.
    """
    n1, n2 = len(a), len(b)
    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    diff = m1 - m2

    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.sqrt(v1 / n1 + v2 / n2)
        df = (v1 / n1 + v2 / n2) ** 2 / (
            (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        )
        t = diff / se
        p = 2 * stats.t.sf(abs(t), df)
        half_width = stats.t.ppf(0.975, df) * se

        pooled_sd = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
        g = (1 - 3 / (4 * (n1 + n2) - 9)) * diff / pooled_sd

    return {
        "t": t, "df": df, "p_raw": p,
        "mean_difference": diff,
        "ci_low": diff - half_width, "ci_high": diff + half_width,
        "hedges_g": g,
    }

def bh_fdr(p_values):
    """BH-FDR over the tests that could be computed; untestable tests stay NaN."""
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full_like(p_values, np.nan)
    testable = np.isfinite(p_values)
    if testable.any():
        q_values[testable] = multipletests(p_values[testable], alpha=ALPHA, method="fdr_bh")[1]
    return q_values


def format_p(p):
    if pd.isna(p):
        return "NA"
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < ALPHA else ""
    return ("<0.001" if p < 0.001 else f"{p:.3f}") + stars


def mean_sd(x, decimals):
    if len(x) == 0:
        return "NA"
    if len(x) == 1:
        return f"{x.mean():.{decimals}f} ± NA"
    return f"{x.mean():.{decimals}f} ± {x.std(ddof=1):.{decimals}f}"


def fmt(x, decimals):
    return "NA" if pd.isna(x) else f"{x:.{decimals}f}"


# %% Run the statistics for one metric
def analyze_metric(metric):
    wide = metric_matrix(metric)
    values = wide.to_numpy(dtype=float)
    masks = [(subject_groups == g).to_numpy() for g in GROUP_ORDER]

    def values_for(group, roi):
        x = wide.loc[subject_groups == group, roi].to_numpy(dtype=float)
        return x[np.isfinite(x)]

    # Omnibus: all ROIs in one call, one set of relabelings shared across ROIs
    F_values, p_values = welch_anova_edgewise_permutation(
        values, masks, n_permutations=N_PERMUTATIONS, random_seed=RANDOM_SEED
    )

    omnibus = pd.DataFrame({
        "ROI": roi_order,
        "F": F_values,
        "df1": len(GROUP_ORDER) - 1,
        "p_raw": p_values,
    })
    omnibus["q_FDR"] = bh_fdr(omnibus["p_raw"])

    # Planned comparisons: permutation p-values, parametric effect sizes
    contrast_rows = []
    for group_a, group_b in PAIRWISE_COMPARISONS:
        a_mask = (subject_groups == group_a).to_numpy()
        b_mask = (subject_groups == group_b).to_numpy()

        _, contrast_p = welch_t_edgewise(
            values[a_mask], values[b_mask],
        )

        for roi_index, roi in enumerate(roi_order):
            result = welch_contrast(values_for(group_a, roi), values_for(group_b, roi))
            result["p_raw"] = contrast_p[roi_index]      # permutation p replaces the t-distribution p
            contrast_rows.append({
                "ROI": roi, "Contrast": f"{group_a} vs {group_b}", **result
            })

        contrasts = pd.DataFrame(contrast_rows)
        contrasts["q_FDR_pooled"] = bh_fdr(contrasts["p_raw"])

        contrasts["ROI"] = pd.Categorical(contrasts["ROI"], categories=roi_order, ordered=True)
        contrasts["Contrast"] = pd.Categorical(
        contrasts["Contrast"],
        categories=[f"{a} vs {b}" for a, b in PAIRWISE_COMPARISONS],
        ordered=True,
    )
    contrasts = contrasts.sort_values(["ROI", "Contrast"]).reset_index(drop=True)
    contrasts[["ROI", "Contrast"]] = contrasts[["ROI", "Contrast"]].astype(str)

    return omnibus, contrasts, values_for

# %% Build formatted tables for one metric
def build_tables(omnibus, contrasts, values_for, decimals):
    table_omnibus = pd.DataFrame({"ROI": roi_order})
    for g in GROUP_ORDER:
        table_omnibus[f"{g} (n={group_n[g]})"] = [
            mean_sd(values_for(g, roi), decimals) for roi in roi_order
        ]
    table_omnibus["Welch F"] = [
        "not testable†" if pd.isna(r.F) else f"{r.F:.2f}" for r in omnibus.itertuples()
    ]

    table_omnibus["q (BH)"] = omnibus["q_FDR"].map(format_p).to_numpy()

    table_contrasts = pd.DataFrame({
        "ROI": contrasts["ROI"],
        "Contrast": contrasts["Contrast"],
        "Mean difference [95% CI]": [
            f"{fmt(r.mean_difference, decimals)} "
            f"[{fmt(r.ci_low, decimals)}, {fmt(r.ci_high, decimals)}]"
            for r in contrasts.itertuples()
        ],
        "Welch t (df)": [f"{fmt(r.t, 2)} ({fmt(r.df, 1)})" for r in contrasts.itertuples()],
        "Hedges' g": contrasts["hedges_g"].map(lambda g: fmt(g, 2)),
        "p": contrasts["p_raw"].map(format_p),
        "q (pooled BH)": contrasts["q_FDR_pooled"].map(format_p),
    })

    # Which planned contrasts are significant for each ROI.
    significant_contrasts = contrasts["q_FDR_pooled"] < ALPHA
    contrasts_by_roi = (
        contrasts.loc[significant_contrasts]
        .groupby("ROI", sort=False)["Contrast"]
        .apply(lambda c: "; ".join(c))
    )
    table_omnibus["Significant planned contrasts"] = (
        table_omnibus["ROI"].map(contrasts_by_roi).fillna("—")
    )

    # ROIs significant in the omnibus test or in at least one planned contrast.
    roi_keep = (
        (omnibus["q_FDR"] < ALPHA).to_numpy()
        | table_omnibus["ROI"].isin(contrasts_by_roi.index).to_numpy()
    )

    return (
        table_omnibus,
        table_contrasts,
        table_omnibus.loc[roi_keep].reset_index(drop=True),
        table_contrasts.loc[significant_contrasts.to_numpy()].reset_index(drop=True),
    )


# %% Export helpers
def export_docx(table, path, title, caption, note, font_pt=8):
    doc = Document()
    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = section.page_height, section.page_width
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(section, side, Inches(0.5))

    doc.add_heading(title, level=1)
    cap = doc.add_paragraph(caption)
    cap.runs[0].italic = True

    word_table = doc.add_table(rows=1, cols=table.shape[1])
    word_table.style = "Table Grid"

    for cell, header in zip(word_table.rows[0].cells, table.columns):
        cell.text = str(header)

    for row in table.itertuples(index=False):
        for cell, value in zip(word_table.add_row().cells, row):
            cell.text = str(value)

    for r_idx, row in enumerate(word_table.rows):
        for c_idx, cell in enumerate(row.cells):
            for paragraph in cell.paragraphs:
                paragraph.alignment = (
                    WD_ALIGN_PARAGRAPH.LEFT
                    if table.columns[c_idx] in ("ROI", "Contrast")
                    else WD_ALIGN_PARAGRAPH.CENTER
                )
                for run in paragraph.runs:
                    run.font.size = Pt(font_pt)
                    run.bold = r_idx == 0

    # Repeat the header row on each page.
    header_flag = OxmlElement("w:tblHeader")
    header_flag.set(qn("w:val"), "true")
    word_table.rows[0]._tr.get_or_add_trPr().append(header_flag)

    footnote = doc.add_paragraph(note)
    footnote.runs[0].font.size = Pt(8)
    doc.save(path)


def export_pdf(table, path, title, note, rows_per_page=25, font_pt=7, page_size=(14, 8.5)):
    with PdfPages(path) as pdf:
        n_pages = max(1, int(np.ceil(len(table) / rows_per_page)))
        for page in range(n_pages):
            chunk = table.iloc[page * rows_per_page:(page + 1) * rows_per_page]

            fig, ax = plt.subplots(figsize=page_size)
            ax.axis("off")
            fig.suptitle(f"{title} (page {page + 1}/{n_pages})", fontsize=13, fontweight="bold")

            mpl_table = ax.table(
                cellText=chunk.to_numpy(),
                colLabels=list(chunk.columns),
                loc="upper center",
                cellLoc="center",
            )
            mpl_table.auto_set_font_size(False)
            mpl_table.set_fontsize(font_pt)
            mpl_table.auto_set_column_width(range(chunk.shape[1]))
            mpl_table.scale(1, 1.4)
            for (row, _), cell in mpl_table.get_celld().items():
                cell.set_edgecolor("0.8")
                if row == 0:
                    cell.set_text_props(fontweight="bold")

            fig.text(0.5, 0.02, textwrap.fill(note, 160), ha="center", fontsize=7)
            pdf.savefig(fig)
            plt.close(fig)


def export_all(table, stem, title, caption, note, **pdf_kwargs):
    base = os.path.join(OUT, stem)
    table.to_csv(base + ".csv", index=False)
    export_docx(table, base + ".docx", title, caption, note)
    export_pdf(table, base + ".pdf", title, note, **pdf_kwargs)
    print(f"  Saved: {base}.csv / .docx / .pdf")


# %% Run all metrics
summary_rows = []

for metric, (prefix, label, decimals) in METRICS.items():
    print(f"\n[{label}]")
    omnibus, contrasts, values_for = analyze_metric(metric)
    tag = f"{prefix}_density{DENSITY:.2f}"

    # Numeric results are always saved, significant or not.
    omnibus.to_csv(os.path.join(OUT, f"{tag}_omnibus_numeric.csv"), index=False)
    contrasts.to_csv(os.path.join(OUT, f"{tag}_contrasts_numeric.csv"), index=False)

    n_omnibus = int((omnibus["q_FDR"] < ALPHA).sum())
    n_contrasts = int((contrasts["q_FDR_pooled"] < ALPHA).sum())
    n_omnibus_tested = int(omnibus["p_raw"].notna().sum())
    n_contrasts_tested = int(contrasts["p_raw"].notna().sum())

    untestable = omnibus.loc[omnibus["p_raw"].isna(), "ROI"].tolist()
    if untestable:
        print(f"  Omnibus not testable (a group with zero variance): {untestable}")
    if n_contrasts_tested < len(contrasts):
        print(f"  {len(contrasts) - n_contrasts_tested} planned contrasts not testable")
    summary_rows.append({
        "metric": label,
        "ROIs significant (omnibus)": f"{n_omnibus} / {n_omnibus_tested} tested",
        "contrasts significant (pooled BH)": f"{n_contrasts} / {n_contrasts_tested} tested",
        "min omnibus q": omnibus["q_FDR"].min(),
        "min contrast q": contrasts["q_FDR_pooled"].min(),
    })
    print(
        f"  Omnibus: {n_omnibus} / {n_omnibus_tested} ROIs; "
        f"contrasts: {n_contrasts} / {n_contrasts_tested}"
    )

    if n_omnibus == 0 and n_contrasts == 0:
        print(f"  No {label} result survived correction; formatted tables not written.")
        continue

    (
        table_omnibus_full,
        table_contrasts_full,
        table_omnibus_significant,
        table_contrasts_significant,
    ) = build_tables(omnibus, contrasts, values_for, decimals)

    note_omnibus = (
        f"Values are mean ± SD of {label}. Omnibus tests are Welch ANOVAs across "
        f"{len(GROUP_ORDER)} groups, with p-values obtained by permuting group labels "
        f"{N_PERMUTATIONS:,} times; q-values are Benjamini–Hochberg FDR-adjusted "
        f"across the {n_omnibus_tested} testable ROIs. † Not testable: at least one "
        "group had zero variance, so the Welch statistic is undefined; excluded from "
        "the FDR family. * q < 0.05; ** q < 0.01; *** q < 0.001."
    )
    selection_omnibus = (
        " Only ROIs with q < 0.05 in the omnibus test or in at least one planned "
        "contrast are shown; complete results are provided as supplementary data."
    )
    note_contrasts = (
        f"Welch t-tests of {label} for the planned group comparisons (group listed first "
        f"minus group listed second), with p-values obtained by permuting group labels "
        f"{N_PERMUTATIONS:,} times. 95% CIs use Welch degrees of freedom and Hedges' g "
        "uses the pooled SD, so both are parametric. q-values are Benjamini–Hochberg "
        f"FDR-adjusted in one family across all {len(PAIRWISE_COMPARISONS)} comparisons "
        f"× {N_ROI} ROIs ({n_contrasts_tested} testable tests; untestable tests shown "
        "as NA). * q < 0.05; ** q < 0.01; *** q < 0.001."
    )
    selection_contrasts = (
        " Only contrasts with q < 0.05 are shown; complete results are provided "
        "as supplementary data."
    )
    caption_omnibus = f"{label.capitalize()} by group (proportional density {DENSITY:.2f})"
    caption_contrasts = f"Planned pairwise comparisons of {label}"

    export_all(
        table_omnibus_significant, f"{tag}_summary_table_significant",
        f"ROI Group Comparison: {label}", caption_omnibus,
        note_omnibus + selection_omnibus,
        rows_per_page=20,
    )
    if n_contrasts > 0:
        export_all(
            table_contrasts_significant, f"{tag}_planned_contrasts_significant",
            f"Planned Pairwise Comparisons: {label} (q < 0.05)",
            caption_contrasts, note_contrasts + selection_contrasts,
            font_pt=8, page_size=(11, 8.5),
        )

    if EXPORT_FULL_TABLES:
        export_all(
            table_omnibus_full, f"{tag}_summary_table_all",
            f"ROI Group Comparison: {label} (all ROIs)", caption_omnibus, note_omnibus,
            rows_per_page=20,
        )
        export_all(
            table_contrasts_full, f"{tag}_planned_contrasts_all",
            f"Planned Pairwise Comparisons: {label} (all tests)",
            caption_contrasts, note_contrasts,
            font_pt=8, page_size=(11, 8.5),
        )

# %% Overview across metrics (report nulls in the text from this)
overview = pd.DataFrame(summary_rows)
overview.to_csv(os.path.join(OUT, f"nodal_metrics_overview_density{DENSITY:.2f}.csv"), index=False)
overview