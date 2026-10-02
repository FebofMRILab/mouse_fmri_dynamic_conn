#!/usr/bin/env python3
"""
dfc_group_calibrated.py
Group differences for the two quantities plot_dfc_results.py does not test:
  1. the calibrated per-mouse value d = z_observed - z_replica (DCC, GARCH,
     cPCA envelope; the same 9 measures as Fig_dfc_group_z), and
  2. the transient measures from dfc_confound_checks.py (burst_z, kurt_gs,
     kurt_pc, fc_change_z).
No new statistics: it calls the study's own group_tests() from
plot_dfc_results.py (Welch ANOVA across the 7 groups, 10,000 label
permutations, Holm across measures; the 8 planned contrasts as ordinary
Welch t tests with pooled BH-FDR; both from welch_tests_scipy.py).

Run from the folder that holds plot_dfc_results.py:
    python dfc_group_calibrated.py dfc_dcc_cpca
Writes table_group_{omnibus,contrasts}_{calibrated,transients}.csv into that folder.
"""
import sys
import plot_dfc_results as P          # the study's own loader and group test

in_dir = sys.argv[1] if len(sys.argv) > 1 else "dfc_dcc_cpca"
d = P.load_all(in_dir)["sub"]
d = d[d.role == "test"].copy()
d["d_cal"] = d.phase_z - d.phase_replica_z    # per-mouse calibrated value (as in the calibrated test)
metrics = [m for row in P.GROUP_Z_GRID for m in row if m in set(d.metric)]
groups = P.group_order(d.group)
omni, cons = P.group_tests(d, metrics, "d_cal", groups, "calibrated_d_by_group", P.N_PERMUTATIONS)
omni.to_csv(f"{in_dir}/table_group_omnibus_calibrated.csv", index=False)
cons.to_csv(f"{in_dir}/table_group_contrasts_calibrated.csv", index=False)
print(omni[["metric", "n", "welch_F", "p_perm", "p_Holm"]].round(4).to_string(index=False))
print("contrasts with q_BH < 0.05:", int((cons.q_BH < 0.05).sum()), "of", len(cons))

# Transient / burst measures (dfc_confound_checks.py output), same group test
import os
f = os.path.join(in_dir, "confound_checks", "confound_measures_per_mouse.csv")
if os.path.isfile(f):
    c = P.pd.read_csv(f)
    ms = [m for m in ["burst_z", "kurt_gs", "kurt_pc", "fc_change_z"] if m in c]
    long = c.melt(id_vars=["subject_id", "group"], value_vars=ms, var_name="metric", value_name="v")
    long["group"] = long.group.astype(str).str.strip()
    o2, c2 = P.group_tests(long, ms, "v", P.group_order(long.group), "transients_by_group", P.N_PERMUTATIONS)
    o2.to_csv(f"{in_dir}/table_group_omnibus_transients.csv", index=False)
    c2.to_csv(f"{in_dir}/table_group_contrasts_transients.csv", index=False)
    print(o2[["metric", "n", "welch_F", "p_perm", "p_Holm"]].round(4).to_string(index=False))
    print("contrasts with q_BH < 0.05:", int((c2.q_BH < 0.05).sum()), "of", len(c2))
