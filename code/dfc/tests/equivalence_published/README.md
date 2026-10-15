# Equivalence with published reference code

| Test | Published code | Result here (Octave 8.4, 4 Oct 2026) |
|---|---|---|
| `test_phase_randomise_vs_cbig.m` | `CBIG_RL2017_get_PR_surrogate.m` (Liégeois et al. 2017) | identical (max difference 0) |
| `make_cpca_reference.py` + `test_cpca_vs_published.m` | `cpca.py` (Bolt et al. 2022, tsb46/complex_pca) | identical to machine precision with the Hilbert transform per scan (1 − \|⟨v, v′⟩\| ≤ 4e-16); 0.978–0.99996 when the published script's concatenate-then-Hilbert order is used |

**The one deviation.** The published `cpca.py` concatenates all scans and then takes the Hilbert transform, which joins the end of one scan to the start of the next. `dfc_analytic_signal.m` takes it per scan. Everything else (conjugation, centring, decomposition) gives the same components.

**DCC/GARCH: matches the published estimator (HiPerGator, 4 Oct 2026).** `compare_dcc_with_lindquist.m` compares `dfc_dcc_subject.m` with `dcc_mvgarch.m` from the Lindquist DCC toolbox (github.com/canlab/Lindquist_Dynamic_Correlation) on identical input. DCC a and b agreed within 0.0006 in 5 mice and in simulated data with known parameters; the mean GARCH alpha per mouse within 0.001. GARCH beta differs only where it is weakly or not identified (alpha near 0). Details and the raw table: `results_hipergator_2026-10-04/`. Job: `hipergator_tests/run_dcc_crosscheck.sbatch`.

```bash
python make_cpca_reference.py cpca_reference
octave --eval "test_cpca_vs_published('cpca_reference')"
octave --eval "test_phase_randomise_vs_cbig('/path/to/CBIG/stable_projects/fMRI_dynamics/Liegeois2017_Surrogates')"
```
