# DCC cross-check on HiPerGator, 4 Oct 2026 (job 44737367, MATLAB R2024b)

`dfc_dcc_subject.m` (ours, base MATLAB) against `dcc_mvgarch.m` (Lindquist DCC toolbox, fmincon), identical input: simulated GARCH(1,1)-DCC(1,1) data with known parameters (5 series, T = 2000) and 5 mice (25 group-PCA components, odd-volume polyphase component, AR(2) residuals). The job log ends with an error in the summary printer (E40); the comparison table was written before it.

| Quantity | Agreement |
|---|---|
| DCC a | max abs difference 0.00064 (2% relative); r = 0.998 across the 5 mice |
| DCC b | max abs difference 0.00060 |
| Simulated truth (a = 0.05, b = 0.90) | ours 0.0488 / 0.9035; published 0.0490 / 0.9035 |
| GARCH alpha, per series | within 0.001 for 87% of series, within 0.005 for 95%; max 0.023 |
| GARCH alpha, mean per mouse (the reported measure) | within 0.001 |
| GARCH beta | differs by > 0.05 in 45 of 130 series; 34 of them have alpha < 0.001 in both fits, where beta is not identified (constant variance). The other 11 have small alpha (< 0.03), where alpha and beta trade off on a flat likelihood; one published fit stopped at its iteration limit (EXITFLAG = 0 in the log). |

**Reading.** For the quantities the paper tests (DCC a; mean GARCH alpha), our implementation reproduces the published estimator to optimiser tolerance. GARCH beta and persistence are diagnostics only, as stated in `dfc_dcc_subject.m`. Because every observed, surrogate and replica series goes through the same estimator, small estimator differences also enter the observed-vs-surrogate comparison equally.
