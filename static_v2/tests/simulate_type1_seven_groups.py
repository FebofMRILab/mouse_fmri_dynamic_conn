# Type I error of permutation vs parametric tests at our group sizes (equal means, unequal variances).
import numpy as np, warnings
from scipy import stats
warnings.filterwarnings("ignore")
rng = np.random.default_rng(9)
sizes = [19, 9, 25, 13, 5, 12, 12]
n_sim, k = 2000, 999
def wf(*s, axis=0): return stats.f_oneway(*s, axis=axis, equal_var=False).statistic
def cf(*s, axis=0): return stats.f_oneway(*s, axis=axis).statistic
print(f"{'variance pattern (SD of n=5 group)':<38}{'F perm':>8}{'Welch perm':>12}{'Welch param':>12}{'F param':>9}")
for label, sds in [("all equal", [1]*7), ("n=5 group SD 3", [1,1,1,1,3,1,1]), ("n=5 group SD 0.33", [1,1,1,1,1/3,1,1]), ("mixed 0.5-2", [1,2,0.5,1.5,2,0.7,1])]:
    samples = [rng.normal(0, s, (n, n_sim)) for n, s in zip(sizes, sds)]
    res = []
    for stat in (cf, wf):
        r = stats.permutation_test(samples, stat, permutation_type="independent", vectorized=True, axis=0,
                                   alternative="greater", n_resamples=k, rng=3, batch=50)
        res.append(np.mean(r.pvalue < 0.05))
    res.append(np.mean(stats.f_oneway(*samples, axis=0, equal_var=False).pvalue < 0.05))
    res.append(np.mean(stats.f_oneway(*samples, axis=0).pvalue < 0.05))
    print(f"{label:<38}" + "".join(f"{v:>10.3f}" for v in res))
