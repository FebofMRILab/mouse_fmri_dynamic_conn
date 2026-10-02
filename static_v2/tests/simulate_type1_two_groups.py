# Type I error of permutation vs parametric tests at our group sizes (equal means, unequal variances).
import numpy as np, warnings
from scipy import stats
warnings.filterwarnings("ignore")
rng = np.random.default_rng(5)
n_sim, k = 4000, 999
print(f"{'design':<28}{'pooled perm':>12}{'Welch perm':>12}{'Welch param':>12}{'Student param':>14}")
for dist in ("normal", "lognormal"):
    for (na, nb), r in [((9,5),1),((9,5),3),((9,5),1/3),((19,5),3),((19,5),1/3),((25,19),2)]:
        if dist == "normal":
            xa = rng.normal(0,1,(na,n_sim)); xb = rng.normal(0,r,(nb,n_sim))
        else:
            xa = rng.lognormal(0,0.8,(na,n_sim)); xa -= np.exp(0.32)
            xb = r*(rng.lognormal(0,0.8,(nb,n_sim)) - np.exp(0.32))
        pm = stats.PermutationMethod(n_resamples=k, rng=11)
        out = [np.mean(stats.ttest_ind(xa,xb,axis=0,equal_var=ev,method=m).pvalue < 0.05)
               for ev, m in ((True,pm),(False,pm),(False,None),(True,None))]
        print(f"{dist[:4]} n={na} vs {nb}, SD x{r:.2g}".ljust(28) + "".join(f"{v:>12.3f}" for v in out[:3]) + f"{out[3]:>14.3f}")
