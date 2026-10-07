# paired comparison of per-truth-feature SD across files (s): corrected variant minus native
import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon
ROOT = Path(__file__).resolve().parents[2]
rows = []
for run in sys.argv[1:]:
    z = np.load(ROOT / f"results/rt_harm/{run}.npz"); nat = z["nat"]; dt = float(z["dt"])
    sd = lambda a: 60 * np.nanstd(a, 1, ddof=1)
    sn = sd(nat)
    rng = np.random.default_rng(0)
    for v in ("gate_off", "gate_on"):
        sc = sd(np.load(ROOT / f"results/rt_harm/{run}__{v}.npy")); d = sc - sn
        bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
        p = wilcoxon(sc, sn).pvalue
        rows.append(dict(run=run, variant=v, n=len(d), dt_s=60 * dt, mean_diff_s=d.mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                         median_diff_s=np.median(d), wilcoxon_p=p, verdict="better" if (np.percentile(bs, 97.5) < 0) else ("worse" if np.percentile(bs, 2.5) > 0 else "no detectable difference")))
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).round(4).to_string(index=False))
