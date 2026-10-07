# where does RT correction hurt? per truth feature x file cell: |dev from median of the other files| corrected
# minus native (s), split by whether the cell lies outside the file's matched-anchor range (constant-shift
# extrapolation) or inside; plus a cross-fitted "oracle" smooth correction fitted on the truth itself
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
pd.set_option("display.width", 250)

def oracle(nat, trt, nb=12):
    """per file: binned-median truth drift (fit on even features, applied to odd and vice versa)"""
    N = nat.shape[1]; out = nat.copy()
    idx = np.argsort(np.nanmedian(nat, 1)); fold = np.zeros(len(nat), int); fold[idx[1::2]] = 1
    for j in range(N):
        oth = np.delete(np.arange(N), j)
        dev = nat[:, j] - np.nanmedian(nat[:, oth], 1)
        for f in (0, 1):
            tr = (fold != f) & np.isfinite(dev); te = (fold == f) & np.isfinite(dev)
            x = nat[tr, j]; e = np.quantile(x, np.linspace(0, 1, nb + 1))
            kx = np.array([np.median(x[(x >= e[k]) & (x <= e[k + 1])]) for k in range(nb)])
            ky = np.array([np.median(dev[tr][(x >= e[k]) & (x <= e[k + 1])]) for k in range(nb)])
            out[te, j] = nat[te, j] - np.interp(nat[te, j], kx, ky)
    return out

for run in sys.argv[1:]:
    z = np.load(ROOT / f"results/rt_harm/{run}.npz"); stems = list(z["stems"]); nat, cor = z["nat"], z["cor"]; dt = float(z["dt"])
    A = pd.read_csv(ROOT / f"results/peak3d/{run}/peak3d_out/rt_correction/anchors.tsv", sep="\t")
    N = len(stems); rows = []
    orc = oracle(nat, z["trt"])
    for j, s in enumerate(stems):
        oth = np.delete(np.arange(N), j)
        dn = 60 * (nat[:, j] - np.nanmedian(nat[:, oth], 1)); dc = 60 * (cor[:, j] - np.nanmedian(cor[:, oth], 1))
        do = 60 * (orc[:, j] - np.nanmedian(orc[:, oth], 1))
        a = A[A.file == s].rt_native.values
        lo, hi = (a.min(), a.max()) if len(a) else (np.inf, -np.inf)
        m = np.isfinite(dn); edge = (nat[:, j] < lo) | (nat[:, j] > hi)
        for nm, sel in (("edge", m & edge), ("interior", m & ~edge)):
            h = np.abs(dc[sel]) - np.abs(dn[sel]); ho = np.abs(do[sel]) - np.abs(dn[sel])
            rows.append(dict(file=s[-10:], region=nm, n=int(sel.sum()), absdev_nat=np.mean(np.abs(dn[sel])), absdev_cor=np.mean(np.abs(dc[sel])),
                             absdev_oracle=np.mean(np.abs(do[sel])), worse_gt_scan=np.mean(h > 60 * dt), better_gt_scan=np.mean(h < -60 * dt),
                             oracle_worse_gt_scan=np.mean(ho > 60 * dt), anc_lo=lo, anc_hi=hi))
    R = pd.DataFrame(rows)
    print(f"=== {run}  dt {60*dt:.2f} s"); print(R.round(2).to_string(index=False))
    # whole-run per-feature SD: corrected vs oracle
    sd = lambda a: 60 * np.nanstd(a, 1, ddof=1)
    sn, sc, so = sd(nat), sd(cor), sd(orc)
    print(f"  SD median nat {np.median(sn):.2f} cor {np.median(sc):.2f} oracle {np.median(so):.2f} | widened >0.25s: cor {(sc - sn > .25).mean():.1%} oracle {(so - sn > .25).mean():.1%}"
          f" | widened >0: cor {(sc > sn).mean():.1%} oracle {(so > sn).mean():.1%}")
