# Is the truth "harm" real drift or an isomer-matching artifact? Split truth features by whether the matched
# feature is unambiguous in every file (no other feature within 5 ppm and +/- 0.25 min of it) and compare
# native vs corrected spread per class (published corrected axis).
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench/diag"))
from apex_spread import truth_table
for run in sys.argv[1:]:
    d = ROOT / "results/peak3d" / run / "peak3d_out"
    tr = truth_table(run).reset_index(drop=True)
    stems = sorted(p.stem for p in (d / "features").glob("*.tsv") if not p.name.endswith(("manifest.tsv", "rejected.tsv")))
    dt = float(np.load(ROOT / f"results/rt_harm/{run}.npz")["dt"])
    nat = np.full((len(tr), len(stems)), np.nan); cor = nat.copy(); amb = np.zeros((len(tr), len(stems)), bool)
    # also: another truth compound with the same m/z (5 ppm) within 0.5 min
    o = np.argsort(tr.mz.values); smz = tr.mz.values[o]; srt = tr.rt.values[o]
    tamb = np.zeros(len(tr), bool)
    for i, (m, t) in enumerate(zip(tr.mz.values, tr.rt.values)):
        lo = np.searchsorted(smz, m * (1 - 5e-6)); hi = np.searchsorted(smz, m * (1 + 5e-6), side="right")
        tamb[i] = (np.abs(srt[lo:hi] - t) <= 0.5).sum() > 1
    for j, s in enumerate(stems):
        f = pd.read_csv(d / "features" / f"{s}.tsv", sep="\t", usecols=["mz", "rt", "rt_corr", "height"])
        o2 = np.argsort(f.mz.values); fmz = f.mz.values[o2]
        for i, (m, t) in enumerate(zip(tr.mz.values, tr.rt.values)):
            lo = np.searchsorted(fmz, m * (1 - 5e-6)); hi = np.searchsorted(fmz, m * (1 + 5e-6), side="right")
            k = o2[lo:hi]; kk = k[np.abs(f.rt.values[k] - t) <= 0.15]
            if len(kk):
                b = kk[np.argmax(f.height.values[kk])]
                nat[i, j], cor[i, j] = f.rt.values[b], f.rt_corr.values[b]
                amb[i, j] = (np.abs(f.rt.values[k] - f.rt.values[b]) <= 0.25).sum() > 1
    N = len(stems); need = N if N <= 3 else max(3, int(np.ceil(0.5 * N)))
    ok = np.isfinite(nat).sum(1) >= need
    clean = ok & ~tamb & ~(amb & np.isfinite(nat)).any(1)
    sd = lambda a: 60 * np.nanstd(a, 1, ddof=1)
    print(f"=== {run} dt {60*dt:.2f} s")
    for nm, sel in (("all", ok), ("unambiguous", clean), ("ambiguous", ok & ~clean)):
        sn, sc = sd(nat[sel]), sd(cor[sel]); dd = sc - sn
        print(f"  {nm:12s} n={sel.sum():4d}  SD med native {np.median(sn):.2f} -> corr {np.median(sc):.2f}  mean {np.mean(sn):.2f} -> {np.mean(sc):.2f} | "
              f"widened >0.25 s {np.mean(dd > .25):.1%}, > 1 scan {np.mean(dd > 60*dt):.1%} | tightened >0.25 s {np.mean(dd < -.25):.1%}, > 1 scan {np.mean(dd < -60*dt):.1%}")
    np.savez(ROOT / f"results/rt_harm/{run}_clean.npz", clean=clean[ok])
