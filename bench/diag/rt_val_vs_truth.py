# LI2018-type check: native deviation (file minus median of the other files, s) of the gate's validation
# features vs the truth features, per file and RT window, against the anchor-fitted shift
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench/diag"))
from rt_gate_eval import load
from peak3d.align import AlignParams, align_run, _validation_set, _loo_median
run = sys.argv[1]; files = sys.argv[2].split(","); wins = [tuple(map(float, w.split("-"))) for w in sys.argv[3].split(",")]
stems, feats, grids, nat, cor, dt = load(run)
res = align_run({s: f.copy() for s, f in feats.items()}, grids, AlignParams(gate=False))
P = AlignParams(); dtm = float(np.median([np.median(np.diff(g)) for g in grids.values()]))
# validation membership as the gate builds it, but deviations on the NATIVE axes
val = _validation_set(res.files, feats, stems, P, dtm)
gids = np.concatenate([val[s][1] for s in stems]); xs = np.concatenate([val[s][0] for s in stems])
ref = _loo_median(gids, xs); off = np.cumsum([0] + [len(val[s][0]) for s in stems])
bins = np.arange(-8, 8.01, 1.0)
for f in files:
    k = stems.index(f); x = val[f][0]; d = 60 * (x - ref[off[k]:off[k + 1]])
    oth = np.delete(np.arange(len(stems)), k); td = 60 * (nat[:, k] - np.nanmedian(nat[:, oth], 1)); tx = nat[:, k]
    w = res.files[f].warp
    for a, b in wins:
        s1 = (x >= a) & (x < b); s2 = np.isfinite(td) & (tx >= a) & (tx < b)
        sh = -60 * np.median(w.shift(x[s1])) if s1.any() else np.nan
        hv = np.histogram(np.clip(d[s1], -7.99, 7.99), bins)[0]; ht = np.histogram(np.clip(td[s2], -7.99, 7.99), bins)[0]
        print(f"{f} {a:4.1f}-{b:4.1f} min: anchor fit says native dev = {sh:+.2f} s | validation n={s1.sum():4d} med {np.median(d[s1]):+.2f} | truth n={s2.sum():3d} med {np.median(td[s2]) if s2.any() else np.nan:+.2f}")
        print("    bins(s) " + " ".join(f"{v:+4.0f}" for v in bins[:-1]))
        print("    valid % " + " ".join(f"{v:4.0f}" for v in 100 * hv / max(1, hv.sum())))
        print("    truth % " + " ".join(f"{v:4.0f}" for v in 100 * ht / max(1, ht.sum())))
