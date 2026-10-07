import sys, numpy as np, pandas as pd
sys.path.insert(0, "/quobyte/metabolomicsgrp/fanzhou/hoffmann/bench/diag")
from rt_external_validation import collect
from pathlib import Path
ROOT = Path("/quobyte/metabolomicsgrp/fanzhou/hoffmann")
out = Path(sys.argv[1])
for run in sys.argv[2:]:
    tr, stems, dt, nat, cor = collect(run)
    N = len(stems)
    need = N if N <= 3 else max(3, int(np.ceil(0.5 * N)))
    ok = np.isfinite(nat).sum(axis=1) >= need
    nat, cor, trr = nat[ok], cor[ok], tr.rt.values[ok]
    np.savez(out / f"{run}.npz", nat=nat, cor=cor, trt=trr, stems=np.array(stems), dt=dt)
    sd_n = 60 * np.nanstd(nat, axis=1, ddof=1); sd_c = 60 * np.nanstd(cor, axis=1, ddof=1)
    rg_n = 60 * (np.nanmax(nat, 1) - np.nanmin(nat, 1)); rg_c = 60 * (np.nanmax(cor, 1) - np.nanmin(cor, 1))
    d = sd_c - sd_n
    ds = 60 * dt
    print(f"== {run}: {ok.sum()} truth, N={N}, dt {ds:.2f} s")
    print(f"  SD   native med {np.median(sd_n):.2f} mean {np.mean(sd_n):.2f} | corr med {np.median(sd_c):.2f} mean {np.mean(sd_c):.2f}")
    print(f"  range native med {np.median(rg_n):.2f} | corr med {np.median(rg_c):.2f}")
    for thr in (0, 0.1, 0.25, 0.5, ds):
        print(f"  SD widened by > {thr:.2f} s: {(d > thr).mean():.1%}   tightened by > {thr:.2f}: {(d < -thr).mean():.1%}")
    # per-file deviation from the median of the OTHER files (native vs corrected), binned by RT
    for j, s in enumerate(stems):
        oth = np.delete(np.arange(N), j)
        dn = 60 * (nat[:, j] - np.nanmedian(nat[:, oth], axis=1))
        dc = 60 * (cor[:, j] - np.nanmedian(cor[:, oth], axis=1))
        m = np.isfinite(dn)
        x = trr[m]
        edges = np.quantile(x, np.linspace(0, 1, 9))
        b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, 7)
        row = []
        for k in range(8):
            q = b == k
            row.append(f"{edges[k]:5.1f}:{np.median(dn[m][q]):+5.2f}/{np.median(dc[m][q]):+5.2f}")
        print(f"  {s[-12:]:>12s} |dev| med {np.median(np.abs(dn[m])):.2f}->{np.median(np.abs(dc[m])):.2f}  " + " ".join(row))
