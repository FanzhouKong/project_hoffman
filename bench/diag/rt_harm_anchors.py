# per file and RT bin: what the matched anchors say (residual_before = consensus - native, s) vs what the truth says
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
run = sys.argv[1]; files = sys.argv[2].split(","); bw = float(sys.argv[3])
A = pd.read_csv(ROOT / f"results/peak3d/{run}/peak3d_out/rt_correction/anchors.tsv", sep="\t")
z = np.load(ROOT / f"results/rt_harm/{run}.npz"); stems = list(z["stems"]); nat, cor = z["nat"], z["cor"]
pd.set_option("display.width", 250)
for f in files:
    j = stems.index(f); oth = np.delete(np.arange(len(stems)), j)
    tsh = 60 * (np.nanmedian(nat[:, oth], 1) - nat[:, j]); tx = nat[:, j]
    a = A[A.file == f]
    w = pd.read_csv(ROOT / f"results/peak3d/{run}/peak3d_out/rt_correction/{f}.tsv", sep="\t")
    rows = []
    for b0 in np.arange(0, a.rt_native.max() + bw, bw):
        s = (a.rt_native >= b0) & (a.rt_native < b0 + bw)
        t = np.isfinite(tsh) & (tx >= b0) & (tx < b0 + bw)
        g = (w.rt_native >= b0) & (w.rt_native < b0 + bw)
        if s.sum() == 0 and t.sum() == 0: continue
        d = a.residual_before_s[s].values
        rows.append(dict(bin=b0, n_anc=int(s.sum()), n_inl=int(a.is_inlier[s].sum()), anc_med=np.median(d) if len(d) else np.nan,
                         anc_q25=np.percentile(d, 25) if len(d) else np.nan, anc_q75=np.percentile(d, 75) if len(d) else np.nan,
                         anc_resid_after=np.median(a.residual_after_s[s]) if s.sum() else np.nan,
                         warp_sh=60 * np.median(w.rt_corr[g] - w.rt_native[g]),
                         n_tru=int(t.sum()), tru_med=np.median(tsh[t]) if t.sum() else np.nan,
                         tru_q25=np.percentile(tsh[t], 25) if t.sum() else np.nan, tru_q75=np.percentile(tsh[t], 75) if t.sum() else np.nan))
    print("---", run, f); print(pd.DataFrame(rows).round(2).to_string(index=False))
