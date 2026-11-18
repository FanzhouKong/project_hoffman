#!/usr/bin/env python3
"""The numbers of the explained-component test (peak3d.gc.deconv._explained) for the components of a time window:
neighbours used, share of the ions' variance the others leave, the allowance, F, who carries the explanation.

    python bench/gc/diag/explained_trace.py OUTDIR STEM T0 T1 [name=value ...]
OUTDIR: a peak3d output folder (features/, .cache/); T0, T1 seconds."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from peak3d.gc import deconv as DC                       # noqa: E402
from peak3d.io import Cloud, eic_kernel                  # noqa: E402

out_dir, stem, t0, t1 = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
kw = {}
ref = DC.DeconvParams()
for a in sys.argv[5:]:
    k, v = a.split("=")
    kw[k] = bool(int(v)) if isinstance(getattr(ref, k), bool) else type(getattr(ref, k))(float(v))
P = DC.params_from_dict(json.loads((out_dir / "features" / f"{stem}.params.json").read_text()))
cloud = Cloud.load_npz(out_dir / ".cache" / f"{stem}.npz")
F = pd.read_csv(out_dir / "features" / f"{stem}.tsv", sep="\t", low_memory=False)
rows = []
real = DC._explained


def traced(off, mz, inten, f_mz, f_tol, mem_off, mem_idx, k_top, c_t, c_lo, c_hi, c_rcut, alive, rank, m_off, m_val,
           max_nb, f_max, smaller):
    C = len(c_t)
    out = alive.copy()
    nb = np.empty(max_nb, np.int64)
    for b in np.argsort(rank):
        if not out[b] or mem_off[b + 1] == mem_off[b]:
            continue
        lo, hi = c_lo[b], c_hi[b]
        W = hi - lo + 1
        n_nb = DC._nearest_live(c_t[b], lo, hi, b, c_t, c_lo, c_hi, out, rank, -1 if smaller else rank[b], nb)
        rec = dict(b=int(b), t=c_t[b] * 60, lo=cloud.rt[lo] * 60, hi=cloud.rt[hi] * 60, n_mem=int(mem_off[b + 1] - mem_off[b]), W=int(W),
                   nb=" ".join(f"{c_t[a] * 60:.1f}" for a in nb[:n_nb]), rcut=c_rcut[b])
        if n_nb == 0 or W < n_nb + 4:
            rec["verdict"] = "untested"; rows.append(rec); continue
        X = np.zeros((W, n_nb + 2)); X[:, 0] = 1.0
        for q in range(n_nb):
            a = nb[q]
            for w in range(W):
                s = lo + w
                if c_lo[a] <= s <= c_hi[a]:
                    X[w, q + 1] = m_val[m_off[a] + s - c_lo[a]]
        X[:, n_nb + 1] = m_val[m_off[b]:m_off[b] + W]
        G = X.T @ X
        num = den = left = allowed = by_l = by_s = tot_all = 0.0
        cols = np.empty(max_nb + 2, np.int64)
        for q in range(min(k_top, mem_off[b + 1] - mem_off[b])):
            j = mem_idx[mem_off[b] + q]
            y = eic_kernel(off, mz, inten, f_mz[j], f_tol[j], lo, hi + 1)
            yy = float(y @ y); sy = float(y.sum()); g = X.T @ y
            cols[:n_nb + 1] = np.arange(n_nb + 1)
            coef, sse0 = DC._nnls_fit(G, g, yy, cols, n_nb + 1)
            for c in range(n_nb):
                sh = coef[c + 1] * G[0, c + 1]
                if rank[nb[c]] > rank[b]:
                    by_l += sh
                else:
                    by_s += sh
            cols[0] = n_nb + 1; cols[1:n_nb + 2] = np.arange(n_nb + 1)
            sse1 = min(DC._nnls_sse(G, g, yy, cols, n_nb + 2), sse0)
            tot = yy - sy * sy / W
            noise = tot * (1.0 - c_rcut[b] ** 2)
            num += max(sse0 - sse1, 0.0); den += max(sse1, noise); left += sse0; allowed += noise; tot_all += tot
        Fv = num / (den / (W - n_nb - 2)) if den > 0 else np.nan
        gone = by_l > by_s and den > 0 and left <= allowed and Fv < f_max
        rec.update(left=left / max(tot_all, 1e-12), allowed=allowed / max(tot_all, 1e-12), F=Fv, larger=by_l / max(by_l + by_s, 1e-12),
                   verdict="dissolved" if gone else ("kept: smaller explain" if by_l <= by_s else "kept: not described" if left > allowed
                                                    else "kept: own model needed"))
        if gone:
            out[b] = False
        rows.append(rec)
    chk = real(off, mz, inten, f_mz, f_tol, mem_off, mem_idx, k_top, c_t, c_lo, c_hi, c_rcut, alive, rank, m_off, m_val,
               max_nb, f_max, smaller)
    assert np.array_equal(chk, out), "trace differs from the kernel"
    return out


DC._explained = traced
d = DC.deconvolve(cloud, F, P, DC.DeconvParams(**kw))
T = pd.DataFrame(rows)
T = T[(T.t >= t0) & (T.t <= t1)].sort_values("t")
pd.set_option("display.width", 250, "display.max_rows", 500)
print(kw, {k: d.info.get(k) for k in ("n_components", "n_explained", "n_rejoined")})
print(T.round(3).to_string(index=False))
