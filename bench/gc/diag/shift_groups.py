"""Apex-shifted fragment groups: is a component that holds some of a truth compound's ions, a fraction of a second
after the compound's main component, a delayed copy of it (front edge at the same time, apex and tail later) or a
shifted copy (everything later, as a second compound would be)? Edge times of the two components' profiles
(median of the normalised chromatograms of the strongest tight members), compared with truth pairs of different
compounds that elute close together.

    python bench/gc/diag/shift_groups.py peak3d score 100 250 500 1000
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from peak3d.gc import deconv as DC
from peak3d.io import Cloud
from bench.gc.deconv_eval import compound_features

R = Path("/quobyte/metabolomicsgrp/fanzhou/hoffmann/results/gc_mrh")


def profile(cloud, F, members, s0, s1, k=5):
    ys = []
    for i in members[:k]:
        y = np.asarray(cloud.eic(float(F["mz"].values[i]), 10.0, s0, s1), float)
        if y.max() > 0:
            ys.append(y / y.max())
    if not ys:
        return None
    p = np.median(np.array(ys), axis=0) if len(ys) >= 3 else np.mean(np.array(ys), axis=0)
    return p / p.max() if p.max() > 0 else None


def crossing(t, p, level, side):
    """time where the profile crosses `level` on the left (side -1) or right (+1) of its maximum, linear between scans"""
    a = int(np.argmax(p))
    j = a
    while 0 <= j + side < len(p) and p[j + side] >= level:
        j += side
    k = j + side
    if not 0 <= k < len(p):
        return np.nan
    return t[j] + (t[k] - t[j]) * (p[j] - level) / (p[j] - p[k])


def model_compare(pa, pb, d_max, step=0.05):
    """B's profile as a shifted copy of A's (delay d scans) and as A's passed through a first-order lag (time
    constant tau scans): residual sums of squares of the best fit of each, with the amplitude free"""
    n = np.arange(len(pa), dtype=float)
    def rss(x):
        den = float(x @ x)
        a = float(x @ pb) / den if den > 0 else 0.0
        return float(((pb - a * x) ** 2).sum())
    best_s, best_c = (np.inf, 0.0), (np.inf, 0.0)
    for dl in np.arange(0.0, d_max + step, step):
        r = rss(np.interp(n - dl, n, pa, left=0.0, right=0.0))
        if r < best_s[0]:
            best_s = (r, dl)
    for tau in np.arange(0.0, d_max + step, step):
        if tau == 0:
            y = pa
        else:
            al = np.exp(-1.0 / tau)
            y = np.empty_like(pa); acc = 0.0
            for k in range(len(pa)):
                acc = (1 - al) * pa[k] + al * acc
                y[k] = acc
        r = rss(y)
        if r < best_c[0]:
            best_c = (r, tau)
    return best_s, best_c


def general_fit(pa, pb, d_max, step=0.05):
    """B = A lagged (time constant tau) and shifted (delay d): best fit over both, and the best with d = 0;
    F = how much the shift improves on the lag alone, per residual degree of freedom"""
    n = np.arange(len(pa), dtype=float)
    N = int((np.maximum(pa, pb) >= 0.05).sum())
    best = (np.inf, 0.0, 0.0); lag_only = np.inf
    for tau in np.arange(0.0, d_max + step, step):
        if tau == 0:
            y = pa
        else:
            al = np.exp(-1.0 / tau)
            y = np.empty_like(pa); acc = 0.0
            for k in range(len(pa)):
                acc = (1 - al) * pa[k] + al * acc
                y[k] = acc
        for dl in np.arange(0.0, d_max + step, step):
            x = y if dl == 0 else np.interp(n - dl, n, y, left=0.0, right=0.0)
            den = float(x @ x)
            a = float(x @ pb) / den if den > 0 else 0.0
            r = float(((pb - a * x) ** 2).sum())
            if dl == 0 and r < lag_only:
                lag_only = r
            if r < best[0]:
                best = (r, dl, tau)
    F = (lag_only - best[0]) / (best[0] / max(N - 3, 1)) if best[0] > 0 else 0.0
    return dict(F_shift=F, gen_delay=best[1], gen_tau=best[2], n_pts=N, rss_gen=best[0])


def pair_models(cloud, F, d, ca, cb, comp, f_h, w):
    C = d.components
    if C["rt"].values[cb] < C["rt"].values[ca]:
        ca, cb = cb, ca
    s0 = max(0, int(C["scan_apex"].values[ca]) - 3 * w)
    s1 = min(cloud.n_scans, int(C["scan_apex"].values[cb]) + 5 * w + 1)
    P = []
    for c in (ca, cb):
        mem = np.flatnonzero((comp == c) & ~d.feature_loose)
        P.append(profile(cloud, F, mem[np.argsort(-f_h[mem])], s0, s1))
    if P[0] is None or P[1] is None:
        return None
    (rs, dl), (rc, tau) = model_compare(P[0], P[1], 1.5 * w)
    tot = float(((P[1] - P[1].mean()) ** 2).sum())
    g = general_fit(P[0], P[1], 1.5 * w)
    g["rss_gen"] /= tot
    return dict(rss_shift=rs / tot, rss_lag=rc / tot, delay_scans=dl, tau_scans=tau, **g)


def edges(cloud, F, d, c, comp, f_h, w):
    C = d.components
    ap = int(C["scan_apex"].values[c])
    s0, s1 = max(0, ap - 3 * w), min(cloud.n_scans, ap + 3 * w + 1)
    mem = np.flatnonzero((comp == c) & ~d.feature_loose)
    mem = mem[np.argsort(-f_h[mem])]
    p = profile(cloud, F, mem, s0, s1)
    if p is None:
        return None
    t = cloud.rt[s0:s1] * 60
    return dict(apex=float(C["rt"].values[c]) * 60, f25=crossing(t, p, 0.25, -1), f50=crossing(t, p, 0.5, -1),
                b50=crossing(t, p, 0.5, 1), b25=crossing(t, p, 0.25, 1), n=len(mem), h=float(f_h[mem[0]]) if len(mem) else 0.0)


def main():
    src, score = sys.argv[1], sys.argv[2]
    out = R / src
    T = pd.read_csv(R / score / "truth_rows.tsv", sep="\t", dtype={"level": str})
    rows = []
    for lv in sys.argv[3:]:
        stem = f"NEWSTD_{lv}"
        P = DC.params_from_dict(json.loads((out / "features" / f"{stem}.params.json").read_text()))
        cloud = Cloud.load_npz(out / ".cache" / f"{stem}.npz")
        F = pd.read_csv(out / "features" / f"{stem}.tsv", sep="\t", low_memory=False)
        d = DC.deconvolve(cloud, F, P)
        comp = d.feature_component
        f_h = F["height"].values.astype(float)
        w = int(round(P.fwhm_med / P.dt))
        cf = compound_features(T[T["level"] == lv])
        main_of = {}
        for e, (rt, fid) in cf.items():
            c = comp[fid]
            vals, cnt = np.unique(c[c >= 0], return_counts=True)
            if not len(vals):
                continue
            m = int(vals[np.argmax(cnt)])
            main_of[e] = m
            for v, n in zip(vals, cnt):
                if v != m and n >= 2:
                    rows.append(dict(kind="split-off group", level=lv, a=e[:26], b=f"{n} of its ions", ca=m, cb=int(v)))
        names = sorted(main_of, key=lambda e: cf[e][0])
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if abs(cf[b][0] - cf[a][0]) > 1.0 * P.fwhm_med:
                    break
                if main_of[a] != main_of[b] and not len(np.intersect1d(cf[a][1], cf[b][1])):
                    rows.append(dict(kind="two compounds", level=lv, a=a[:26], b=b[:26], ca=main_of[a], cb=main_of[b]))
        for r in rows:
            if r["level"] != lv or "d_apex" in r:
                continue
            A, B = edges(cloud, F, d, r["ca"], comp, f_h, w), edges(cloud, F, d, r["cb"], comp, f_h, w)
            if A is None or B is None:
                r["d_apex"] = np.nan
                continue
            sgn = 1.0 if B["apex"] >= A["apex"] else -1.0
            r.update(d_apex=sgn * (B["apex"] - A["apex"]), d_f25=sgn * (B["f25"] - A["f25"]), d_f50=sgn * (B["f50"] - A["f50"]),
                     d_b50=sgn * (B["b50"] - A["b50"]), d_b25=sgn * (B["b25"] - A["b25"]),
                     w_a=A["b50"] - A["f50"], w_b=B["b50"] - B["f50"], n_a=A["n"], n_b=B["n"], h_a=A["h"], h_b=B["h"], fwhm=P.fwhm_med * 60)
            r.update(pair_models(cloud, F, d, r["ca"], r["cb"], comp, f_h, w) or {})
    D = pd.DataFrame(rows).dropna(subset=["d_apex"])
    D["front_over_apex"] = D["d_f50"] / D["d_apex"].where(D["d_apex"].abs() > 0.02)
    D["back_over_apex"] = D["d_b50"] / D["d_apex"].where(D["d_apex"].abs() > 0.02)
    D["width_ratio"] = D["w_b"] / D["w_a"]
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 300); pd.set_option("display.max_columns", 40)
    D["lag_over_shift"] = D["rss_lag"] / D["rss_shift"].clip(lower=1e-12)
    cols = ["kind", "level", "a", "b", "n_a", "n_b", "d_apex", "front_over_apex", "rss_shift", "rss_lag", "rss_gen", "lag_over_shift", "tau_scans", "gen_delay", "gen_tau", "n_pts", "F_shift"]
    print(D.sort_values(["kind", "a", "level"])[cols].round(4).to_string(index=False))
    print(D.groupby("kind")[["front_over_apex", "lag_over_shift", "rss_shift", "rss_lag"]].describe().round(3).T.to_string())
    for f in (5, 10, 20, 40):
        print(f"no shift needed (F < {f}):", D.groupby("kind").apply(lambda g: f"{(g['F_shift'] < f).sum()} of {len(g)}").to_dict())
    print("lag fits better (ratio < 1):", D.groupby("kind").apply(lambda g: f"{(g['lag_over_shift'] < 1).sum()} of {len(g)}").to_dict())
    D.to_csv(R / "deconv_eval" / f"shift_groups_{src}.tsv", sep="\t", index=False, float_format="%.5g")


if __name__ == "__main__":
    main()
