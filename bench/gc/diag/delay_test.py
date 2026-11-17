"""The production delayed-copy test (deconv._profile, deconv._lag_test) on labelled pairs of the MRH set: split-off
groups of one truth compound and pairs of different truth compounds, components taken from a run without the merge.

    python bench/gc/diag/delay_test.py peak3d score 25 50 100 250 500 1000
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from peak3d.gc import deconv as DC
from peak3d.io import Cloud
from bench.gc.deconv_eval import compound_features

R = Path("/quobyte/metabolomicsgrp/fanzhou/hoffmann/results/gc_mrh")


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
        d = DC.deconvolve(cloud, F, P, DC.DeconvParams(delay=False))
        comp = d.feature_component
        C = d.components
        f_mz = F["mz"].values.astype(float); f_h = F["height"].values.astype(float)
        f_tol = DC.ion_tol(f_mz, f_h, P)
        tight = np.where(d.feature_loose, -1, comp)
        mem_off, mem_idx = DC._member_lists(tight, f_h, len(C))
        cf = compound_features(T[T["level"] == lv])
        main_of, pairs = {}, []
        for e, (rt, fid) in cf.items():
            c = comp[fid]
            vals, cnt = np.unique(c[c >= 0], return_counts=True)
            if not len(vals):
                continue
            m = int(vals[np.argmax(cnt)])
            main_of[e] = m
            pairs += [("split-off group", e[:26], f"{n} of its ions", m, int(v)) for v, n in zip(vals, cnt) if v != m and n >= 2]
        names = sorted(main_of, key=lambda e: cf[e][0])
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if abs(cf[b][0] - cf[a][0]) > 1.0 * P.fwhm_med:
                    break
                if main_of[a] != main_of[b] and not len(np.intersect1d(cf[a][1], cf[b][1])):
                    pairs.append(("two compounds", a[:26], b[:26], main_of[a], main_of[b]))
        ct = C["rt"].values; apex = C["scan_apex"].values.astype(np.int64)
        for kind, na, nb, ca, cb in pairs:
            a, b = (ca, cb) if ct[ca] <= ct[cb] else (cb, ca)
            w0 = int(round(P.fwhm_med / P.dt))
            s0 = max(0, int(apex[a]) - 3 * w0)
            pa0 = DC._profile(cloud.off, cloud.mz, cloud.inten, f_mz, f_tol, mem_off, mem_idx, a, s0, min(cloud.n_scans, int(apex[a]) + 6 * w0), 5)
            width = float((pa0 >= 0.5).sum())
            w = int(width + 0.5)
            s0, s1 = max(0, int(apex[a]) - 3 * w), min(cloud.n_scans, int(apex[b]) + 6 * w + 1)
            pa = DC._profile(cloud.off, cloud.mz, cloud.inten, f_mz, f_tol, mem_off, mem_idx, a, s0, s1, 5)
            pb = DC._profile(cloud.off, cloud.mz, cloud.inten, f_mz, f_tol, mem_off, mem_idx, b, s0, s1, 5)
            d_scans = (ct[b] - ct[a]) / P.dt
            r = dict(kind=kind, level=lv, a=na, b=nb, first="a" if a == ca else "b", n_a=int(mem_off[a + 1] - mem_off[a]), n_b=int(mem_off[b + 1] - mem_off[b]),
                     d_scans=d_scans, width=width, in_reach=bool(0 < d_scans <= width))
            lag, move, both, tot, npts = DC._lag_stats(pa, pb, d_scans + 1.0, min(1.5 * width, 3.0 * d_scans + 1.0))
            rcut = float(d.info["shape_classes"][-1]) if d.info.get("calibrated") else 0.8
            n_prof = min(5, r["n_b"])
            dof = max(npts - 3, 1); noise = tot * (1 - rcut ** 2) / n_prof
            r.update(r_lag=float(np.sqrt(max(0.0, 1 - lag / tot))) if tot > 0 else 0.0, rss_lag=lag / tot, rss_move=move / tot, rss_both=both / tot,
                     noise=noise / tot, F_lag=(move - lag) / (max(lag, noise) / dof), F_move=(lag - both) / (max(both, noise) / dof))
            r["copy"] = bool(DC._is_delayed_copy(lag, move, both, tot, npts, rcut, n_prof, 5.0)) and r["in_reach"]
            rows.append(r)
    D = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    print(D.sort_values(["kind", "a", "level"]).round(3).to_string(index=False))
    print("called a copy:", D.groupby("kind")["copy"].agg(["sum", "size"]).to_dict("index"))


if __name__ == "__main__":
    main()
