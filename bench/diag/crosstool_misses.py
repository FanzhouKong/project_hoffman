# cross-tool feature comparison, part 2: why does peak3d lack the features that >= 2 other tools agree on
# (results/diag_crosstool/<RUN>/missed_consensus.tsv from crosstool_overlap.py)? Every position is traced through
# peak3d's pipeline like the truth misses: near miss (aligned feature within 20 ppm / 0.2 min), dropped by the presence
# rule, rejected per file (which gate / chromatogram check / score), or no candidate (then the raw replicate evidence
# from part 1 says whether there was a peak). Galleries for the consensus misses by class and for peak3d's single-tool
# features (replicated vs not). usage: crosstool_misses.py RUN   (<= 3-file runs; reads mzML -> srun)
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench/diag"))
from bench.runs import RUNS  # noqa: E402
from miss_common import STAGE_RANK, eic_desc, gate_reasons, gate_short, near, noise_lookup, plot_gallery, repick, to_native  # noqa: E402

RUN = sys.argv[1]
PPM, RT_TOL = 10.0, 0.1
NEAR_PPM, NEAR_RT = 20.0, 0.2
CTX_PPM, CTX_RT = 30.0, 0.3
D = ROOT / "results/peak3d" / RUN / "peak3d_out"
OUT = ROOT / "results/diag_crosstool" / RUN
EIC_PPM = 10.0 if RUN == "IDSL003" else 5.0
rng = np.random.default_rng(3)


def log(*a):
    print(*a, flush=True)


def main():
    files = RUNS[RUN]["files"]
    stems = [Path(f).stem for f in files]
    M0 = pd.read_csv(OUT / "missed_consensus.tsv", sep="\t")
    A = pd.read_csv(D / "aligned_feature_table.tsv", sep="\t")
    PM = pd.read_csv(D / "presence_mask.tsv", sep="\t").set_index("group_id")
    KEPT, W, C, P, RJ, LOOK = {}, {}, {}, {}, {}, {}
    for f, s in zip(files, stems):
        KEPT[s] = pd.read_csv(D / "features" / f"{s}.tsv", sep="\t")
        W[s] = pd.read_csv(D / "rt_correction" / f"{s}.tsv", sep="\t") if (D / "rt_correction" / f"{s}.tsv").exists() else None
        c, p, b, fe, rj = repick(f)
        C[s], P[s], RJ[s] = c, p, rj
        LOOK[s] = noise_lookup(c, p, b)
        log(s, "re-pick kept", len(fe), "benchmark kept", len(KEPT[s]), "rejected", len(rj))
    multi = len(files) > 1
    o_mz = np.argsort(A["mz"].values)
    a_mz, a_rt = A["mz"].values[o_mz], A["rt"].values[o_mz]
    rows, gallery = [], {}
    for n_done, r in enumerate(M0.itertuples()):
        mz, rt = float(r.mz), float(r.rt)
        row = dict(idx=r.Index, mz=mz, rt=rt, tool=r.tool, n_tools=int(r.n_tools), intensity=float(r.intensity),
                   is_isotopologue=bool(r.is_isotopologue), rep_files=int(getattr(r, "rep_files", -1)),
                   raw_apex=float(getattr(r, "raw_apex", np.nan)), raw_snr_cell=float(getattr(r, "raw_snr_cell", np.nan)),
                   raw_prom_rel=float(getattr(r, "raw_prom_rel", np.nan)))
        for col in ("truth", "cred"):
            if hasattr(r, col):
                row[col] = getattr(r, col)
        lo_, hi_ = np.searchsorted(a_mz, mz * (1 - PPM * 1e-6)), np.searchsorted(a_mz, mz * (1 + PPM * 1e-6), side="right")
        row["same_mz_any_rt_drt_s"] = float(60 * np.min(np.abs(a_rt[lo_:hi_] - rt))) if hi_ > lo_ else np.nan
        g = near(A, mz, rt, NEAR_PPM, NEAR_RT)
        if len(g):
            j = np.argmin(np.hypot(g["dppm"].values / NEAR_PPM, g["drt_s"].values / (60 * NEAR_RT)))
            x = g.iloc[j]
            row.update(near_dppm=float(x["dppm"]), near_drt_s=float(x["drt_s"]), near_in_span=bool(x["rt_min"] <= rt <= x["rt_max"]))
        best_rej, best_rank, n_pres, panels = None, -1, 0, []
        for s in stems:
            c, p = C[s], P[s]
            rt_nat = to_native(W[s], rt) if W[s] is not None else rt
            cell = LOOK[s](mz, rt_nat)
            k = KEPT[s]
            if multi:
                kf = near(k, mz, rt, PPM, RT_TOL, rt_col="rt_corr", mz_col="mz_corr" if "mz_corr" in k else "mz")
                kctx = near(k, mz, rt, CTX_PPM, CTX_RT, rt_col="rt_corr", mz_col="mz_corr" if "mz_corr" in k else "mz")
            else:
                kf = near(k, mz, rt, PPM, RT_TOL)
                kctx = near(k, mz, rt, CTX_PPM, CTX_RT)
            rf = near(RJ[s], mz, rt_nat, PPM, RT_TOL)
            rctx = near(RJ[s], mz, rt_nat, CTX_PPM, CTX_RT)
            if len(kf) and "group_kept" in kf and not bool(kf.iloc[np.argmin(np.abs(kf["drt_s"].values))]["group_kept"]):
                n_pres += 1
            if len(rf):
                rf = rf.assign(rank=[STAGE_RANK.get(v, 0) for v in rf["stage"]])
                x = rf.sort_values(["rank", "height"], ascending=[False, False]).iloc[0]
                if x["rank"] > best_rank:
                    best_rank, best_rej = int(x["rank"]), (s, x)
            d5 = eic_desc(c, mz, EIC_PPM, rt_nat, 0.5, RT_TOL, p.floor, cell)
            panels.append(dict(rt=d5["rt"], e=d5["e"], e_wide=None, rt_truth=rt_nat, tol=RT_TOL, noise=cell,
                               kept=[dict(rt_min=float(x.rt_min), rt_max=float(x.rt_max), label=f"kept {x.dppm:+.0f}ppm") for x in kctx.itertuples()],
                               rejected=[dict(rt_min=float(x.rt_min), rt_max=float(x.rt_max), label=f"{x.stage} {x.dppm:+.0f}ppm") for x in rctx.itertuples()],
                               label=f"{s[-6:]}: apex {d5['apex_h']:.2g} S/N {d5['snr_cell']:.1f} n{d5['n_contig']} dRT {d5['drt_s']:+.1f}s"))
        if len(g):
            cls = "near"
            sub = "rt_off" if abs(row["near_dppm"]) <= PPM else "mz_off" if abs(row["near_drt_s"]) <= 60 * RT_TOL else "both"
            if sub == "rt_off" and row.get("near_in_span"):
                sub = "rt_off_in_span"
        elif n_pres:
            cls, sub = "presence", "single_detection"
        elif best_rej is not None:
            s, x = best_rej
            cls, sub = "rejected", gate_short(x, P[s])
            row.update(rej_stage=x["stage"], rej_why="; ".join(gate_reasons(x, P[s])), rej_height=float(x["height"]), rej_snr=float(x["snr"]),
                       rej_n_scans=int(x["n_scans"]), rej_fwhm_s=float(60 * x["fwhm"]), rej_far_level=float(x.get("far_level", np.nan)),
                       rej_eic_snr=float(x.get("eic_snr", np.nan)), rej_score=float(x.get("score_rule", np.nan)))
        else:
            cls = "nocand"
            rf_ = row["rep_files"]
            sub = "raw_peak" if rf_ >= 2 else "raw_peak_1file" if rf_ == 1 else "raw_none"
        row.update(cls=cls, sub=sub)
        rows.append(row)
        gallery.setdefault(f"{cls}:{sub}", []).append(dict(title=f"#{r.Index} {mz:.4f}@{rt:.2f} {r.tool} n{int(r.n_tools)} I{float(r.intensity):.1e}", panels=panels))
        if n_done % 200 == 0:
            log("traced", n_done, "of", len(M0))
    M = pd.DataFrame(rows)
    M.to_csv(OUT / "missed_consensus_attributed.tsv", sep="\t", index=False, float_format="%.6g")
    for key, items in gallery.items():
        pick = [items[i] for i in rng.choice(len(items), size=min(8, len(items)), replace=False)]
        plot_gallery(pick, OUT / f"gallery_missed_{key.replace(':', '_')}.png", f"{RUN}: features of >= 2 other tools that peak3d lacks, {key} ({len(items)})")
    # galleries of peak3d-only features: replicated vs not
    U = pd.read_csv(OUT / "peak3d_only.tsv", sep="\t")
    for key, sel in (("replicated", U["rep_files"] >= 2), ("unreplicated", U["rep_files"] == 0)) if "rep_files" in U else ():
        sub = U[sel]
        if len(sub) == 0:
            continue
        pick = sub.iloc[rng.choice(len(sub), size=min(8, len(sub)), replace=False)]
        items = []
        for r in pick.itertuples():
            panels = []
            for s in stems:
                rt_nat = to_native(W[s], float(r.rt)) if W[s] is not None else float(r.rt)
                d5 = eic_desc(C[s], float(r.mz), EIC_PPM, rt_nat, 0.5, RT_TOL, P[s].floor, LOOK[s](float(r.mz), rt_nat))
                k = KEPT[s]
                kctx = near(k, float(r.mz), float(r.rt), CTX_PPM, CTX_RT, rt_col="rt_corr" if multi else "rt", mz_col="mz_corr" if "mz_corr" in k else "mz")
                panels.append(dict(rt=d5["rt"], e=d5["e"], e_wide=None, rt_truth=rt_nat, tol=RT_TOL, noise=LOOK[s](float(r.mz), rt_nat),
                                   kept=[dict(rt_min=float(x.rt_min), rt_max=float(x.rt_max), label=f"kept {x.dppm:+.0f}ppm") for x in kctx.itertuples()],
                                   rejected=[], label=f"{s[-6:]}: apex {d5['apex_h']:.2g} S/N {d5['snr_cell']:.1f} n{d5['n_contig']}"))
            items.append(dict(title=f"{r.mz:.4f}@{r.rt:.2f} I{float(r.intensity):.1e}{' iso' if r.is_isotopologue else ''}", panels=panels))
        plot_gallery(items, OUT / f"gallery_peak3d_only_{key}.png", f"{RUN}: peak3d-only features, {key} in the other injections ({int(sel.sum())})")
    with open(OUT / "missed_summary.log", "w") as fh:
        def w(*a):
            s_ = " ".join(str(x) for x in a)
            fh.write(s_ + "\n")
            print(s_, flush=True)
        w(f"== {RUN}: {len(M)} consensus-missed positions (>= 2 other tools, no peak3d feature)")
        agg = dict(n=("idx", "size"), med_intensity=("intensity", "median"), iso=("is_isotopologue", "mean"), rep2=("rep_files", lambda v: (v >= 2).mean()),
                   prom=("raw_prom_rel", "median"), snr_cell=("raw_snr_cell", "median"))
        if "truth" in M:
            agg["truth"] = ("truth", "mean")
        if "cred" in M:
            agg["cred"] = ("cred", "mean")
        w(M.groupby(["cls", "sub"]).agg(**agg).round(3).to_string())
        d = M["same_mz_any_rt_drt_s"]
        w("\nnearest peak3d feature at the same m/z (10 ppm), any RT: |dRT| quantiles (s)", np.nanpercentile(d, [10, 25, 50, 75, 90]).round(1).tolist(),
          "; none within 10 ppm:", int(d.isna().sum()), "; within 6-12 s:", int(((d > 6) & (d <= 12)).sum()), "; within 12-30 s:", int(((d > 12) & (d <= 30)).sum()))
        if "rejected" in M["cls"].values:
            x = M[M["cls"] == "rejected"]
            w("\n-- rejected: gate values")
            w(x.groupby("sub").agg(n=("idx", "size"), height=("rej_height", "median"), snr=("rej_snr", "median"), eic_snr=("rej_eic_snr", "median"),
                                   far=("rej_far_level", "median"), fwhm_s=("rej_fwhm_s", "median"), n_scans=("rej_n_scans", "median"), score=("rej_score", "median")).round(2).to_string())
    log("wrote", OUT)


if __name__ == "__main__":
    main()
