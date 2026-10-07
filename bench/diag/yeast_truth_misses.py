# diagnostic: why does peak3d (iteration 2, results/peak3d/YEAST_12C) miss ~31 % of the YEAST credentialed truth
# (data/truth/YEAST/credentialed.tsv)? Every truth feature is traced through the pipeline: aligned hit at 10 ppm /
# 0.1 min (the benchmark rule), near miss at 20 ppm / 0.2 min, dropped by the presence rule, rejected per file (which
# gate / chromatogram check / score), or no candidate at all (then the raw EIC in the three injections says whether
# there was a peak to find). Other tools and earlier peak3d versions on the same truth features for reference.
# Writes results/diag_yeast/{yeast_misses.tsv, summary.log, gallery_<class>.png}.
# usage: yeast_truth_misses.py   (reads mzML -> srun)
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench/diag"))
from bench.runs import LOCAL_TOOLS, RUNS  # noqa: E402
from bench.score import hits, load  # noqa: E402
from miss_common import STAGE_RANK, eic_desc, gate_reasons, gate_short, near, noise_lookup, plot_gallery, repick, to_native  # noqa: E402

RUN = "YEAST_12C"
PPM, RT_TOL = 10.0, 0.1
NEAR_PPM, NEAR_RT = 20.0, 0.2
CTX_PPM, CTX_RT = 30.0, 0.3
EIC_HALF = 0.5
D = ROOT / "results/peak3d" / RUN / "peak3d_out"
OUT = ROOT / "results/diag_yeast"
OUT.mkdir(parents=True, exist_ok=True)
SNAPSHOTS = {"oct05": ROOT / "results/peak3d_2026-10-05" / RUN / "peak3d_out",
             "iter2": ROOT / "results/peak3d_2026-10-06_iter2" / RUN / "peak3d_out",
             "iter1": ROOT / "results/peak3d_2026-10-06_iter1" / RUN / "peak3d_out"}
rng = np.random.default_rng(7)


def log(*a):
    print(*a, flush=True)


def main():
    T = pd.read_csv(ROOT / "data/truth/YEAST/credentialed.tsv", sep="\t")
    files = RUNS[RUN]["files"]
    stems = [Path(f).stem for f in files]
    A = pd.read_csv(D / "aligned_feature_table.tsv", sep="\t")
    PM = pd.read_csv(D / "presence_mask.tsv", sep="\t").set_index("group_id")
    # ---- benchmark hit + other tools
    T["hit"] = hits(A[["mz", "rt"]], T["mz"].values, T["rt"].values, PPM, RT_TOL)
    ref = {}
    for k in ("asari", "masscube", *LOCAL_TOOLS, "idslipa"):
        x = load(k, RUN)
        if x is not None:
            ref[k] = x[0]
    for k, d in SNAPSHOTS.items():
        if (d / "aligned_feature_table.tsv").exists():
            t = pd.read_csv(d / "aligned_feature_table.tsv", sep="\t")
            ref[k] = pd.DataFrame({"mz": t["mz"], "rt": t["rt"]})
    for k, f in ref.items():
        T[f"hit_{k}"] = hits(f, T["mz"].values, T["rt"].values, PPM, RT_TOL)
    log("truth", len(T), "hit", int(T["hit"].sum()), {k: int(T[f"hit_{k}"].sum()) for k in ref})
    # ---- per-file tables, warps, re-pick with rejected candidates
    KEPT, W, C, P, RJ, LOOK = {}, {}, {}, {}, {}, {}
    for f, s in zip(files, stems):
        KEPT[s] = pd.read_csv(D / "features" / f"{s}.tsv", sep="\t")
        W[s] = pd.read_csv(D / "rt_correction" / f"{s}.tsv", sep="\t")
        c, p, b, fe, rj = repick(f)
        C[s], P[s], RJ[s] = c, p, rj
        LOOK[s] = noise_lookup(c, p, b)
        log(s, "re-pick kept", len(fe), "benchmark kept", len(KEPT[s]), "rejected", len(rj),
            "stages", rj["stage"].value_counts().to_dict())
    # ---- frame bias of the hits: truth rt vs matched aligned rt
    drt_hit, dppm_hit = [], []
    for r in T[T["hit"]].itertuples():
        g = near(A, r.mz, r.rt, PPM, RT_TOL)
        j = np.argmin(np.abs(g["drt_s"].values))
        drt_hit.append(g["drt_s"].values[j])
        dppm_hit.append(g["dppm"].values[j])
    drt_hit, dppm_hit = np.array(drt_hit), np.array(dppm_hit)
    # ---- trace every missed truth feature
    rows, gallery = [], {}
    miss = T[~T["hit"]]
    for n_done, r in enumerate(miss.itertuples()):
        mz, rt = float(r.mz), float(r.rt)
        row = dict(truth_id=r.truth_id, mz=mz, rt=rt, tier=r.tier, feature_class=r.feature_class, height12=r.height12,
                   reps12=r.reps12, n_cands=r.n_cands, r_shape=r.r)
        for k in ref:
            row[f"hit_{k}"] = bool(getattr(r, f"hit_{k}"))
        g = near(A, mz, rt, NEAR_PPM, NEAR_RT)
        if len(g):
            j = np.argmin(np.hypot(g["dppm"].values / NEAR_PPM, g["drt_s"].values / (60 * NEAR_RT)))
            x = g.iloc[j]
            row.update(near_group=int(x["group_id"]), near_dppm=float(x["dppm"]), near_drt_s=float(x["drt_s"]),
                       near_in_span=bool(x["rt_min"] <= rt <= x["rt_max"]), near_n_det=int(x["n_detected"]),
                       near_n_conf=int(x["n_confirmed"]))
        # per file
        best_rej, best_rank, n_rej_files, n_pres_files = None, -1, 0, 0
        pres_info, raw5, raw15, panels = [], [], [], []
        for s in stems:
            c, p = C[s], P[s]
            rt_nat = to_native(W[s], rt)
            cell = LOOK[s](mz, rt_nat)
            kf = near(KEPT[s], mz, rt, PPM, RT_TOL, rt_col="rt_corr", mz_col="mz_corr")
            kctx = near(KEPT[s], mz, rt, CTX_PPM, CTX_RT, rt_col="rt_corr", mz_col="mz_corr")
            rf = near(RJ[s], mz, rt_nat, PPM, RT_TOL)
            rctx = near(RJ[s], mz, rt_nat, CTX_PPM, CTX_RT)
            d5 = eic_desc(c, mz, 5.0, rt_nat, EIC_HALF, RT_TOL, p.floor, cell)
            d15 = eic_desc(c, mz, 15.0, rt_nat, EIC_HALF, RT_TOL, p.floor, cell)
            raw5.append(d5)
            raw15.append(d15)
            if len(kf):
                k0 = kf.iloc[np.argmin(np.abs(kf["drt_s"].values))]
                if not bool(k0["group_kept"]):
                    n_pres_files += 1
                    gid = int(k0["group_id"])
                    pres_info.append(dict(file=s, height=float(k0["height"]), snr=float(k0["snr"]), score=float(k0["score"]),
                                          eic_snr=float(k0["eic_snr"]), n_det=int(PM.loc[gid, "n_detected"]),
                                          n_conf=int(PM.loc[gid, "n_confirmed"])))
            if len(rf):
                n_rej_files += 1
                rf = rf.assign(rank=[STAGE_RANK.get(v, 0) for v in rf["stage"]])
                x = rf.sort_values(["rank", "height"], ascending=[False, False]).iloc[0]
                if x["rank"] > best_rank:
                    best_rank, best_rej = int(x["rank"]), (s, x)
            # context for the gallery / neighbour analysis
            ctx = []
            for x in kctx.itertuples():
                ctx.append(f"K{x.dppm:+.0f}ppm/{x.drt_s:+.0f}s")
            for x in rctx.itertuples():
                ctx.append(f"R:{x.stage}{x.dppm:+.0f}ppm/{x.drt_s:+.0f}s")
            row[f"ctx_{s[-1]}"] = " ".join(ctx[:6])
            panels.append(dict(rt=d5["rt"], e=d5["e"], e_wide=d15["e"], rt_truth=rt_nat, tol=RT_TOL, noise=cell,
                               kept=[dict(rt_min=float(np.interp(x.rt_min_corr, W[s]["rt_corr"], W[s]["rt_native"])) if "rt_min_corr" in kctx else float(x.rt_min),
                                          rt_max=float(np.interp(x.rt_max_corr, W[s]["rt_corr"], W[s]["rt_native"])) if "rt_max_corr" in kctx else float(x.rt_max),
                                          label=f"kept {x.dppm:+.0f}ppm sc{x.score:.2f}{'' if x.group_kept else ' PRES-DROP'}")
                                     for x in kctx.itertuples()],
                               rejected=[dict(rt_min=float(x.rt_min), rt_max=float(x.rt_max),
                                              label=f"{x.stage} {x.dppm:+.0f}ppm")
                                         for x in rctx.itertuples()],
                               label=f"{s[-1]}: apex {d5['apex_h']:.2g} S/N {d5['snr_cell']:.1f} n{d5['n_contig']} "
                                     f"dRT {d5['drt_s']:+.1f}s cell {cell:.0f}"))
        # raw evidence summary (5 ppm; 15 ppm as the m/z-off check)
        def peak_files(ds):
            return sum(1 for d in ds if d["local_max"] and d["snr_cell"] >= 3 and d["n_contig"] >= 4 and abs(d["drt_s"]) <= 60 * RT_TOL)
        n_peak5, n_peak15 = peak_files(raw5), peak_files(raw15)
        n_any5 = sum(1 for d in raw5 if d["apex_h"] > 0)
        row.update(raw_peak_files_5ppm=n_peak5, raw_peak_files_15ppm=n_peak15, raw_any_files_5ppm=n_any5,
                   raw_max_snr_cell=max(d["snr_cell"] for d in raw5), raw_max_apex=max(d["apex_h"] for d in raw5),
                   raw_max_contig=max(d["n_contig"] for d in raw5),
                   raw_med_drt_s=float(np.nanmedian([d["drt_s"] for d in raw5])) if n_any5 else np.nan,
                   n_rej_files=n_rej_files, n_pres_files=n_pres_files)
        # classification
        if len(g):
            cls = "near"
            sub = ("rt_off" if abs(row["near_dppm"]) <= PPM else "mz_off" if abs(row["near_drt_s"]) <= 60 * RT_TOL else "both")
            if sub == "rt_off" and row["near_in_span"]:
                sub = "rt_off_in_span"
        elif n_pres_files:
            cls, sub = "presence", f"det{pres_info[0]['n_det']}_conf{pres_info[0]['n_conf']}"
            row.update(pres_height=pres_info[0]["height"], pres_snr=pres_info[0]["snr"], pres_score=pres_info[0]["score"],
                       pres_eic_snr=pres_info[0]["eic_snr"])
            # evidence in the files where it was not detected
            others = [d for d, s in zip(raw5, stems) if s not in {q["file"] for q in pres_info}]
            row.update(pres_other_max_snr_cell=max([d["snr_cell"] for d in others], default=np.nan),
                       pres_other_max_contig=max([d["n_contig"] for d in others], default=np.nan),
                       pres_other_peak_files=peak_files(others))
        elif best_rej is not None:
            s, x = best_rej
            cls, sub = "rejected", gate_short(x, P[s])
            row.update(rej_file=s, rej_stage=x["stage"], rej_why="; ".join(gate_reasons(x, P[s])), rej_height=float(x["height"]),
                       rej_snr=float(x["snr"]), rej_n_scans=int(x["n_scans"]), rej_fwhm_s=float(60 * x["fwhm"]),
                       rej_far_level=float(x.get("far_level", np.nan)), rej_eic_snr=float(x.get("eic_snr", np.nan)),
                       rej_score=float(x.get("score_rule", np.nan)), rej_dppm=float(x["dppm"]), rej_drt_s=float(x["drt_s"]),
                       rej_noise=float(x["noise"]))
        else:
            cls = "nocand"
            sub = ("raw_peak" if n_peak5 >= 2 else "raw_peak_1file" if n_peak5 == 1 else
                   "mz_off_raw" if n_peak15 >= 2 else "raw_weak" if n_any5 >= 2 else "raw_none")
        row.update(cls=cls, sub=sub)
        rows.append(row)
        key = f"{cls}:{sub}"
        gallery.setdefault(key, []).append(dict(title=f"#{r.truth_id} {mz:.4f}@{rt:.2f} {r.tier}/{r.feature_class[:4]} h{r.height12:.1e}",
                                                panels=panels))
        if n_done % 100 == 0:
            log("traced", n_done, "of", len(miss))
    M = pd.DataFrame(rows)
    M.to_csv(OUT / "yeast_misses.tsv", sep="\t", index=False)
    # ---- galleries: up to 8 random examples per class
    for key, items in gallery.items():
        pick = [items[i] for i in rng.choice(len(items), size=min(8, len(items)), replace=False)]
        plot_gallery(pick, OUT / f"gallery_{key.replace(':', '_')}.png", f"{RUN} missed truth: {key} ({len(items)})")
    # ---- summary
    with open(OUT / "summary.log", "w") as fh:
        def w(*a):
            s = " ".join(str(x) for x in a)
            fh.write(s + "\n")
            print(s, flush=True)
        n = len(T)
        w(f"YEAST credentialed truth {n}; peak3d (iter2) hits {int(T['hit'].sum())} = {T['hit'].mean():.3f} at {PPM} ppm / {RT_TOL} min")
        w("other tools / versions on the same truth:", {k: f"{T[f'hit_{k}'].mean():.3f}" for k in ref})
        w(f"frame check on hits: median drt {np.median(drt_hit):+.2f} s (IQR {np.percentile(drt_hit, 25):+.2f}..{np.percentile(drt_hit, 75):+.2f}), "
          f"median dppm {np.median(dppm_hit):+.2f} (IQR {np.percentile(dppm_hit, 25):+.2f}..{np.percentile(dppm_hit, 75):+.2f})")
        w("\n== recall by tier / feature class / height12 ==")
        for col in ("tier", "feature_class"):
            w(T.groupby(col)["hit"].agg(["size", "mean"]).round(3).to_string())
        hb = pd.cut(np.log10(T["height12"]), [0, 4, 4.5, 5, 5.5, 6, 9])
        w(T.groupby(hb, observed=True)["hit"].agg(["size", "mean"]).round(3).to_string())
        w("\n== missed truth by class ==")
        w(M.groupby(["cls", "sub"]).size().to_string())
        w("\n== per class: tier A share, primary share, median height12, found by masscube / oct05 ==")
        agg = M.groupby("cls").agg(n=("truth_id", "size"), tierA=("tier", lambda x: (x == "A").mean()),
                                   primary=("feature_class", lambda x: (x == "primary").mean()),
                                   med_height12=("height12", "median"),
                                   **{f"hit_{k}": (f"hit_{k}", "mean") for k in ("masscube", "oct05") if f"hit_{k}" in M})
        w(agg.round(3).to_string())
        w("\n== missed truth found by others (what a perfect rescue could reach) ==")
        for k in ref:
            w(f"  {k}: finds {int(M[f'hit_{k}'].sum())} of the {len(M)} misses")
        if "near" in M["cls"].values:
            x = M[M["cls"] == "near"]
            w("\n== near misses: nearest aligned feature within 20 ppm / 0.2 min ==")
            w(x.groupby("sub").agg(n=("truth_id", "size"), med_dppm=("near_dppm", "median"), med_drt_s=("near_drt_s", "median"),
                                   in_span=("near_in_span", "mean"), med_height12=("height12", "median")).round(2).to_string())
            w("  |drt| quantiles (s):", np.percentile(np.abs(x["near_drt_s"]), [25, 50, 75, 90]).round(1).tolist(),
              " |dppm| quantiles:", np.percentile(np.abs(x["near_dppm"]), [25, 50, 75, 90]).round(1).tolist())
        if "presence" in M["cls"].values:
            x = M[M["cls"] == "presence"]
            w("\n== presence-dropped: the detected feature and the raw evidence in the other injections ==")
            w(x[["pres_height", "pres_snr", "pres_eic_snr", "pres_score", "pres_other_max_snr_cell", "pres_other_max_contig"]]
              .describe(percentiles=[0.25, 0.5, 0.75]).round(2).to_string())
            w("  other-injection raw peak (local max, S/N_cell >= 3, >= 4 scans, |dRT| <= 6 s) in >= 1 file:",
              int((x["pres_other_peak_files"] >= 1).sum()), "of", len(x))
        if "rejected" in M["cls"].values:
            x = M[M["cls"] == "rejected"]
            w("\n== rejected per file: main reason ==")
            w(x.groupby("sub").agg(n=("truth_id", "size"), med_height=("rej_height", "median"), med_snr=("rej_snr", "median"),
                                   med_eic_snr=("rej_eic_snr", "median"), med_far=("rej_far_level", "median"),
                                   med_fwhm_s=("rej_fwhm_s", "median"), med_n_scans=("rej_n_scans", "median"),
                                   med_score=("rej_score", "median"), masscube=("hit_masscube", "mean")).round(2).to_string())
            w("  rejected in how many files:", x["n_rej_files"].value_counts().sort_index().to_dict())
        if "nocand" in M["cls"].values:
            x = M[M["cls"] == "nocand"]
            w("\n== no candidate at all: raw EIC evidence (5 ppm, +/- 0.1 min of the truth RT) ==")
            w(x.groupby("sub").agg(n=("truth_id", "size"), med_apex=("raw_max_apex", "median"), med_snr_cell=("raw_max_snr_cell", "median"),
                                   med_contig=("raw_max_contig", "median"), med_height12=("height12", "median"),
                                   tierA=("tier", lambda v: (v == "A").mean()), masscube=("hit_masscube", "mean"),
                                   oct05=("hit_oct05", "mean") if "hit_oct05" in x else ("truth_id", "size")).round(2).to_string())
            w("  context (kept K / rejected R candidates within 30 ppm / 0.3 min) of the raw_peak cases, file a:")
            for q in x[x["sub"] == "raw_peak"].head(25).itertuples():
                w(f"   #{q.truth_id} {q.mz:.4f}@{q.rt:.2f} h{q.height12:.1e} apex {q.raw_max_apex:.2g} snr_cell {q.raw_max_snr_cell:.1f}: {q.ctx_a} | {q.ctx_b} | {q.ctx_c}")
    log("wrote", OUT)


if __name__ == "__main__":
    main()
