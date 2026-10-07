# diagnostic: why does peak3d (iteration 2, results/peak3d/IDSL003) miss 17 % of the 1,676 cleaned IDSL003 TP labels
# (data/truth/IDSL003/labels_clean.csv; every one of them passed a blind visual check: a peak apex inside +/- 0.1 min
# at 10 ppm)? Each missed label is traced: near miss (kept feature within 20 ppm / 0.2 min, or 30 ppm / 0.3 min),
# rejected candidate within 10 ppm / 0.1 min (which gate / chromatogram check / score), or no candidate (then the raw
# EIC at 10 and 30 ppm says what the validators saw). Published tools, our runs of other tools and the Oct 5 picker
# on the same labels for reference. Writes results/diag_idsl_tp/{idsl_tp_misses.tsv, summary.log, gallery_<class>.png}.
# usage: idsl_tp_misses.py   (reads 003.mzML -> srun)
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench/diag"))
from bench.runs import LOCAL_TOOLS, RUNS  # noqa: E402
from bench.score import IDSL_TRUTH, hits, load  # noqa: E402
from miss_common import STAGE_RANK, eic_desc, gate_reasons, gate_short, near, noise_lookup, plot_gallery, repick  # noqa: E402

RUN = "IDSL003"
PPM, RT_TOL = 10.0, 0.1
NEAR_PPM, NEAR_RT = 20.0, 0.2
CTX_PPM, CTX_RT = 30.0, 0.3
EIC_HALF = 0.5
D = ROOT / "results/peak3d" / RUN / "peak3d_out"
OUT = ROOT / "results/diag_idsl_tp"
OUT.mkdir(parents=True, exist_ok=True)
SNAPSHOTS = {"oct05": ROOT / "results/peak3d_2026-10-05" / RUN / "peak3d_out",
             "iter2": ROOT / "results/peak3d_2026-10-06_iter2" / RUN / "peak3d_out",
             "iter1": ROOT / "results/peak3d_2026-10-06_iter1" / RUN / "peak3d_out"}
PUBLISHED = ["IDSL.IPA", "XCMS", "MZMINE", "MSDIAL"]
rng = np.random.default_rng(7)


def log(*a):
    print(*a, flush=True)


def main():
    lab = pd.read_csv(IDSL_TRUTH)
    T = lab[lab["Manual Curation"] == "TP"].reset_index(drop=True)
    T = T.rename(columns={"m/z": "mz", "RT(min)": "rt"})
    f = RUNS[RUN]["files"][0]
    KEPT = pd.read_csv(D / "features" / f"{Path(f).stem}.tsv", sep="\t")
    A = pd.read_csv(D / "aligned_feature_table.tsv", sep="\t")
    T["hit"] = hits(A[["mz", "rt"]], T["mz"].values, T["rt"].values, PPM, RT_TOL)
    ref = {}
    for k in ("idslipa", "masscube", *LOCAL_TOOLS):
        x = load(k, RUN)
        if x is not None:
            ref[k] = x[0]
    for k, d in SNAPSHOTS.items():
        if (d / "aligned_feature_table.tsv").exists():
            t = pd.read_csv(d / "aligned_feature_table.tsv", sep="\t")
            ref[k] = pd.DataFrame({"mz": t["mz"], "rt": t["rt"]})
    for k, ft in ref.items():
        T[f"hit_{k}"] = hits(ft, T["mz"].values, T["rt"].values, PPM, RT_TOL)
    for c in PUBLISHED:
        T[f"hit_{c}"] = T[c].isin(["TP", "FP"]).values
    log("TP labels", len(T), "hit", int(T["hit"].sum()), {k: int(T[f"hit_{k}"].sum()) for k in list(ref) + PUBLISHED})
    c, P, b, fe, RJ = repick(f)
    look = noise_lookup(c, P, b)
    log("re-pick kept", len(fe), "benchmark kept", len(KEPT), "rejected", len(RJ), RJ["stage"].value_counts().to_dict())
    # ---- hits: offsets of the matched feature (frame / centring check)
    drt_hit, dppm_hit = [], []
    for r in T[T["hit"]].itertuples():
        g = near(KEPT, r.mz, r.rt, PPM, RT_TOL)
        j = np.argmin(np.abs(g["drt_s"].values))
        drt_hit.append(g["drt_s"].values[j])
        dppm_hit.append(g["dppm"].values[j])
    drt_hit, dppm_hit = np.array(drt_hit), np.array(dppm_hit)
    # ---- raw EIC descriptors for ALL TP labels (hits too), so the misses can be compared with the hits
    desc10 = [eic_desc(c, float(r.mz), 10.0, float(r.rt), EIC_HALF, RT_TOL, P.floor, look(float(r.mz), float(r.rt))) for r in T.itertuples()]
    T["raw_apex10"] = [d["apex_h"] for d in desc10]
    T["raw_snr_cell10"] = [d["snr_cell"] for d in desc10]
    T["raw_contig10"] = [d["n_contig"] for d in desc10]
    T["raw_fwhm_s10"] = [d["fwhm_s"] for d in desc10]
    T["raw_localmax10"] = [d["local_max"] for d in desc10]
    T["raw_drt_s10"] = [d["drt_s"] for d in desc10]
    rows, gallery = [], {}
    miss = T[~T["hit"]]
    for n_done, (i, r) in enumerate(miss.iterrows()):
        mz, rt = float(r["mz"]), float(r["rt"])
        cell = look(mz, rt)
        row = dict(label_id=int(r["id"]), mz=mz, rt=rt)
        for k in list(ref) + PUBLISHED:
            row[f"hit_{k}"] = bool(r[f"hit_{k}"])
        d10, d30 = desc10[i], eic_desc(c, mz, 30.0, rt, EIC_HALF, RT_TOL, P.floor, cell)
        row.update(raw_apex10=d10["apex_h"], raw_snr_cell10=d10["snr_cell"], raw_snr10=d10["snr"], raw_contig10=d10["n_contig"],
                   raw_fwhm_s10=d10["fwhm_s"], raw_localmax10=d10["local_max"], raw_drt_s10=d10["drt_s"], raw_level_out10=d10["level_out"],
                   raw_apex30=d30["apex_h"], raw_snr_cell30=d30["snr_cell"], raw_contig30=d30["n_contig"], raw_localmax30=d30["local_max"],
                   cell_noise=cell, floor=P.floor)
        kf = near(KEPT, mz, rt, NEAR_PPM, NEAR_RT)
        kctx = near(KEPT, mz, rt, CTX_PPM, CTX_RT)
        rf = near(RJ, mz, rt, PPM, RT_TOL)
        rctx = near(RJ, mz, rt, CTX_PPM, CTX_RT)
        if len(kf):
            j = np.argmin(np.hypot(kf["dppm"].values / NEAR_PPM, kf["drt_s"].values / (60 * NEAR_RT)))
            x = kf.iloc[j]
            row.update(near_dppm=float(x["dppm"]), near_drt_s=float(x["drt_s"]), near_in_span=bool(x["rt_min"] <= rt <= x["rt_max"]),
                       near_height=float(x["height"]), near_fwhm_s=float(60 * x["fwhm"]), near_score=float(x["score"]),
                       near_flags=str(x["flags"]) if isinstance(x["flags"], str) else "")
        elif len(kctx):
            j = np.argmin(np.hypot(kctx["dppm"].values / CTX_PPM, kctx["drt_s"].values / (60 * CTX_RT)))
            x = kctx.iloc[j]
            row.update(ctx_dppm=float(x["dppm"]), ctx_drt_s=float(x["drt_s"]), ctx_in_span=bool(x["rt_min"] <= rt <= x["rt_max"]),
                       ctx_height=float(x["height"]))
        best = None
        if len(rf):
            rf = rf.assign(rank=[STAGE_RANK.get(v, 0) for v in rf["stage"]])
            best = rf.sort_values(["rank", "height"], ascending=[False, False]).iloc[0]
        # classification
        if len(kf):
            cls = "near"
            sub = ("rt_off" if abs(row["near_dppm"]) <= PPM else "mz_off" if abs(row["near_drt_s"]) <= 60 * RT_TOL else "both")
            if sub == "rt_off" and row["near_in_span"]:
                sub = "rt_off_in_span"
        elif best is not None:
            cls, sub = "rejected", gate_short(best, P)
            row.update(rej_stage=best["stage"], rej_why="; ".join(gate_reasons(best, P)), rej_height=float(best["height"]),
                       rej_snr=float(best["snr"]), rej_n_scans=int(best["n_scans"]), rej_fwhm_s=float(60 * best["fwhm"]),
                       rej_far_level=float(best.get("far_level", np.nan)), rej_eic_snr=float(best.get("eic_snr", np.nan)),
                       rej_score=float(best.get("score_rule", np.nan)), rej_dppm=float(best["dppm"]), rej_drt_s=float(best["drt_s"]),
                       rej_noise=float(best["noise"]), rej_prom=float(best["prominence_rel"]), rej_ridge=float(best.get("ridge_ratio", np.nan)),
                       rej_background=float(best.get("background_ratio", np.nan)), rej_mz_sd=float(best["mz_sd_ppm"]))
        elif len(kctx):
            cls = "neighbour"
            sub = "in_span" if row.get("ctx_in_span") else "nearby"
        else:
            cls = "nocand"
            ok10 = d10["local_max"] and d10["snr_cell"] >= 3 and d10["n_contig"] >= 3 and abs(d10["drt_s"]) <= 60 * RT_TOL
            ok30 = d30["local_max"] and d30["snr_cell"] >= 3 and d30["n_contig"] >= 3 and abs(d30["drt_s"]) <= 60 * RT_TOL
            sub = ("raw_peak" if ok10 else "mz_off_raw" if ok30 else "raw_weak" if d10["apex_h"] > 0 else "raw_none")
        row.update(cls=cls, sub=sub)
        rows.append(row)
        key = f"{cls}:{sub}"
        panel = dict(rt=d10["rt"], e=d10["e"], e_wide=d30["e"], rt_truth=rt, tol=RT_TOL, noise=cell,
                     kept=[dict(rt_min=float(x.rt_min), rt_max=float(x.rt_max), label=f"kept {x.dppm:+.0f}ppm sc{x.score:.2f}")
                           for x in kctx.itertuples()],
                     rejected=[dict(rt_min=float(x.rt_min), rt_max=float(x.rt_max), label=f"{x.stage} {x.dppm:+.0f}ppm")
                               for x in rctx.itertuples()],
                     label=f"apex {d10['apex_h']:.2g} S/N_cell {d10['snr_cell']:.1f} n{d10['n_contig']} fwhm {d10['fwhm_s']:.1f}s dRT {d10['drt_s']:+.1f}s")
        gallery.setdefault(key, []).append(dict(title=f"#{int(r['id'])} {mz:.4f}@{rt:.2f}", panels=[panel]))
        if n_done % 50 == 0:
            log("traced", n_done, "of", len(miss))
    M = pd.DataFrame(rows)
    M.to_csv(OUT / "idsl_tp_misses.tsv", sep="\t", index=False)
    T.to_csv(OUT / "idsl_tp_all.tsv", sep="\t", index=False)
    for key, items in gallery.items():
        pick = [items[i] for i in rng.choice(len(items), size=min(12, len(items)), replace=False)]
        plot_gallery(pick, OUT / f"gallery_{key.replace(':', '_')}.png", f"{RUN} missed TP labels: {key} ({len(items)})", per_row=3)
    with open(OUT / "summary.log", "w") as fh:
        def w(*a):
            s = " ".join(str(x) for x in a)
            fh.write(s + "\n")
            print(s, flush=True)
        w(f"IDSL003 clean TP labels {len(T)}; peak3d (iter2) hits {int(T['hit'].sum())} = {T['hit'].mean():.3f} at {PPM} ppm / {RT_TOL} min")
        w("others on the same labels:", {k: f"{T[f'hit_{k}'].mean():.3f}" for k in list(ref) + PUBLISHED})
        w(f"matched-feature offsets on hits: median drt {np.median(drt_hit):+.2f} s (IQR {np.percentile(drt_hit, 25):+.2f}..{np.percentile(drt_hit, 75):+.2f}), "
          f"median dppm {np.median(dppm_hit):+.2f} (IQR {np.percentile(dppm_hit, 25):+.2f}..{np.percentile(dppm_hit, 75):+.2f})")
        w("\n== raw EIC at the label (10 ppm, +/- 0.1 min): hits vs misses ==")
        for name, x in (("hits", T[T["hit"]]), ("misses", T[~T["hit"]])):
            w(f"  {name} n={len(x)}: apex quantiles {np.percentile(x['raw_apex10'], [10, 25, 50, 75, 90]).round(0).tolist()}, "
              f"S/N_cell quantiles {np.percentile(x['raw_snr_cell10'], [10, 25, 50, 75, 90]).round(1).tolist()}, "
              f"contig scans quantiles {np.percentile(x['raw_contig10'], [10, 25, 50, 75, 90]).round(0).tolist()}, "
              f"local max {x['raw_localmax10'].mean():.2f}, |dRT| median {np.nanmedian(np.abs(x['raw_drt_s10'])):.1f} s")
        ab = pd.cut(np.log10(np.maximum(T["raw_apex10"], 1)), [0, 2.5, 3, 3.5, 4, 5, 9])
        w("recall by raw apex height (10 ppm):")
        w(T.groupby(ab, observed=True)["hit"].agg(["size", "mean"]).round(3).to_string())
        w("\n== missed TP labels by class ==")
        w(M.groupby(["cls", "sub"]).size().to_string())
        w("\n== per class: found by IPA (published) / our idslipa / masscube / oct05; raw apex, S/N_cell, scans ==")
        cols = {f"{k}": (f"hit_{k}", "mean") for k in ("IDSL.IPA", "idslipa", "masscube", *LOCAL_TOOLS, "oct05") if f"hit_{k}" in M}
        w(M.groupby(["cls", "sub"]).agg(n=("label_id", "size"), med_apex=("raw_apex10", "median"), med_snr_cell=("raw_snr_cell10", "median"),
                                        med_contig=("raw_contig10", "median"), med_fwhm_s=("raw_fwhm_s10", "median"), **cols).round(2).to_string())
        w("\n== found by any published tool among the misses ==")
        anyp = M[[f"hit_{c}" for c in PUBLISHED]].any(axis=1)
        w(f"  {int(anyp.sum())} of {len(M)} misses are reported by >= 1 published tool; by IPA {int(M['hit_IDSL.IPA'].sum())}; by none {int((~anyp).sum())}")
        if "near" in M["cls"].values:
            x = M[M["cls"] == "near"]
            w("\n== near misses (kept feature within 20 ppm / 0.2 min) ==")
            w(x.groupby("sub").agg(n=("label_id", "size"), med_dppm=("near_dppm", "median"), med_drt_s=("near_drt_s", "median"),
                                   in_span=("near_in_span", "mean"), med_height=("near_height", "median"), med_fwhm_s=("near_fwhm_s", "median"),
                                   raw_drt_s=("raw_drt_s10", "median")).round(2).to_string())
            w("  |drt| quantiles (s):", np.percentile(np.abs(x["near_drt_s"]), [25, 50, 75, 90]).round(1).tolist(),
              " |dppm| quantiles:", np.percentile(np.abs(x["near_dppm"]), [25, 50, 75, 90]).round(1).tolist())
        if "rejected" in M["cls"].values:
            x = M[M["cls"] == "rejected"]
            w("\n== rejected candidates: main reason ==")
            w(x.groupby("sub").agg(n=("label_id", "size"), med_height=("rej_height", "median"), med_snr=("rej_snr", "median"),
                                   med_eic_snr=("rej_eic_snr", "median"), med_far=("rej_far_level", "median"), med_ridge=("rej_ridge", "median"),
                                   med_bg=("rej_background", "median"), med_fwhm_s=("rej_fwhm_s", "median"), med_n_scans=("rej_n_scans", "median"),
                                   med_prom=("rej_prom", "median"), med_score=("rej_score", "median"), med_cell=("rej_noise", "median"),
                                   ipa=("hit_IDSL.IPA", "mean")).round(2).to_string())
            w("  examples (first 15):")
            for q in x.head(15).itertuples():
                w(f"   #{q.label_id} {q.mz:.4f}@{q.rt:.2f} apex {q.raw_apex10:.3g} S/N_cell {q.raw_snr_cell10:.1f}: {q.rej_stage} {q.rej_why}")
        if "nocand" in M["cls"].values:
            x = M[M["cls"] == "nocand"]
            w("\n== no candidate: raw evidence ==")
            w(x.groupby("sub").agg(n=("label_id", "size"), med_apex=("raw_apex10", "median"), med_snr_cell=("raw_snr_cell10", "median"),
                                   med_contig=("raw_contig10", "median"), med_apex30=("raw_apex30", "median"), ipa=("hit_IDSL.IPA", "mean"),
                                   oct05=("hit_oct05", "mean") if "hit_oct05" in x else ("label_id", "size")).round(2).to_string())
        if "neighbour" in M["cls"].values:
            x = M[M["cls"] == "neighbour"]
            w("\n== only a kept feature 20-30 ppm / 0.2-0.3 min away ==")
            w(x.groupby("sub").agg(n=("label_id", "size"), med_dppm=("ctx_dppm", "median"), med_drt_s=("ctx_drt_s", "median"),
                                   med_apex=("raw_apex10", "median")).round(2).to_string())
    log("wrote", OUT)


if __name__ == "__main__":
    main()
