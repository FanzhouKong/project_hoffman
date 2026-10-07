# diagnostic: after iteration 1 (chromatogram check + presence rule), which kept features of the replicate
# runs are still judged junk by the yardsticks, and why? Only features whose group passed the presence rule
# (group_kept) count. Per run: class breakdown of the remaining junk, descriptor medians of junk vs good,
# the share of junk that is single-file (n_detected 1, kept through confirmation) and a gallery.
# usage: remaining_junk.py RUN [RUN ...]
import json, os, sys
from multiprocessing import get_context
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench/diag"))
from bench.runs import RUNS
from junk_classes import classify_file, rep_check
from version_junk import reference_noise
from peak3d.io import load_cloud
OUT = ROOT / "results/diag_idsl"


def one(job):
    run, fi = job
    files = RUNS[run]["files"]; stems = [Path(f).stem for f in files]; stem = stems[fi]
    d = ROOT / "results/peak3d" / run / "peak3d_out"
    F = pd.read_csv(d / "features" / f"{stem}.tsv", sep="\t")
    P = json.load(open(d / "features" / f"{stem}.params.json"))
    G = pd.read_csv(d / "presence_mask.tsv", sep="\t", usecols=["group_id", "n_detected", "n_confirmed"]).set_index("group_id")
    cloud = load_cloud(files[fi])
    F["noise_ref"] = reference_noise(cloud)(F["mz"].values, F["rt"].values)
    F2 = F.rename(columns={"noise": "noise_pick", "noise_ref": "noise"})
    C = classify_file(cloud, F2, P)
    others = [(load_cloud(files[k]), pd.read_csv(d / "rt_correction" / f"{stems[k]}.tsv", sep="\t"),
               json.load(open(d / "features" / f"{stems[k]}.params.json"))) for k in range(len(files)) if k != fi]
    rep = rep_check(F2.assign(_top=C["top_s"].values), P, others)
    R = F.copy()
    R["cls"] = C["cls"].values; R["rep_frac"] = rep; R["snr_s"] = C["snr_s"].values
    R["n_detected"] = G.loc[R["group_id"].values, "n_detected"].values
    R["n_confirmed"] = G.loc[R["group_id"].values, "n_confirmed"].values
    R["stem"] = stem
    R["fwhm_rel"] = R["fwhm"] / P["fwhm_med"]
    return run, R


def gallery(run, J, path, title):
    files = RUNS[run]["files"]; stems = [Path(f).stem for f in files]
    d = ROOT / "results/peak3d" / run / "peak3d_out"
    pick = J.sample(min(40, len(J)), random_state=0)
    ncol = 5; nrow = int(np.ceil(len(pick) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 2.6 * nrow), dpi=110, squeeze=False)
    clouds = {}
    for ax, (_, f) in zip(axes.ravel(), pick.iterrows()):
        st = f["stem"]
        if st not in clouds:
            clouds[st] = load_cloud(files[stems.index(st)])
        cloud = clouds[st]; P = json.load(open(d / "features" / f"{st}.params.json"))
        sa, slo, shi = int(f["scan_apex"]), int(f["scan_min"]), int(f["scan_max"])
        W = int(max(10 * P["fwhm_scans"], 30)); s0, s1 = max(0, sa - W), min(cloud.n_scans, sa + W + 1)
        sig = np.sqrt(P["sig_a"] ** 2 + P["sig_b"] ** 2 / max(f["height"], 1)); ppm = max(5.0, 3 * sig + 1)
        x = cloud.rt[s0:s1] * 60
        ax.plot(x, cloud.eic(float(f["mz"]), 30.0, s0, s1), color="#bbbbbb", lw=0.8)
        ax.plot(x, cloud.eic(float(f["mz"]), ppm, s0, s1), color="#0969da", lw=1.0, marker=".", ms=2.5)
        ax.axvspan(cloud.rt[slo] * 60, cloud.rt[shi] * 60, color="#bf8700", alpha=0.15)
        ax.axhline(f["noise"], color="#cf222e", lw=0.6, ls="--")
        ax.set_title(f"{st[-10:]} m/z {f['mz']:.4f} rt {f['rt']:.2f} h {f['height']:.3g} snr {f['snr']:.0f} eic {f['eic_snr']:.0f} sc {f['score']:.2f}\n"
                     f"cls {f['cls']} rep {f['rep_frac']:.2f} det {int(f['n_detected'])}+{int(f['n_confirmed'])} fwhm {f['fwhm_rel']:.1f}x tv {f['tv_ratio']:.1f} far {f['far_level']:.2f} iso {int(f['iso_offset'])}", fontsize=6.3, loc="left")
        ax.tick_params(labelsize=5)
    for ax in axes.ravel()[len(pick):]: ax.axis("off")
    fig.suptitle(title, fontsize=9); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main(runs):
    jobs = [(r, k) for r in runs for k in range(len(RUNS[r]["files"]))]
    from peak3d.pipeline import _warm_jit
    _warm_jit()
    with get_context("spawn").Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))) as pool:
        res = pool.map(one, jobs, chunksize=1)
    pd.set_option("display.width", 230)
    for run in runs:
        R = pd.concat([x[1] for x in res if x[0] == run], ignore_index=True)
        K_ = R[R["group_kept"]]
        junk = (K_["rep_frac"] == 0) | (K_["cls"] != "ok")
        good = (K_["rep_frac"] >= 0.5) & (K_["cls"] == "ok")
        print(f"\n================ {run}: per-file features {len(R)}, in kept groups {len(K_)} ({len(K_) / len(R):.1%}); "
              f"junk among kept {junk.sum()} ({junk.mean():.1%}), good {good.sum()} ({good.mean():.1%})")
        print("  junk removed by the presence rule alone (features in dropped groups): %d, of which yardstick-junk %.1f%%" % (
            (~R["group_kept"]).sum(), 100 * (((R["rep_frac"] == 0) | (R["cls"] != "ok")) & ~R["group_kept"]).mean() / max((~R["group_kept"]).mean(), 1e-9)))
        J = K_[junk]
        print("  remaining junk by yardstick class: " + ", ".join(f"{k} {v}" for k, v in J["cls"].value_counts().items()) +
              f"; no replicate peak but EIC ok: {int(((J['rep_frac'] == 0) & (J['cls'] == 'ok')).sum())}")
        print("  remaining junk by presence: " + ", ".join(f"det {int(k)}: {v}" for k, v in J["n_detected"].value_counts().sort_index().items()) +
              f"  (kept via confirmation only, det=1: {int((J['n_detected'] == 1).sum())})")
        cols = ["height", "snr", "eic_snr", "score", "n_scans", "fwhm_rel", "mz_sd_ppm", "gauss_r2", "tv_ratio", "far_level", "n_valleys", "ridge_ratio", "prominence_rel"]
        T = pd.DataFrame({"junk_med": J[cols].median(), "good_med": K_[good][cols].median(),
                          "junk_q90": J[cols].quantile(0.9), "good_q10": K_[good][cols].quantile(0.1)})
        print(T.round(3).T.to_string())
        print("  isotopologue share junk %.2f good %.2f; flags among junk: %s" % ((J["iso_offset"] > 0).mean(), (K_[good]["iso_offset"] > 0).mean(),
              ", ".join(f"{k} {v}" for k, v in J["flags"].fillna("").str.split(";").explode().value_counts().head(6).items())))
        R.to_csv(OUT / f"remaining_{run}.tsv", sep="\t", index=False, float_format="%.6g")
        if len(J):
            gallery(run, J, OUT / f"remaining_junk_{run}.png", f"{run}: kept features still judged junk after iteration 1")
            gallery(run, J[J["cls"] == "ok"], OUT / f"remaining_junk_norep_{run}.png", f"{run}: kept features with a clean EIC but no replicate peak")


if __name__ == "__main__":
    main(sys.argv[1:])
