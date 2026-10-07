# diagnostic: do the junk yardsticks (junk_classes.classify_file EIC check, rep_check replicate check) pass the
# ground-truth features that peak3d detected? Per run with truth: the kept feature matched to each truth entry
# (10 ppm / 0.1 min on the corrected RT, highest if several) is classified; failures are listed and drawn.
# usage: truth_junk_check.py RUN [RUN ...]      (reads results/peak3d, the mzML files; run in srun)
import json, os, sys
from multiprocessing import get_context
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench/diag"))
from bench.runs import RUNS
from fix_eval import eval_truth
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
    cloud = load_cloud(files[fi])
    F["noise"] = reference_noise(cloud)(F["mz"].values, F["rt"].values)
    C = classify_file(cloud, F, P)                      # EIC check on every kept feature (dup needs them all)
    T = eval_truth(run)
    mz, rtc, h = F["mz"].values, F["rt_corr"].values, F["height"].values
    o = np.argsort(mz); smz = mz[o]
    rows = []
    for ti, (m, t) in enumerate(zip(T["mz"].values, T["rt"].values)):
        lo = np.searchsorted(smz, m * (1 - 1e-5)); hi = np.searchsorted(smz, m * (1 + 1e-5), side="right")
        k = o[lo:hi]; k = k[np.abs(rtc[k] - t) <= 0.1]
        if len(k):
            rows.append((ti, int(k[np.argmax(h[k])])))
    if not rows:
        return run, stem, None, None
    ti, idx = map(np.array, zip(*rows))
    sub = F.iloc[idx].copy()
    sub["_top"] = C["top_s"].values[idx]
    others = [(load_cloud(files[k]), pd.read_csv(d / "rt_correction" / f"{stems[k]}.tsv", sep="\t"),
               json.load(open(d / "features" / f"{stems[k]}.params.json"))) for k in range(len(files)) if k != fi]
    rep = rep_check(sub, P, others)
    R = pd.DataFrame({"truth_i": ti, "row": idx, "cls": C["cls"].values[idx], "rep_frac": rep,
                      "snr_s": C["snr_s"].values[idx], "prom_s_rel": C["prom_s_rel"].values[idx]})
    for c in ("mz", "rt", "rt_corr", "height", "snr", "n_scans", "fwhm", "score", "iso_offset", "flags", "scan_apex", "scan_min", "scan_max", "noise"):
        R[c] = F[c].values[idx]
    R["fwhm_rel"] = R["fwhm"] / P["fwhm_med"]
    allc = pd.DataFrame({"cls": C["cls"].values})
    return run, stem, R, allc


def gallery(run, fails, path):
    files = RUNS[run]["files"]; stems = [Path(f).stem for f in files]
    d = ROOT / "results/peak3d" / run / "peak3d_out"
    pick = fails.sample(min(30, len(fails)), random_state=0)
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
        ax.set_title(f"{st[-12:]} m/z {f['mz']:.4f} rt {f['rt']:.2f} h {f['height']:.3g} snr {f['snr']:.0f} iso {int(f['iso_offset'])}\n"
                     f"cls {f['cls']} rep {f['rep_frac']:.2f} snr_s {f['snr_s']:.1f} fwhm {f['fwhm_rel']:.1f}x [{f['flags']}]", fontsize=6.5, loc="left")
        ax.tick_params(labelsize=5)
    for ax in axes.ravel()[len(pick):]: ax.axis("off")
    fig.suptitle(f"{run}: truth features that the junk yardsticks FAIL (blue EIC at the feature tolerance, grey 30 ppm, orange bounds, red noise floor)", fontsize=9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main(runs):
    jobs = [(r, k) for r in runs for k in range(len(RUNS[r]["files"]))]
    from peak3d.pipeline import _warm_jit
    _warm_jit()
    with get_context("spawn").Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))) as pool:
        res = pool.map(one, jobs, chunksize=1)
    pd.set_option("display.width", 220)
    for run in runs:
        parts = [(s, R, A) for r, s, R, A in res if r == run and R is not None]
        R = pd.concat([x[1].assign(stem=x[0]) for x in parts], ignore_index=True)
        A = pd.concat([x[2] for x in parts], ignore_index=True)
        n_files = len(RUNS[run]["files"])
        eic_ok = R["cls"] == "ok"; rep_half = R["rep_frac"] >= 0.5; rep_zero = R["rep_frac"] == 0
        junk = rep_zero | ~eic_ok; good = rep_half & eic_ok
        print(f"\n================ {run}: {n_files} files, {len(R)} truth matches (truth entries detected, counted per file)")
        print(f"  pass EIC check: {eic_ok.sum()} ({eic_ok.mean():.1%})   replicate peak in >= half of the other files: {rep_half.sum()} ({rep_half.mean():.1%})   "
              f"in none: {rep_zero.sum()} ({rep_zero.mean():.1%})")
        print(f"  judged JUNK (no replicate or EIC fail): {junk.sum()} ({junk.mean():.1%})   judged GOOD: {good.sum()} ({good.mean():.1%})   "
              f"neither: {(~junk & ~good).sum()}")
        print(f"  for contrast, EIC check on ALL {len(A)} kept features of the run: fail {(A['cls'] != 'ok').mean():.1%}")
        print("  EIC classes among truth matches: " + ", ".join(f"{k} {v}" for k, v in R["cls"].value_counts().items()))
        f = R[~eic_ok]
        if len(f):
            print(f"  EIC failures: median height {f['height'].median():.3g} (all truth matches {R['height'].median():.3g}), median snr {f['snr'].median():.1f}, "
                  f"isotopologue share {(f['iso_offset'] > 0).mean():.2f} (all {(R['iso_offset'] > 0).mean():.2f}), fwhm_rel median {f['fwhm_rel'].median():.2f}, "
                  f"flags: " + ", ".join(f"{k} {v}" for k, v in f["flags"].fillna("").str.split(";").explode().value_counts().head(5).items()))
        z = R[rep_zero]
        if len(z):
            print(f"  no-replicate cases: median height {z['height'].median():.3g}, median snr {z['snr'].median():.1f}, EIC ok among them {(z['cls'] == 'ok').mean():.2f}")
        R.to_csv(OUT / f"truth_junk_{run}.tsv", sep="\t", index=False, float_format="%.6g")
        fails = R[junk]
        if len(fails):
            gallery(run, fails, OUT / f"truth_junk_fail_{run}.png")


if __name__ == "__main__":
    main(sys.argv[1:])
