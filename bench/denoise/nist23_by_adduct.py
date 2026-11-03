"""NIST23 denoised-vs-raw similarity by adduct, for T1 (formula given), T0 top-5 (formula inferred) and the formula-free rule.

usage: python bench/denoise/nist23_by_adduct.py run POS.npz NEG.npz OUT.csv [--per-adduct 1500] [--min-count 1000] [--workers N]
       python bench/denoise/nist23_by_adduct.py fig OUT.csv OUT.png OUT.md

Singly charged adducts with at least --min-count spectra; up to --per-adduct spectra sampled per adduct (seed 0); only
spectra whose formula + adduct agree with the precursor m/z (so T1 applies). Similarity = weighted entropy similarity
(Li et al. 2021, 0.02 Da) of the denoised spectrum against the raw one."""
import argparse, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from specdenoise import formula as fm
from specdenoise.denoise import denoise, Params
from specdenoise.similarity import similarity

G = {}
MODES = ["t1", "t0_top5", "free"]


def one(i):
    off = G["off"]; a, b = int(off[i]), int(off[i + 1])
    mz = G["mz"][a:b].astype(np.float64); it = G["it"][a:b].astype(np.float64)
    o = np.argsort(mz); mz = mz[o]; it = it[o]
    f = fm.parse(G["formula"][i]); ptype = str(G["ptype"][i]); pmz = float(G["pmz"][i])
    pa = fm.precursor_formula(f, ptype) if f is not None else None
    if pa is None or len(mz) == 0:
        return None
    p, z = pa
    if abs(z) != 1 or abs(fm.ion_mz(p, z) - pmz) > 0.02:
        return None
    P = Params()
    r1 = denoise(mz, it, pmz, formula=G["formula"][i], adduct=ptype, params=P)
    if r1.tier != 1:
        return None
    r0 = denoise(mz, it, pmz, charge=z, params=P)
    keeps = {"t1": r1.keep, "t0_top5": r0.keep, "free": r0.variant(free=True)}
    tot_i = it.sum()
    row = dict(pol=G["pol"], adduct=ptype, pmz=pmz, nion=len(mz), ncand=int(len(r0.candidates)) if r0.candidates is not None else 0,
               overflow=int(r0.overflow))
    for m, k in keeps.items():
        s = similarity(mz[k], it[k], mz, it, "weighted_entropy", 0.02, pmz, pmz)[0] if k.any() else 0.0
        row[f"sim_{m}"] = s; row[f"ions_{m}"] = k.mean(); row[f"inten_{m}"] = it[k].sum() / tot_i
    return row


def run(a):
    import pandas as pd
    rows = []
    for pol, path in (("pos", a.pos), ("neg", a.neg)):
        t0 = time.time()
        z = np.load(path, allow_pickle=True)
        G["off"] = z["peak_offsets"]; G["mz"] = z["peak_mz"]; G["it"] = z["peak_intensity"]
        G["formula"] = z["formula"]; G["ptype"] = z["precursor_type"]; G["pmz"] = z["precursor_mz"]; G["pol"] = pol
        ptype = np.asarray(G["ptype"]).astype(str)
        u, c = np.unique(ptype, return_counts=True)
        rng = np.random.default_rng(0)
        idx = []; counts = {}
        for t, n in zip(u, c):
            pa = fm.parse_adduct(t)
            if n < a.min_count or pa is None or abs(pa[2]) != 1:
                continue
            counts[t] = int(n)
            w = np.flatnonzero(ptype == t)
            idx += list(rng.choice(w, min(a.per_adduct, n), replace=False))
        print(f"{pol}: {len(ptype)} spectra loaded in {time.time() - t0:.0f} s; {len(counts)} adducts, {len(idx)} sampled", flush=True)
        one(idx[0])
        with Pool(a.workers) as pool:
            res = [r for r in pool.imap_unordered(one, idx, chunksize=20) if r is not None]
        for r in res:
            r["in_nist"] = counts[r["adduct"]]
        rows += res
        print(f"{pol}: {len(res)} spectra done in {time.time() - t0:.0f} s", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


def fig(a):
    import pandas as pd
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from figures import C, NAME, panel_label, title, TEXT2
    R = pd.read_csv(a.csv)
    lines = ["# NIST23 denoised vs raw by adduct (weighted entropy similarity, 0.02 Da)", "",
             "| polarity | adduct | in NIST | tested | T1 mean | T1 median | T1 > 0.95 | T0 top-5 mean | T0 median | T0 > 0.95 | free mean | "
             "T1 intensity kept | T0 intensity kept | free intensity kept | T0 median candidates | T0 overflow |", "|" + "---|" * 16]
    pols = [("pos", "Positive"), ("neg", "Negative")]
    nrows = {pol: R[R.pol == pol].adduct.nunique() for pol, _ in pols}
    XMIN = 0.8
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 0.2 * (nrows["pos"] + nrows["neg"]) + 2.6),
                             gridspec_kw={"height_ratios": [nrows["pos"] + 1.5, nrows["neg"] + 1.5], "hspace": 0.12})
    for ax, (pol, label) in zip(axes, pols):
        S = R[R.pol == pol]
        ads = S.groupby("adduct")["in_nist"].first().sort_values(ascending=False).index.tolist()
        y = np.arange(len(ads))[::-1]
        for yi in y:
            ax.axhline(yi, color="#eeede9", lw=0.6, zorder=0)
        ax.axvline(0.95, color="#c9c8c3", lw=0.8, ls="--", zorder=0)
        for j, m in enumerate(MODES):
            v = np.array([S[S.adduct == t][f"sim_{m}"].mean() for t in ads])
            inside = v >= XMIN
            ax.scatter(v[inside], y[inside], s=26, color=C[m], label=NAME[m], zorder=3, edgecolor="white", lw=0.6)
            for yi, vi in zip(y[~inside], v[~inside]):            # off-scale means: clipped marker + value
                ax.scatter([XMIN + 0.003], [yi], s=34, marker="<", color=C[m], zorder=3)
                ax.text(XMIN + 0.012, yi, f"{vi:.2f}", va="center", fontsize=6.5, color=C[m])
        ax.set_yticks(y); ax.set_yticklabels(ads, fontsize=7)
        for yi, t in zip(y, ads):
            T = S[S.adduct == t]
            ax.text(1.003, yi, f"{int(T.in_nist.iloc[0]):,}", va="center", ha="left", fontsize=6, color=TEXT2)
        ax.set_xlim(XMIN, 1.0); ax.set_ylim(-0.8, len(ads) + 0.6)
        ax.grid(axis="y", visible=False); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
        ax.text(1.003, len(ads) - 0.3, "in NIST23", va="bottom", ha="left", fontsize=6, color=TEXT2)
        title(ax, f"{label} mode: mean similarity of the denoised spectrum to the raw one, by adduct ({len(S):,} spectra, up to {int(S.groupby('adduct').size().max())} per adduct)")
        for t in ads:
            T = S[S.adduct == t]
            lines.append(f"| {pol} | {t} | {int(T.in_nist.iloc[0])} | {len(T)} | {T.sim_t1.mean():.3f} | {T.sim_t1.median():.3f} | "
                         f"{100 * (T.sim_t1 > 0.95).mean():.1f} % | {T.sim_t0_top5.mean():.3f} | {T.sim_t0_top5.median():.3f} | "
                         f"{100 * (T.sim_t0_top5 > 0.95).mean():.1f} % | {T.sim_free.mean():.3f} | {100 * T.inten_t1.mean():.1f} % | "
                         f"{100 * T.inten_t0_top5.mean():.1f} % | {100 * T.inten_free.mean():.1f} % | {int(T.ncand.median())} | {100 * T.overflow.mean():.1f} % |")
    axes[0].set_xlabel(""); axes[1].set_xlabel("Weighted entropy similarity, denoised vs raw (mean over spectra)")
    axes[0].legend(fontsize=7.5, loc="lower left", ncol=3, bbox_to_anchor=(0, 1.02))
    panel_label(axes[0], "a"); panel_label(axes[1], "b")
    fig.savefig(a.png, bbox_inches="tight"); plt.close(fig)
    open(a.md, "w").write("\n".join(lines) + "\n"); print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("pos"); r.add_argument("neg"); r.add_argument("out")
    r.add_argument("--per-adduct", type=int, default=1500); r.add_argument("--min-count", type=int, default=1000)
    r.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    f = sub.add_parser("fig"); f.add_argument("csv"); f.add_argument("png"); f.add_argument("md")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else fig(a)


if __name__ == "__main__":
    main()
