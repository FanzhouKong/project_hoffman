# diagnostic: blind visual review of a stratified random sample of IDSL.IPA labels. Panels show only a code: 003 EIC at
# 10 ppm (blue, label point = red dot), 30 ppm (grey), 6 other injections at 10 ppm (orange), label RT (red line).
# Calls (fixed before viewing):
#   P  chromatographic peak with its apex at the point (within ~2 s), clearly above the local trace
#   W  weak / questionable peak at the point (few points, flicker level, barely above the local trace)
#   T  point on the flank / tail / shoulder of a larger peak whose apex is > ~3 s away
#   N  no peak at the point in the 10 ppm trace: noise, flicker, background plateau or dips, nothing
#   M  a peak at the point only in the 30 ppm trace (a nearby m/z), not in the label's 10 ppm trace
# usage: idsl_label_blind.py render AUDIT_DIR      -> AUDIT_DIR/blind/sheet_XX.png + key.tsv   (reads mzML -> srun)
#        idsl_label_blind.py score  AUDIT_DIR      (reads blind/calls.tsv: code<TAB>call)
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

mode, D = sys.argv[1], Path(sys.argv[2])
B = D / "blind"
B.mkdir(exist_ok=True)
STRATA = [("TN confirmed", 40), ("TN uncertain", 30),
          ("TN contradicted|TN doubtful|TN off-apex|TN conflicting|TN defensible", 40),
          ("TP confirmed", 30), ("TP probable|TP plausible", 30), ("TP misplaced", 20), ("TP uncertain", 30),
          ("TP unsupported", 40)]

if mode == "render":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from bench.msdata import load_run
    A = pd.read_csv(D / "verdict.tsv", sep="\t")
    rng = np.random.default_rng(7)
    picks = []
    for pat, n in STRATA:
        sel = A[A.verdict.str.contains(pat)]
        idx = rng.choice(len(sel), min(n, len(sel)), replace=False)
        s = sel.iloc[idx].copy()
        s["stratum"] = pat
        s["stratum_size"] = len(sel)
        picks.append(s)
    K = pd.concat(picks).sample(frac=1, random_state=11).reset_index(drop=True)
    K["code"] = [f"{i // 30 + 1:02d}-{i % 30 + 1:02d}" for i in range(len(K))]
    K[["code", "id", "m/z", "RT(min)", "Manual Curation", "verdict", "stratum", "stratum_size"]].to_csv(
        B / "key.tsv", sep="\t", index=False)
    run = load_run(ROOT / "data/raw/IDSL_IPA/003.mzML")
    reps = [load_run(ROOT / f"data/mzml/MTBLS1684/{s}.mzML") for s in ("004", "007", "010", "015", "020", "025")]
    HALF = 0.4
    for sheet, G in K.groupby(K.code.str[:2]):
        fig, axes = plt.subplots(6, 5, figsize=(17, 15), dpi=100, squeeze=False)
        for ax, (_, f) in zip(axes.ravel(), G.iterrows()):
            mz, rt = float(f["m/z"]), float(f["RT(min)"])
            for r in reps:
                lo, hi = r.scan_window(rt, HALF)
                ax.plot(r.rt[lo:hi] * 60, r.eic(mz, 10)[lo:hi], color="#e3a008", lw=0.5, alpha=0.7)
            lo, hi = run.scan_window(rt, HALF)
            x = run.rt[lo:hi] * 60
            ax.plot(x, run.eic(mz, 30)[lo:hi], color="#aaaaaa", lw=0.7)
            e = run.eic(mz, 10)[lo:hi]
            ax.plot(x, e, color="#0969da", lw=1.0, marker=".", ms=2)
            k = int(np.argmin(np.abs(x - rt * 60)))
            ax.plot(x[k], e[k], "o", color="#cf222e", ms=4)
            ax.axvline(rt * 60, color="#cf222e", lw=0.6)
            ax.set_title(f["code"], fontsize=9, loc="left")
            ax.tick_params(labelsize=5)
        for ax in axes.ravel()[len(G):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(B / f"sheet_{sheet}.png")
        plt.close(fig)
    print(f"{len(K)} panels, {K.code.str[:2].nunique()} sheets -> {B}")

elif mode == "score":
    K = pd.read_csv(B / "key.tsv", sep="\t")
    C = pd.read_csv(B / "calls.tsv", sep="\t", names=["code", "call"], comment="#")
    K = K.merge(C, on="code", how="left")
    assert K.call.notna().all(), K[K.call.isna()].code.tolist()
    pd.set_option("display.width", 200)
    print("== blind calls by stratum (counts)")
    print(pd.crosstab([K.stratum, K["Manual Curation"]], K.call, margins=True))
    # label agrees with the blind call: TP <-> P, TN <-> N/M; W and T are judgement / matching-rule cases
    agree = np.where(K.call.isin(["W", "T"]), np.nan,
                     ((K["Manual Curation"] == "TP") == (K.call == "P")).astype(float))
    K["agree"] = agree
    print("\n== stratum-weighted estimates over all 20,000 labels (weights = stratum size / sample size)")
    K["w"] = K.stratum_size / K.groupby("stratum").code.transform("size")
    for lab in ("TP", "TN"):
        S = K[K["Manual Curation"] == lab]
        tot = S.w.sum()
        shares = S.groupby("call").w.sum() / tot
        print(f"  {lab} labels (covered by the sampled strata: {tot:.0f}):", {k: round(v, 3) for k, v in shares.items()})
        # bootstrap CI within strata
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(2000):
            parts = []
            for st, g in S.groupby("stratum"):
                parts.append(g.sample(len(g), replace=True, random_state=int(rng.integers(1 << 31))))
            bb = pd.concat(parts)
            boots.append((bb.groupby("call").w.sum() / bb.w.sum()).reindex(list("PWTNM")).fillna(0).values)
        lo, hi = np.quantile(np.array(boots), [0.025, 0.975], axis=0)
        print("    95% CI:", {c: (round(a, 3), round(b, 3)) for c, a, b in zip("PWTNM", lo, hi)})
    K.to_csv(B / "scored.tsv", sep="\t", index=False)
