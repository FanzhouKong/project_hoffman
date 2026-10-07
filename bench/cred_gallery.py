#!/usr/bin/env python3
"""QC for a credentialed truth set: image gallery for manual review, a review sheet,
and a known-metabolite check (the credentialed carbon count n must equal the formula's
carbon number for [M+H]+ matches).

Writes data/truth/<DS>/gallery/{tier_A,tier_B}/<truth_id>.png,
       data/truth/<DS>/gallery/review_sheet.csv   (fill in `verdict`: real / not_real / unsure)
       data/truth/<DS>/known_metabolite_check.tsv, data/truth/<DS>/qc_report.md
usage: cred_gallery.py DATASET [--n 100]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.msdata import C13, load_run  # noqa: E402

PPM = 5.0
MONO = {"C": 12.0, "H": 1.00782503207, "N": 14.0030740048, "O": 15.99491461956,
        "P": 30.97376163, "S": 31.97207100}
PROTON = 1.007276466812
# common primary metabolites present in both E. coli and yeast (neutral formulas)
KNOWN = {
    "glycine": "C2H5NO2", "alanine": "C3H7NO2", "serine": "C3H7NO3", "proline": "C5H9NO2",
    "valine/betaine": "C5H11NO2", "threonine/homoserine": "C4H9NO3", "cysteine": "C3H7NO2S",
    "leucine/isoleucine": "C6H13NO2", "asparagine": "C4H8N2O3", "aspartate": "C4H7NO4",
    "glutamine": "C5H10N2O3", "lysine": "C6H14N2O2", "glutamate": "C5H9NO4",
    "methionine": "C5H11NO2S", "histidine": "C6H9N3O2", "phenylalanine": "C9H11NO2",
    "arginine": "C6H14N4O2", "tyrosine": "C9H11NO3", "tryptophan": "C11H12N2O2",
    "ornithine": "C5H12N2O2", "citrulline": "C6H13N3O3", "pyroglutamate": "C5H7NO3",
    "adenine": "C5H5N5", "guanine": "C5H5N5O", "hypoxanthine": "C5H4N4O", "xanthine": "C5H4N4O2",
    "uracil": "C4H4N2O2", "cytosine": "C4H5N3O", "thymine": "C5H6N2O2",
    "adenosine": "C10H13N5O4", "guanosine": "C10H13N5O5", "uridine": "C9H12N2O6",
    "cytidine": "C9H13N3O5", "inosine": "C10H12N4O5", "AMP": "C10H14N5O7P",
    "GMP": "C10H14N5O8P", "UMP": "C9H13N2O9P", "CMP": "C9H14N3O8P",
    "glutathione": "C10H17N3O6S", "glutathione disulfide": "C20H32N6O12S2",
    "carnitine": "C7H15NO3", "acetylcarnitine": "C9H17NO4", "spermidine": "C7H19N3",
    "putrescine": "C4H12N2", "nicotinamide": "C6H6N2O", "pantothenate": "C9H17NO5",
    "S-adenosylhomocysteine": "C14H20N6O5S", "methylthioadenosine": "C11H15N5O3S",
    "N-acetylglutamate": "C7H11NO5", "riboflavin": "C17H20N4O6",
}


def mass(formula):
    return sum(MONO[e] * (int(c) if c else 1) for e, c in re.findall(r"([A-Z][a-z]?)(\d*)", formula))


def n_carbon(formula):
    m = re.search(r"C(\d*)(?![a-z])", formula)
    return int(m.group(1) or 1) if m else 0


def known_check(truth):
    rows = []
    mz = truth["mz"].values
    for name, f in KNOWN.items():
        target = mass(f) + PROTON
        hit = np.abs(mz - target) <= target * PPM * 1e-6
        for _, t in truth[hit].iterrows():
            rows.append(dict(metabolite=name, formula=f, mz_expected=round(target, 5),
                             truth_id=t["truth_id"], mz=round(t["mz"], 5), rt=round(t["rt"], 3),
                             tier=t["tier"], feature_class=t["feature_class"],
                             n_formula=n_carbon(f), n_credentialed=int(t["n"]),
                             agrees=int(t["n"]) == n_carbon(f)))
    return pd.DataFrame(rows)


def eic(run, mz, t0, t1):
    s0, s1 = np.searchsorted(run.rt, [t0, t1])
    tol = mz * PPM * 1e-6
    a, b = np.searchsorted(run.mz, [mz - tol, mz + tol])
    e = np.zeros(s1 - s0)
    sc = run.scan[a:b]
    k = (sc >= s0) & (sc < s1)
    np.maximum.at(e, sc[k] - s0, run.inten[a:b][k])
    return run.rt[s0:s1], e


def plot(row, R12, R13, off12, off13, fwhm_s, path):
    half = max(8 * fwhm_s / 60, 0.15)
    t0, t1 = row.rt - half, row.rt + half
    m, n = row.mz, int(row.n)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4), constrained_layout=True)
    cols = ["#1f77b4", "#2ca02c", "#9467bd"]
    mean12, mean13, mean_m1 = [], [], []
    grid = np.linspace(t0, t1, 300)
    for i, (r, o) in enumerate(zip(R12, off12)):
        t, e = eic(r, m, t0 + o, t1 + o)
        ax[0].plot(t - o, e, color=cols[i], lw=1.2, label=f"12C rep{i + 1}")
        mean12.append(np.interp(grid, t - o, e, left=0, right=0))
        t, e1 = eic(r, m + C13, t0 + o, t1 + o)
        mean_m1.append(np.interp(grid, t - o, e1, left=0, right=0))
    for i, (r, o) in enumerate(zip(R13, off13)):
        t, e = eic(r, m + n * C13, t0 + o, t1 + o)
        ax[1].plot(t - o, e, color=cols[i], lw=1.2, label=f"13C rep{i + 1}")
        mean13.append(np.interp(grid, t - o, e, left=0, right=0))
    for a in ax[:2]:
        a.axvline(row.rt, color="0.6", lw=0.8, ls=":")
        a.set_xlabel("RT (min, aligned)"); a.legend(fontsize=7, frameon=False)
    ax[0].set_title(f"12C  m/z {m:.4f}", fontsize=9)
    ax[1].set_title(f"13C  m/z {m + n * C13:.4f}  (+{n} x 13C)", fontsize=9)
    nz = lambda v: v / v.max() if v.max() > 0 else v  # noqa: E731
    ax[2].plot(grid, nz(np.mean(mean12, 0)), color="#d62728", lw=1.5, label="12C mean")
    ax[2].plot(grid, nz(np.mean(mean13, 0)), color="#1f77b4", lw=1.5, label="13C mean")
    m1 = np.mean(mean_m1, 0)
    if m1.max() > 0:
        ax[2].plot(grid, nz(m1), color="0.4", lw=1, ls="--", label="12C M+1")
    ax[2].axvline(row.rt, color="0.6", lw=0.8, ls=":")
    ax[2].set_title("normalized overlay", fontsize=9); ax[2].legend(fontsize=7, frameon=False)
    ax[2].set_xlabel("RT (min, aligned)")
    n_iso = "n/a" if pd.isna(row.n_iso) else f"{row.n_iso:.1f}"
    fig.suptitle(f"{row.truth_id}   tier {row.tier}   {row.feature_class}   RT {row.rt:.3f} min   "
                 f"n={n}   shape r={row.r:.2f}   n from M+1={n_iso}   "
                 f"height 12C {row.height12:.2e} / 13C {row.height13:.2e}", fontsize=9)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--n", type=int, default=100, help="images per tier")
    a = ap.parse_args()
    out = ROOT / "data/truth" / a.dataset
    p = json.loads((out / "params.json").read_text())
    truth = pd.read_csv(out / "credentialed.tsv", sep="\t")

    kc = known_check(truth)
    kc.to_csv(out / "known_metabolite_check.tsv", sep="\t", index=False)

    R12 = [load_run(f) for f in p["files_12C"]]
    R13 = [load_run(f) for f in p["files_13C"]]
    off12 = [o / 60 for o in p["rt_offsets_12C_s"]]
    off13 = [o / 60 for o in p["rt_offsets_13C_s"]]
    gal = out / "gallery"
    sheet = []
    rng = np.random.default_rng(0)
    for tier in ("A", "B"):
        d = gal / f"tier_{tier}"
        d.mkdir(parents=True, exist_ok=True)
        sub = truth[truth["tier"] == tier]
        pick = sub.iloc[np.sort(rng.choice(len(sub), size=min(a.n, len(sub)), replace=False))] if len(sub) else sub
        for row in pick.itertuples():
            f = d / f"{row.truth_id}.png"
            plot(row, R12, R13, off12, off13, p["FWHM_s"], f)
            sheet.append(dict(truth_id=row.truth_id, tier=tier, feature_class=row.feature_class,
                              mz=round(row.mz, 5), rt_min=round(row.rt, 3), n_carbons=int(row.n),
                              shape_r=round(row.r, 3),
                              n_from_M1=None if pd.isna(row.n_iso) else round(row.n_iso, 1),
                              height12=f"{row.height12:.3g}", image=str(f.relative_to(gal)),
                              verdict="", notes=""))
    pd.DataFrame(sheet).to_csv(gal / "review_sheet.csv", index=False)

    qc = pd.read_csv(out / "decoy_qc.tsv", sep="\t")
    by = truth.groupby(["tier", "feature_class"]).size().unstack(fill_value=0)
    agree = kc.groupby("tier")["agrees"].agg(["sum", "count"]) if len(kc) else None
    lines = [f"# Credentialed truth: {a.dataset}", "",
             f"- 12C files: {', '.join(Path(f).name for f in p['files_12C'])}",
             f"- 13C files: {', '.join(Path(f).name for f in p['files_13C'])}",
             f"- seeds tested: {p['n_seeds']}; truth features: {len(truth)} "
             f"(tier A {int((truth['tier'] == 'A').sum())}, tier B {int((truth['tier'] == 'B').sum())})",
             f"- tolerances: {p['PPM']} ppm, apex +/-{p['apex_tol_s']:.2f} s, shape r >= {p['R_MIN']}, "
             f">= {p['MIN_REPS']}/3 replicates per side", "",
             "## Decoy searches (same pipeline, impossible partners)", "", qc.to_markdown(index=False), "",
             "## Feature classes", "", by.to_markdown(), "",
             "## Known-metabolite carbon check ([M+H]+, 5 ppm)", "",
             (agree.rename(columns={"sum": "n agrees", "count": "matches"}).to_markdown()
              if agree is not None else "no matches"), "",
             "Mismatches can be isomers/other compounds at the same mass; see known_metabolite_check.tsv.", "",
             f"## Gallery: {len(sheet)} images in gallery/ ; fill verdict in gallery/review_sheet.csv"]
    (out / "qc_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
