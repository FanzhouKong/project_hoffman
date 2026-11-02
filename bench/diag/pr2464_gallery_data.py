#!/usr/bin/env python3
"""Data for the PR002464 truth gallery page: the chromatogram of every truth peak in the injections that
define its truth set, on a common time grid, compressed for the browser.

For each study (ST004500 / 01 / 02) and truth set:
  spike   spiked_truth.tsv;  vials 18, 21, 22, 20, 19 (A:B 1000:0 .. 0:1000), 9-12 (1:1 dilution), blanks
  yeast   yeast_truth.tsv;   vials 23 (12C), 25 (12C:13C 1:1), 5-8 (yeast dilution), 24 (13C only), blanks
  plasma  plasma_truth.tsv;  vials 1-4 (plasma dilution), 13-17 (SRM 1950 : 8231 mixing), blanks
Every injection's EIC (5 ppm) is read around the peak's RT plus that injection's offset (files.tsv of the
spike truth) and resampled onto GRID minutes around the peak, so the traces overlay. Intensities are
stored per peak relative to the peak's largest value, as one byte on a 4-decade log scale
(b = 0: nothing; else value = scale * 10^(4 * b / 255 - 4)).
Writes OUT/index_<study>_<set>.json (traces, peaks, test values) and OUT/<study>_<set>_<k>.png: CHUNK peaks,
strongest first, as a lossless greyscale image (one row per peak, traces x grid bytes per row; artifacts do not
serve raw binary files, and PNG compresses the empty stretches).
usage: pr2464_gallery_data.py OUT_DIR [--cpus N]
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench import spiketruth as stt  # noqa: E402
from bench.msdata import load_run  # noqa: E402
from PIL import Image  # noqa: E402

TRUTH = ROOT / "data/truth/PR002464"
GRID = np.round(np.arange(-0.25, 0.2501, 0.01), 4)   # minutes around the peak, 51 points
CHUNK = 128
DECADES = 4.0
SETS = {"spike": [18, 21, 22, 20, 19, 9, 10, 11, 12], "yeast": [23, 25, 5, 6, 7, 8, 24],
        "plasma": [1, 2, 3, 4, 13, 14, 15, 16, 17]}


def offsets(study, files):
    o = pd.read_csv(TRUTH / study / "files.tsv", sep="\t").set_index("file")["rt_offset_s"] / 60
    return files["file"].map(o).fillna(0.0).values


def peaks_table(study, kind, files):
    """one row per truth peak: id, mz, rt (offset-free), tier, flags, label, height, sort key, extra fields"""
    if kind == "spike":
        t = pd.read_csv(TRUTH / study / "spiked_truth.tsv", sep="\t", keep_default_na=False, na_values=[""])
        sd = pd.read_csv(ROOT / "data/raw/PR002464/design/standards.tsv", sep="\t").set_index("std")
        out = pd.DataFrame({"id": t.truth_id, "mz": t.mz_obs.fillna(t.mz), "rt": t.rt_apex, "tier": t.tier,
                            "flags": t["flags"].fillna(""), "label": t.name.str.slice(0, 48) + " · " + t.annotation.fillna(""),
                            "height": t.height_dom,
                            "a": t["std"].map(sd.spikeA_vol), "b": t["std"].map(sd.spikeB_vol),
                            "std": t["std"], "source": t.source, "cls": t.annotation_class,
                            "log2FC": t.log2FC, "obs": t.log2_18_19.round(2), "resid": t.e_fit_resid.round(2),
                            "blank": t.blank_frac.round(3), "rank": t.rt_rank})
    elif kind == "yeast":
        t = pd.read_csv(TRUTH / study / "yeast_truth.tsv", sep="\t")
        f23 = files[files["vial"] == 23].iloc[0]
        out = pd.DataFrame({"id": t.truth_id.str.replace("PR2464_", "", regex=False), "mz": t.mz,
                            "rt": t.rt - f23["off"], "tier": t.tier, "flags": np.where(t.confirmed, "", "unconfirmed"),
                            "label": t.feature_class + " · " + t.n.astype(str) + " C", "height": t.height12,
                            "n": t.n, "mz13": t.mz_13C.round(5), "r": t.r.round(2), "conf": t.confirmed,
                            "n_iso": t.n_iso.round(1), "dil": t.dil_slope_12C.round(2)})
    else:
        t = pd.read_csv(TRUTH / study / "plasma_truth.tsv", sep="\t", keep_default_na=False, na_values=[""])
        lab = t.annotation.where(t.spike_truth.fillna("") == "", t.annotation + " · spike peak")
        out = pd.DataFrame({"id": t.truth_id, "mz": t.mz_obs, "rt": t.rt_apex, "tier": t.tier,
                            "flags": t["flags"].fillna(""), "label": lab, "height": t.height_17,
                            "slope": t.dil_slope.round(2), "r2": t.dil_r2.round(3), "mix": t.mix_resid.round(2),
                            "srm": t.log2_1950_8231.round(2), "in_yeast": t.also_in_yeast, "in_spike": t.also_in_spike})
    out = out[np.isfinite(out.mz) & np.isfinite(out.rt)]
    return out.sort_values("height", ascending=False).reset_index(drop=True)


def w_traces(args):
    path, off, mzs, rts, ppm = args
    run = load_run(path)
    out = np.zeros((len(mzs), len(GRID)), np.float32)
    for i, (m, t) in enumerate(zip(mzs, rts)):
        s0, s1 = run.scan_window(t + off, 0.3)
        if s1 - s0 < 2:
            continue
        e, _ = stt.eic_win(run, m, ppm, s0, s1)
        out[i] = np.interp(GRID, run.rt[s0:s1] - off - t, e, left=0, right=0)
    return out


def encode(X):
    """X: peaks x traces x grid float -> (uint8 same shape, per-peak scale)"""
    scale = X.reshape(len(X), -1).max(axis=1)
    rel = X / np.maximum(scale, 1)[:, None, None]
    with np.errstate(divide="ignore"):
        b = np.round(255 * (np.log10(np.maximum(rel, 10 ** -DECADES)) + DECADES) / DECADES)
    b[rel <= 10 ** -DECADES] = 0
    return b.astype(np.uint8), scale


def build(study, kind, out, pool):
    mode, instrument, _ = stt.STUDIES[study]
    files = stt.study_files(study)
    files["off"] = offsets(study, files)
    pk = peaks_table(study, kind, files)
    tr = files[files["vial"].isin(SETS[kind])].copy()
    tr["vord"] = tr["vial"].map({v: i for i, v in enumerate(SETS[kind])})
    tr = tr.sort_values(["vord", "batch"])
    bl = pd.concat([files[files["sample_type"] == "process.blank"].head(2), files[files["sample_type"] == "solvent.blank"].head(2)])
    tr = pd.concat([tr, bl]).reset_index(drop=True)
    res = pool.map(w_traces, [(p, o, pk.mz.values, pk.rt.values, stt.PPM[instrument]) for p, o in zip(tr.path, tr.off)])
    X = np.stack(res, axis=1)                       # peaks x traces x grid
    B, scale = encode(X)
    nchunk = 0
    for k in range(0, len(pk), CHUNK):
        blk = B[k:k + CHUNK]
        Image.fromarray(blk.reshape(len(blk), -1), mode="L").save(out / f"{study}_{kind}_{k // CHUNK:03d}.png", optimize=True)
        nchunk += 1
    traces = [dict(file=r.file, vial=int(r.vial), batch=int(r.batch), type=r.sample_type,
                   A=float(r.A), B=float(r.B), srm1950=float(r["SRM 1950"]), srm8231=float(r["SRM 8231"]),
                   y12=float(r["Yeast 12C"]), y13=float(r["Yeast 13C"])) for _, r in tr.iterrows()]
    cols = list(pk.columns)
    rows = []
    for vals in pk.itertuples(index=False):
        row = []
        for v in vals:
            if isinstance(v, (float, np.floating)):
                row.append(None if not np.isfinite(v) else round(float(v), 5))
            elif isinstance(v, (np.integer,)):
                row.append(int(v))
            elif isinstance(v, (bool, np.bool_)):
                row.append(bool(v))
            else:
                row.append(v)
        rows.append(row)
    idx = dict(study=study, set=kind, mode=mode, grid=[float(GRID[0]), float(GRID[-1]), len(GRID)], decades=DECADES,
               chunk=CHUNK, n_chunks=nchunk, traces=traces, cols=cols + ["scale"],
               peaks=[r + [round(float(s), 1)] for r, s in zip(rows, scale)])
    (out / f"index_{study}_{kind}.json").write_text(json.dumps(idx, separators=(",", ":")))
    mb = sum((out / f"{study}_{kind}_{i:03d}.png").stat().st_size for i in range(nchunk)) / 1e6
    print(f"[{study} {kind}] {len(pk)} peaks x {len(tr)} traces -> {nchunk} chunks, {mb:.1f} MB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--cpus", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with mp.Pool(a.cpus) as pool:
        for study in ("ST004501", "ST004502", "ST004500"):
            for kind in ("spike", "yeast", "plasma"):
                build(study, kind, out, pool)


if __name__ == "__main__":
    main()
