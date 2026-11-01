#!/usr/bin/env python3
"""Truth tables for the MassCube paper's peak-detection benchmark (Yu et al., Nat Commun 2025, Zenodo 14159704).

Writes data/truth/MASSCUBE/:
  synth_peaks.tsv      the 27,000 inserted signals of simulated_file.mzML (one per m/z, 0.1 Da apart from
                       1550): n_peaks 1/2, noise, resolution, height ratio, the scan span and RT span found in
                       the file, the apex RT of each inserted peak (apex1_rt, and apex2_rt for doubles), and the
                       counts the paper reports for MassCube 1.0, MS-DIAL 4.9, MZmine 3.90, xcms
  eic_labels.tsv       the 722 manually labeled EICs: run, file, m/z, RT (minutes), n_peaks 1/2, taken from the
                       PNG names in Manually_labeled_EICs.zip (the authors' notebook reads them the same way)
  published_features.tsv  m/z + RT of every feature in the authors' own outputs of the eight files
                       (masscube = MassCube 1.0, msdial, mzmine3, xcms), so their numbers can be rescored here
  params.json          scoring rules (copied from the authors' notebooks) and data checks

Scoring follows the paper: a signal is correct when the tool reports exactly n_peaks features at its m/z
(synthetic: |dm/z| < 0.01 Da over the whole run; EICs: |dm/z| < 0.002 Da for single, 0.005 Da for
double peaks, and |dRT| < 0.2 min). Run on a compute node: it reads the synthetic mzML.
usage: masscube_truth.py
"""
import hashlib
import json
import re
import sys
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.msdata import load_run  # noqa: E402
from bench.runs import MC_EXP, MCSI, RUNS  # noqa: E402

SI = MCSI.parent
OUT = ROOT / "data/truth/MASSCUBE"
SYN = MCSI / "Synthetic_data_benchmarking"
EXP = MCSI / "8_experimental_data"
SPAN = 50  # every inserted signal covers 50 consecutive MS1 scans (authors' notebook)

# scoring rules from CODE_synthetic_data_benchmarking.ipynb and CODE_real_data_benchmarking.ipynb
RULES = {
    "synth": {"mz_tol_da": 0.01, "rt": "none (each inserted signal has its own m/z)"},
    "eic_paper": {"mz_tol_da": {"single": 0.002, "double": 0.005}, "rt_tol_min": 0.2},
    "eic_wide": {"mz_tol": "max(paper tolerance, 10 ppm)", "rt_tol_min": 0.2,
                 "why": "label m/z are MassCube's own feature m/z; 0.002 Da is 2-4 ppm above m/z 500"},
    "correct": "count of reported features in the window == n_peaks",
    "summary": "paper average = mean over the 6 noise levels (synthetic) of per-level accuracy; "
               "pooled = correct / n",
}


def reindex_mzml(src, dst):
    """The authors inserted the synthetic peaks by rewriting spectra in place but kept the source file's
    <indexList>, so 9,264 of 9,283 byte offsets (and indexListOffset, fileChecksum) are stale and readers
    that trust the index read the wrong bytes. Write a copy with the index recomputed; spectra unchanged."""
    b = Path(src).read_bytes()
    cut = b.index(b"<indexList ")
    body, tail = b[:cut], b[cut:]
    pos = {m.group(1): m.start() for m in re.finditer(rb'<(?:spectrum|chromatogram) [^>]*?\bid="([^"]*)"', body)}
    n_ref = len(re.findall(rb'<offset idRef="', tail))
    tail = re.sub(rb'(<offset idRef="([^"]*)">)\d+(</offset>)',
                  lambda m: m.group(1) + str(pos[m.group(2)]).encode() + m.group(3), tail)
    tail = re.sub(rb"<indexListOffset>\d+</indexListOffset>", b"<indexListOffset>%d</indexListOffset>" % cut, tail)
    out = body + tail
    k = out.index(b"<fileChecksum>") + len(b"<fileChecksum>")
    out = out[:k] + re.sub(rb"^[0-9a-f]{40}", hashlib.sha1(out[:k]).hexdigest().encode(), out[k:])
    assert len(pos) == n_ref, (len(pos), n_ref)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_bytes(out)


def check_crc(zip_name, members):
    """extracted files must match the archive (an earlier unzip hung on Quobyte)"""
    with zipfile.ZipFile(SI / zip_name) as z:
        for m in members:
            crc = 0
            with open(MCSI / m, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 24), b""):
                    crc = zlib.crc32(chunk, crc)
            if crc != z.getinfo(m).CRC:
                raise SystemExit(f"CRC mismatch: {m}")


def synth():
    pub = {"masscube_reported_number": "MassCube 1.0", "msdial_reported_number": "MS-DIAL 4.9",
           "mzmine_reported_number": "MZmine 3.90", "xcms_reported_number": "xcms 4.0"}
    t = pd.concat([pd.read_csv(SYN / "inserted_single_peaks.csv").assign(n_peaks=1),
                   pd.read_csv(SYN / "Inserted_shoulder_peaks.csv").assign(n_peaks=2)], ignore_index=True)
    t = t.rename(columns={"m/z": "mz", "start_scan_index_of_inserted_signal": "start_scan",
                          "noise_level": "noise"} | {k: f"pub_{v}" for k, v in pub.items()})
    run = load_run(RUNS["MC_SYNTH"]["files"][0])
    # locate each signal in the file: scans with a centroid within 0.005 Da of its m/z
    first, last, n_pts = [], [], []
    for m in t["mz"].values:
        a, b = np.searchsorted(run.mz, [m - 0.005, m + 0.005])
        s = run.scan[a:b]
        first.append(s.min() if len(s) else -1); last.append(s.max() if len(s) else -1); n_pts.append(len(s))
    t["scan_first"], t["scan_last"], t["n_points"] = first, last, n_pts
    t["rt_start"] = run.rt[np.clip(t["start_scan"], 0, run.n_scans - 1)]
    t["rt_end"] = run.rt[np.clip(t["start_scan"] + SPAN - 1, 0, run.n_scans - 1)]
    # apex RTs as generated (CODE_synthetic_data_benchmarking.ipynb: 50 points on x = linspace(0, 4, 50), width 1,
    # apex at x = 2, or 2 -/+ resolution / 2 for the two peaks of a double, the earlier one the larger), placed on the
    # file's own MS1 scan times: the spacing is uneven (0.4-3.2 s), so a straight line from rt_start to rt_end is
    # off by up to 22 s
    def rt_at(x):
        return np.interp(t["start_scan"].values + x * (SPAN - 1) / 4, np.arange(run.n_scans), run.rt)
    dbl, res = t["n_peaks"].values == 2, t["resolution"].values
    t["apex1_rt"] = np.where(dbl, rt_at(2 - res / 2), rt_at(np.full(len(t), 2.0)))
    t["apex2_rt"] = np.where(dbl, rt_at(2 + res / 2), np.nan)
    # background check: real centroids in the inserted m/z range that belong to no inserted signal
    lo, hi = t["mz"].min() - 0.05, t["mz"].max() + 0.05
    a, b = np.searchsorted(run.mz, [lo, hi])
    ins, x = np.sort(t["mz"].values), run.mz[a:b]
    j = np.clip(np.searchsorted(ins, x), 1, len(ins) - 1)
    near = np.minimum(np.abs(x - ins[j - 1]), np.abs(x - ins[j])) < 0.005
    checks = {
        "n_signals": len(t), "n_single": int((t.n_peaks == 1).sum()), "n_double": int((t.n_peaks == 2).sum()),
        "mz_range": [float(t.mz.min()), float(t.mz.max())],
        "min_mz_gap_da": float(np.diff(np.sort(t.mz.values)).min()),
        "signals_found_at_start_scan": int((t.scan_first == t.start_scan).sum()),
        "signals_within_their_50_scans": int(((t.scan_first >= t.start_scan) &
                                               (t.scan_last <= t.start_scan + SPAN - 1)).sum()),
        "background_centroids_in_inserted_range": int((~near).sum()),
        "max_mz_outside_inserted": float(run.mz[run.mz < lo].max()),
        "n_ms1_scans": int(run.n_scans), "scan_time_s": float(np.median(np.diff(run.rt)) * 60),
    }
    return t, checks


def eic_labels():
    rows = []
    with zipfile.ZipFile(SI / "Manually_labeled_EICs.zip") as z:
        for n in z.namelist():
            m = re.match(r"EIC_manually_labeleled/(single|double)_peaks/(.+)_mz_(.+)_rt_(.+)\.png$", n)
            if m:
                rows.append(dict(run=f"MC_{m.group(2)}", file=m.group(2), mz=float(m.group(3)),
                                 rt=float(m.group(4)), n_peaks=1 if m.group(1) == "single" else 2,
                                 png=n.split("/", 1)[1]))
    return pd.DataFrame(rows).sort_values(["file", "n_peaks", "mz"]).reset_index(drop=True)


def published():
    """authors' feature tables for the eight files; msdial/xcms name the NIST plasma files 'NIST_Serum'"""
    spec = {"masscube": ("masscube", ".txt", "\t", "m/z", "RT", 1.0),
            "msdial": ("msdial", ".txt", "\t", "Precursor m/z", "RT (min)", 1.0),
            "mzmine3": ("mzmine3", ".csv", ",", "mz", "rt", 1.0),
            "xcms": ("xcms", ".csv", ",", "mz", "rt", 1 / 60)}
    out = []
    for tool, (d, ext, sep, cmz, crt, f) in spec.items():
        for s in MC_EXP:
            p = EXP / d / f"{s}{ext}"
            if not p.exists():
                p = EXP / d / f"{s.replace('NIST_Plasma', 'NIST_Serum')}{ext}"
            t = pd.read_csv(p, sep=sep, low_memory=False)
            out.append(pd.DataFrame({"tool": tool, "file": s, "mz": t[cmz].astype(float),
                                     "rt": t[crt].astype(float) * f}))
    return pd.concat(out, ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    check_crc("Synthetic_data_benchmarking.zip", ["Synthetic_data_benchmarking/simulated_file.mzML"])
    check_crc("Experimental_data_benchmarking.zip", [f"8_experimental_data/raw_data/{s}.mzML" for s in MC_EXP])
    reindex_mzml(SYN / "simulated_file.mzML", RUNS["MC_SYNTH"]["files"][0])
    t, checks = synth()
    t.to_csv(OUT / "synth_peaks.tsv", sep="\t", index=False)
    lab = eic_labels()
    lab.to_csv(OUT / "eic_labels.tsv", sep="\t", index=False)
    pub = published()
    pub.to_csv(OUT / "published_features.tsv", sep="\t", index=False)
    # where do the label coordinates come from? nearest MassCube 1.0 feature (published table)
    mc = pub[pub.tool == "masscube"]
    d_mz, d_rt = [], []
    for r in lab.itertuples():
        f = mc[mc.file == r.file]
        i = np.argmin(np.abs(f.mz.values - r.mz) / 0.001 + np.abs(f.rt.values - r.rt) / 0.01)
        d_mz.append(abs(f.mz.values[i] - r.mz)); d_rt.append(abs(f.rt.values[i] - r.rt))
    d_mz, d_rt = np.array(d_mz), np.array(d_rt)
    checks_eic = {
        "n_labels": len(lab), "n_single": int((lab.n_peaks == 1).sum()), "n_double": int((lab.n_peaks == 2).sum()),
        "per_file": lab.groupby("file").n_peaks.value_counts().unstack(fill_value=0).to_dict("index"),
        "labels_equal_to_a_masscube1_feature (|dmz| <= 0.00051, |drt| <= 0.0051)":
            int(((d_mz <= 0.00051) & (d_rt <= 0.0051)).sum()),
    }
    params = {"source": "Zenodo 14159704 (Yu et al., Nat Commun 16:5487, 2025)", "rules": RULES,
              "checks_synth": checks, "checks_eic": checks_eic}
    (OUT / "params.json").write_text(json.dumps(params, indent=2, default=int) + "\n")
    print(json.dumps(params, indent=2, default=int))


if __name__ == "__main__":
    main()
