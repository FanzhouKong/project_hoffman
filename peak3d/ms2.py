"""MS2 spectra of the features.

The spectra are collected while io.load_cloud reads a file's MS1 scans (no second pass over the mzML) and stored
per file in <output>/ms2/<stem>.npz. map_to_groups assigns them to the aligned features, the same way for every
kernel (peak3d, asari, MassCube): a spectrum belongs to the feature whose m/z is within max(ppm, da_floor) of its
precursor and whose RT window, widened by max(2 s, a quarter of its width) on each side, holds the spectrum's time
on the aligned axis (the file's own RT correction applied when there is one). When several features qualify, the
one strongest in that file wins, then the one with the closest apex. Each feature keeps all its spectra in
ms2_map.tsv and two spectra for the search: the consensus of all of them, from every file (consensus.build:
precursor-weighted, a peak needs two spectra), in ms2_consensus.msp, and the single spectrum with the highest total
ion current (MassCube's choice) in ms2_spectra.msp."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

MAP_COLUMNS = ["group_id", "file", "scan", "rt_s", "rt_aligned_s", "precursor_mz", "precursor_intensity",
               "precursor_purity", "tic", "n_peaks", "representative", "in_consensus", "consensus_status"]
SPECTRA = {"consensus": "ms2_consensus.msp", "single": "ms2_spectra.msp"}      # the two copies of each feature


class Collector(list):
    """io.load_cloud's MS2 collector: a list of spectra plus the count of empty or profile-mode ones it skipped"""
    n_skipped = 0


@dataclass
class Ms2Set:
    scan: np.ndarray       # str: the mzML spectrum id
    rt: np.ndarray         # f8, minutes, native
    prec_mz: np.ndarray    # f8, nan when the file gives none
    tic: np.ndarray        # f8
    off: np.ndarray        # int64, len n + 1
    mz: np.ndarray         # f8, ascending within a spectrum
    inten: np.ndarray      # f4
    polarity: str = ""
    prec_int: np.ndarray | None = None   # f8, the precursor's intensity in the MS1 scan before (0: not there)
    purity: np.ndarray | None = None     # f8, its share of the isolation window (nan: window unknown)

    def __post_init__(self):
        n = len(self.rt)
        if self.prec_int is None:          # spectra stored before these were recorded
            self.prec_int = np.full(n, np.nan)
        if self.purity is None:
            self.purity = np.full(n, np.nan)

    def __len__(self):
        return len(self.rt)


def from_collected(rows, polarity: str = "") -> Ms2Set:
    """Ms2Set of what load_cloud collected (m/z sorted within each spectrum, non-positive peaks dropped)"""
    mzs, its, counts = [], [], []
    for *_, m, i in rows:
        ok = np.isfinite(m) & np.isfinite(i) & (i > 0)
        m, i = m[ok], i[ok]
        if len(m) > 1 and np.any(np.diff(m) < 0):
            o = np.argsort(m, kind="stable")
            m, i = m[o], i[o]
        mzs.append(m)
        its.append(i)
        counts.append(len(m))
    off = np.zeros(len(rows) + 1, np.int64)
    np.cumsum(counts, out=off[1:])
    pols = {r[3] for r in rows} - {""}
    return Ms2Set(np.array([r[0] for r in rows], dtype=str), np.array([r[1] for r in rows], np.float64),
                  np.array([r[2] for r in rows], np.float64),
                  np.array([float(i.sum()) for i in its], np.float64), off,
                  np.concatenate(mzs) if mzs else np.zeros(0), np.concatenate(its) if its else np.zeros(0, np.float32),
                  polarity or (next(iter(pols)) if len(pols) == 1 else ""),
                  np.array([r[4] for r in rows], np.float64), np.array([r[5] for r in rows], np.float64))


def save(ms: Ms2Set, path) -> None:
    np.savez(path, scan=ms.scan, rt=ms.rt, prec_mz=ms.prec_mz, tic=ms.tic, off=ms.off, mz=ms.mz, inten=ms.inten,
             polarity=np.array(ms.polarity), prec_int=ms.prec_int, purity=ms.purity)


def load(path) -> Ms2Set:
    with np.load(path, allow_pickle=False) as z:
        extra = [z[k] if k in z.files else None for k in ("prec_int", "purity")]
        return Ms2Set(z["scan"], z["rt"], z["prec_mz"], z["tic"], z["off"], z["mz"], z["inten"], str(z["polarity"]),
                      *extra)


# ------------------------------------------------------------------ mapping ----
@njit(cache=True)
def _assign(g_mz_sorted, order, g_lo, g_hi, g_rt, g_val, prec, t, ppm, da_floor):
    """per spectrum: the group index it belongs to (-1 if none); see the module docstring"""
    out = np.full(len(prec), -1, np.int64)
    for i in range(len(prec)):
        p = prec[i]
        if not np.isfinite(p) or not np.isfinite(t[i]):
            continue
        tol = max(p * ppm * 1e-6, da_floor)
        a = np.searchsorted(g_mz_sorted, p - tol)
        b = np.searchsorted(g_mz_sorted, p + tol, side="right")
        best = -1
        bv = -1.0
        bd = np.inf
        for k in range(a, b):
            g = order[k]
            if t[i] < g_lo[g] or t[i] > g_hi[g]:
                continue
            v = g_val[g]
            d = abs(t[i] - g_rt[g])
            if v > bv or (v == bv and d < bd):
                best = g
                bv = v
                bd = d
        out[i] = best
    return out


def _aligned_time(output: Path, stem: str, rt):
    """MS2 times (min) on the aligned axis: the file's RT correction interpolated at them, else unchanged"""
    w = output / "rt_correction" / f"{stem}.tsv"
    if not w.exists():
        return rt
    t = pd.read_csv(w, sep="\t")
    if not len(t):
        return rt
    return np.interp(rt, t["rt_native"].to_numpy(float), t["rt_corr"].to_numpy(float))


def _stale(output: Path):
    for name in ("ms2_map.tsv", *SPECTRA.values()):
        (output / name).unlink(missing_ok=True)


def map_to_groups(output, ppm: float = 10.0, da_floor: float = 0.003, fragment_tol: float = 0.02) -> pd.DataFrame:
    """assign every stored MS2 spectrum to an aligned feature; writes ms2_map.tsv (all assigned spectra),
    ms2_spectra.msp (each feature's highest-TIC spectrum) and ms2_consensus.msp (each feature's consensus spectrum).
    Returns the map (empty when the files had no MS2)."""
    from .consensus import build
    from .library import write_msp_record
    from .similarity import METHOD_CODE, clean, score
    from .pipeline import log
    output = Path(output)
    table = pd.read_csv(output / "aligned_feature_table.tsv", sep="\t")
    ms2_dir = output / "ms2"
    stems = [c for c in table.columns if (ms2_dir / f"{c}.npz").exists()]
    if not stems or not len(table):
        _stale(output)
        return pd.DataFrame(columns=MAP_COLUMNS)
    g_id = table["group_id"].to_numpy(np.int64)
    g_mz = table["mz"].to_numpy(np.float64)
    g_rt = table["rt"].to_numpy(np.float64)
    lo, hi = table["rt_min"].to_numpy(np.float64), table["rt_max"].to_numpy(np.float64)
    pad = np.maximum(2.0 / 60.0, 0.25 * (hi - lo))
    g_lo, g_hi = lo - pad, hi + pad
    order = np.argsort(g_mz, kind="stable")
    g_sorted = g_mz[order]
    parts, sets = [], {}
    for s in stems:
        ms = load(ms2_dir / f"{s}.npz")
        if not len(ms):
            continue
        sets[s] = ms
        val = np.nan_to_num(table[s].to_numpy(np.float64), nan=0.0)
        t_al = _aligned_time(output, s, ms.rt)
        g = _assign(g_sorted, order, g_lo, g_hi, g_rt, val, ms.prec_mz, t_al, float(ppm), float(da_floor))
        k = np.flatnonzero(g >= 0)
        parts.append(pd.DataFrame({"g": g[k], "group_id": g_id[g[k]], "file": s, "k": k, "scan": ms.scan[k],
                                   "rt_s": np.round(ms.rt[k] * 60.0, 3), "rt_aligned_s": np.round(t_al[k] * 60.0, 3),
                                   "precursor_mz": ms.prec_mz[k], "precursor_intensity": ms.prec_int[k],
                                   "precursor_purity": ms.purity[k], "tic": ms.tic[k],
                                   "n_peaks": np.diff(ms.off)[k]}))
    n_all = sum(len(m) for m in sets.values())
    m = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["g", "k", *MAP_COLUMNS])
    if not len(m):
        _stale(output)
        log(f"ms2: 0 of {n_all} spectra mapped to features")
        return pd.DataFrame(columns=MAP_COLUMNS)
    m = m.sort_values(["g", "tic"], ascending=[True, False], kind="stable").reset_index(drop=True)
    m["representative"] = (~m["g"].duplicated()).astype(int)
    n_per = m.groupby("g").size()
    in_cons = np.zeros(len(m), np.int8)
    status = np.full(len(m), "", dtype=object)
    ion = {"pos": "Positive", "neg": "Negative"}
    with open(output / "ms2_consensus.msp", "w", encoding="utf-8") as fh:
        for gi, d in m.groupby("g", sort=True):
            rows = d.index.to_numpy()                       # positions: m has a fresh RangeIndex
            spectra = [(sets[f].mz[sets[f].off[k]:sets[f].off[k + 1]], sets[f].inten[sets[f].off[k]:sets[f].off[k + 1]])
                       for f, k in zip(d["file"], d["k"])]
            c = build(spectra, d["precursor_intensity"].to_numpy(float), d["precursor_purity"].to_numpy(float),
                      float(g_mz[gi]), fragment_tol)
            if c is None:
                status[rows] = "empty"
                continue
            status[rows] = c.status
            in_cons[rows[c.used]] = 1
            t = d.iloc[c.top]
            # how alike the consensus and the highest-TIC spectrum are (weighted entropy, cleaned as for the search)
            r0 = d.iloc[0]                                  # m is sorted by TIC within a group: the highest first
            sm, si = clean(*spectra[0], float(g_mz[gi]), float(fragment_tol))
            vs = score(c.mz, c.inten, sm, si, METHOD_CODE["weighted_entropy"], float(fragment_tol))[0] if len(sm) else 0.0
            write_msp_record(fh, [
                ("NAME", f"feature_{int(t['group_id'])}"), ("PRECURSORMZ", f"{g_mz[gi]:.6f}"),
                ("RETENTIONTIME", f"{g_rt[gi]:.4f}"), ("IONMODE", ion.get(sets[t['file']].polarity, "")),
                ("Comment", f"group_id={int(t['group_id'])} n_ms2={len(d)} n_merged={len(c.used)} "
                            f"n_files={d['file'].iloc[c.used].nunique()} file=\"{t['file']}\" scan=\"{t['scan']}\" "
                            f"rt_s={t['rt_s']:.3f} precursor={t['precursor_mz']:.6f} n_dropped={c.n_dropped} "
                            f"vs_single={vs:.4f}")],
                c.mz, np.round(c.inten / c.inten.max() * 1000.0, 3))
    m["in_consensus"] = in_cons
    m["consensus_status"] = status
    m[MAP_COLUMNS].to_csv(output / "ms2_map.tsv", sep="\t", index=False, float_format="%.6g")
    with open(output / "ms2_spectra.msp", "w", encoding="utf-8") as fh:
        for r in m[m["representative"] == 1].itertuples(index=False):
            ms = sets[r.file]
            a, b = ms.off[r.k], ms.off[r.k + 1]
            write_msp_record(fh, [
                ("NAME", f"feature_{int(r.group_id)}"), ("PRECURSORMZ", f"{g_mz[r.g]:.6f}"),
                ("RETENTIONTIME", f"{g_rt[r.g]:.4f}"),
                ("IONMODE", {"pos": "Positive", "neg": "Negative"}.get(ms.polarity, "")),
                ("Comment", f'group_id={int(r.group_id)} file="{r.file}" scan="{r.scan}" rt_s={r.rt_s:.3f} '
                            f"precursor={r.precursor_mz:.6f} tic={r.tic:.6g} n_ms2={int(n_per[r.g])}")],
                ms.mz[a:b], ms.inten[a:b])
    log(f"ms2: {len(m)} of {n_all} spectra mapped to {m['g'].nunique()} features")
    return m[MAP_COLUMNS]


# key=value pairs of our Comment lines; file and scan are quoted (Thermo spectrum ids hold spaces and "=")
_COMMENT = re.compile(r'(\w+)=(?:"([^"]*)"|(\S+))')


def _comment(text: str) -> dict:
    return {k: a if a or not b else b for k, a, b in _COMMENT.findall(text)}


def read_representatives(output, which: str = "single") -> dict | None:
    """one copy of every feature's MS2 spectrum (which: "single" = ms2_spectra.msp, the highest-TIC spectrum;
    "consensus" = ms2_consensus.msp), as arrays in file order: group_id, n_ms2, n_merged (spectra in the consensus,
    1 for single), vs_single (consensus only: its weighted entropy similarity to the highest-TIC spectrum; nan
    otherwise), prec_mz (the feature's m/z), rt_s (the native time of that, or the consensus's most abundant,
    spectrum), file, scan, ion_mode, off, mz, inten. None when the output has no such file or it is empty."""
    from .library import read_msp
    path = Path(output) / SPECTRA[which]
    if not path.exists():
        return None
    lib = read_msp(path)
    if not len(lib):
        return None
    info = [_comment(c) for c in lib.comments]
    return {"group_id": np.array([int(d.get("group_id", -1)) for d in info], np.int64),
            "n_ms2": np.array([int(d.get("n_ms2", 1)) for d in info], np.int32),
            "n_merged": np.array([int(d.get("n_merged", 1)) for d in info], np.int32),
            "n_files": np.array([int(d.get("n_files", 1)) for d in info], np.int32),
            "vs_single": np.array([float(d.get("vs_single", "nan")) for d in info], np.float64),
            "prec_mz": lib.precursor_mz, "rt_s": np.array([float(d.get("rt_s", "nan")) for d in info], np.float64),
            "file": [d.get("file", "") for d in info], "scan": [d.get("scan", "") for d in info],
            "ion_mode": list(lib.ion_mode), "off": lib.off, "mz": lib.mz, "inten": lib.inten}
