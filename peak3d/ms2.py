"""MS2 spectra of the features.

The spectra are collected while io.load_cloud reads a file's MS1 scans (no second pass over the mzML) and stored
per file in <output>/ms2/<stem>.npz. map_to_groups assigns them to the aligned features, the same way for every
kernel (peak3d, asari, MassCube): a spectrum belongs to the feature whose m/z is within max(ppm, da_floor) of its
precursor (da_floor 0.004 Da = 10 ppm at m/z 400: below it, QTOF precursors scatter by more than their ppm)
and whose RT window, widened by max(2 s, a quarter of its width) on each side, holds the spectrum's time
on the aligned axis (the file's own RT correction applied when there is one). The precursor is first put on the
table's m/z scale: corrected by the file's measured precursor error (io.precursor_calibration: some instruments
write it tens of ppm off their MS1) and by the file's m/z offset from alignment, and ppm is widened to the file's
measured tolerance when that is larger. When several features qualify, the
one strongest in that file wins, then the one with the closest apex. Each feature keeps all its spectra in
ms2_map.tsv and two spectra for the search: the consensus of all of them, from every file (consensus.build:
precursor-weighted, a peak needs two spectra), in ms2_consensus.msp, and the single spectrum with the highest total
ion current (MassCube's choice) in ms2_spectra.msp. The identity search denoises them against each candidate's
formula (library.identify, the denoising search). On request (denoise=True, the search's denoise="t0") both are
also written denoised once per spectrum (denoise_copies: specdenoise's neutral-loss rule, tier T0 top-5, judged from
the feature m/z alone) to ms2_consensus_denoised.msp and ms2_spectra_denoised.msp."""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from .io import precursor_error

MAP_COLUMNS = ["group_id", "file", "scan", "rt_s", "rt_aligned_s", "precursor_mz", "precursor_intensity",
               "precursor_purity", "tic", "n_peaks", "representative", "in_consensus", "consensus_status"]
SPECTRA = {"consensus": "ms2_consensus.msp", "single": "ms2_spectra.msp",      # the two copies of each feature,
           "consensus_denoised": "ms2_consensus_denoised.msp",                  # and each with the noise removed
           "single_denoised": "ms2_spectra_denoised.msp"}
RAW = ("consensus", "single")
DENOISED = {c: f"{c}_denoised" for c in RAW}
DENOISE = dict(tier="t0", top_k=5)       # specdenoise.Params: precursor m/z only; losses of the 5 best candidate formulas
DENOISE_LABEL = "T0 top-5"
LABEL = {"consensus": "consensus spectrum", "single": "highest-TIC spectrum",
         "consensus_denoised": "denoised consensus spectrum", "single_denoised": "denoised highest-TIC spectrum"}
POOL_MIN = 200                           # fewer spectra than this are denoised in-process (a pool costs seconds to start)


class Collector(list):
    """io.load_cloud's MS2 collector: a list of spectra plus the count of empty or profile-mode ones it skipped and
    the file's precursor calibration (io.precursor_calibration)"""
    n_skipped = 0
    cal_mz = cal_ppm = np.zeros(0)
    tol_ppm = 0.0
    cal_n = 0


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
    cal_mz: np.ndarray | None = None     # the file's precursor error curve (io.precursor_calibration): knot m/z,
    cal_ppm: np.ndarray | None = None    # error there (ppm, precursor minus MS1); empty: not measured
    tol_ppm: float = 0.0                 # its tolerance (0: not measured)
    cal_n: int = 0                       # spectra it was measured on

    def __post_init__(self):
        n = len(self.rt)
        if self.prec_int is None:          # spectra stored before these were recorded
            self.prec_int = np.full(n, np.nan)
        if self.purity is None:
            self.purity = np.full(n, np.nan)
        if self.cal_mz is None or self.cal_ppm is None:
            self.cal_mz, self.cal_ppm = np.zeros(0), np.zeros(0)

    def precursor_on_ms1(self) -> np.ndarray:
        """the precursor m/z on the file's MS1 m/z scale (its calibration curve applied)"""
        return self.prec_mz * (1.0 - precursor_error(self.cal_mz, self.cal_ppm, self.prec_mz) * 1e-6)

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
                  np.array([r[4] for r in rows], np.float64), np.array([r[5] for r in rows], np.float64),
                  np.asarray(getattr(rows, "cal_mz", np.zeros(0)), np.float64),
                  np.asarray(getattr(rows, "cal_ppm", np.zeros(0)), np.float64),
                  float(getattr(rows, "tol_ppm", 0.0)), int(getattr(rows, "cal_n", 0)))


def save(ms: Ms2Set, path) -> None:
    np.savez(path, scan=ms.scan, rt=ms.rt, prec_mz=ms.prec_mz, tic=ms.tic, off=ms.off, mz=ms.mz, inten=ms.inten,
             polarity=np.array(ms.polarity), prec_int=ms.prec_int, purity=ms.purity, cal_mz=ms.cal_mz,
             cal_ppm=ms.cal_ppm, tol_ppm=np.array(ms.tol_ppm), cal_n=np.array(ms.cal_n))


def load(path) -> Ms2Set:
    with np.load(path, allow_pickle=False) as z:
        extra = [z[k] if k in z.files else None for k in ("prec_int", "purity", "cal_mz", "cal_ppm")]
        return Ms2Set(z["scan"], z["rt"], z["prec_mz"], z["tic"], z["off"], z["mz"], z["inten"], str(z["polarity"]),
                      *extra, float(z["tol_ppm"]) if "tol_ppm" in z.files else 0.0,
                      int(z["cal_n"]) if "cal_n" in z.files else 0)


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


def _mz_offset_curve(output: Path, stem: str, rt, default: float):
    """the file's m/z offset (ppm) at native times rt (min): its per-scan curve in rt_correction/<stem>.tsv
    (mz_offset_ppm, written since the dynamic offset), else default (its one number)"""
    p = output / "rt_correction" / f"{stem}.tsv"
    rt = np.asarray(rt, np.float64)
    if p.exists():
        t = pd.read_csv(p, sep="\t")
        if "mz_offset_ppm" in t and len(t):
            return np.interp(rt, t["rt_native"].to_numpy(float), t["mz_offset_ppm"].to_numpy(float))
    return np.full(rt.shape, default)


def _mz_offsets(output: Path) -> dict:
    """file -> its m/z offset from alignment (ppm, file minus consensus; the table's m/z is the file's times
    1 - offset), from rt_correction/summary.tsv; empty when the run was not aligned"""
    p = output / "rt_correction" / "summary.tsv"
    if not p.exists():
        return {}
    t = pd.read_csv(p, sep="\t", dtype={"file": str})
    if "mz_offset_ppm" not in t:
        return {}
    return {f: float(v) for f, v in zip(t["file"], t["mz_offset_ppm"]) if np.isfinite(v)}


def _stale(output: Path):
    for name in ("ms2_map.tsv", *SPECTRA.values()):
        (output / name).unlink(missing_ok=True)


def map_to_groups(output, ppm: float = 10.0, da_floor: float = 0.004, fragment_tol: float = 0.02, workers: int = 1,
                  denoise: bool = False) -> pd.DataFrame:
    """assign every stored MS2 spectrum to an aligned feature; writes ms2_map.tsv (all assigned spectra),
    ms2_spectra.msp (each feature's highest-TIC spectrum) and ms2_consensus.msp (each feature's consensus spectrum),
    and, with denoise, their denoised copies (denoise_copies, on workers processes). Returns the map (empty when the
    files had no MS2)."""
    from .consensus import build
    from .library import write_msp_record, write_tsv
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
    parts, sets, qc = [], {}, []
    mz_off = _mz_offsets(output)
    for s in stems:
        ms = load(ms2_dir / f"{s}.npz")
        if not len(ms):
            continue
        sets[s] = ms
        val = np.nan_to_num(table[s].to_numpy(np.float64), nan=0.0)
        t_al = _aligned_time(output, s, ms.rt)
        moff = _mz_offset_curve(output, s, ms.rt, mz_off.get(s, 0.0))
        prec = ms.precursor_on_ms1() * (1.0 - moff * 1e-6)
        tol = max(float(ppm), ms.tol_ppm)
        g = _assign(g_sorted, order, g_lo, g_hi, g_rt, val, prec, t_al, tol, float(da_floor))
        k = np.flatnonzero(g >= 0)
        err = ms.cal_ppm if len(ms.cal_ppm) else np.zeros(1)
        qc.append(dict(file=s, n_ms2=len(ms), n_measured=ms.cal_n, error_ppm_min=err.min(),
                       error_ppm_median=float(np.median(precursor_error(ms.cal_mz, ms.cal_ppm, ms.prec_mz))),
                       error_ppm_max=err.max(), tol_ppm=tol, mz_offset_ppm=mz_off.get(s, 0.0),
                       precursor_in_ms1=int(np.sum(ms.prec_int > 0)), mapped=len(k)))
        parts.append(pd.DataFrame({"g": g[k], "group_id": g_id[g[k]], "file": s, "k": k, "scan": ms.scan[k],
                                   "rt_s": np.round(ms.rt[k] * 60.0, 3), "rt_aligned_s": np.round(t_al[k] * 60.0, 3),
                                   "precursor_mz": ms.prec_mz[k], "precursor_intensity": ms.prec_int[k],
                                   "precursor_purity": ms.purity[k], "tic": ms.tic[k],
                                   "n_peaks": np.diff(ms.off)[k], "mz_off": moff[k]}))
    n_all = sum(len(m) for m in sets.values())
    if qc:
        (output / "qc").mkdir(exist_ok=True)
        q = pd.DataFrame(qc)
        q.to_csv(output / "qc" / "ms2_precursor.tsv", sep="\t", index=False, float_format="%.4g")
        log(f"ms2: precursor m/z error against MS1 median {q['error_ppm_median'].median():+.1f} ppm "
            f"(files {q['error_ppm_median'].min():+.1f} to {q['error_ppm_median'].max():+.1f}), "
            f"tolerance {q['tol_ppm'].min():.0f}-{q['tol_ppm'].max():.0f} ppm; precursor found in the MS1 scan before "
            f"for {q['precursor_in_ms1'].sum()} of {q['n_ms2'].sum()} spectra")
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
                            f"vs_single={vs:.4f} mz_offset_ppm={np.median(d['mz_off'].to_numpy()[c.used]):.3f}")],
                c.mz, np.round(c.inten / c.inten.max() * 1000.0, 3))
    m["in_consensus"] = in_cons
    m["consensus_status"] = status
    write_tsv(m[MAP_COLUMNS], output / "ms2_map.tsv")
    with open(output / "ms2_spectra.msp", "w", encoding="utf-8") as fh:
        for r in m[m["representative"] == 1].itertuples(index=False):
            ms = sets[r.file]
            a, b = ms.off[r.k], ms.off[r.k + 1]
            write_msp_record(fh, [
                ("NAME", f"feature_{int(r.group_id)}"), ("PRECURSORMZ", f"{g_mz[r.g]:.6f}"),
                ("RETENTIONTIME", f"{g_rt[r.g]:.4f}"),
                ("IONMODE", {"pos": "Positive", "neg": "Negative"}.get(ms.polarity, "")),
                ("Comment", f'group_id={int(r.group_id)} file="{r.file}" scan="{r.scan}" rt_s={r.rt_s:.3f} '
                            f"precursor={r.precursor_mz:.6f} tic={r.tic:.6g} n_ms2={int(n_per[r.g])} "
                            f"mz_offset_ppm={r.mz_off:.3f}")],
                ms.mz[a:b], ms.inten[a:b])
    log(f"ms2: {len(m)} of {n_all} spectra mapped to {m['g'].nunique()} features")
    for c in RAW:                   # never an earlier run's denoised copies next to these
        (output / SPECTRA[DENOISED[c]]).unlink(missing_ok=True)
    if denoise:
        try:
            denoise_copies(output, workers)
        except Exception as e:      # the raw copies stand on their own
            for c in RAW:
                (output / SPECTRA[DENOISED[c]]).unlink(missing_ok=True)
            log(f"ms2 denoising skipped: {e!r}")
    return m[MAP_COLUMNS]


# ------------------------------------------------------------------ denoising ----
def _msp_blocks(path) -> list[tuple[list[str], list[str]]]:
    """the records of an MSP file we wrote (library.write_msp_record): (field lines, peak lines) each"""
    out = []
    for block in Path(path).read_text(encoding="utf-8").split("\n\n"):
        lines = [x for x in block.split("\n") if x.strip()]
        k = next((i for i, x in enumerate(lines) if x.lower().startswith("num peaks:")), None)
        if k is not None:
            out.append((lines[:k], lines[k + 1:]))
    return out


def _field(head: list[str], key: str) -> str:
    key = key.lower() + ":"
    return next((x.split(":", 1)[1].strip() for x in head if x.lower().startswith(key)), "")


def _denoise_many(items) -> tuple[list, int]:
    """keep masks of [(mz, intensity, precursor m/z, charge), ...] by specdenoise (DENOISE), and how many spectra it
    could not judge (an error leaves the spectrum whole)"""
    from specdenoise import Params, denoise
    p = Params(**DENOISE)
    out, failed = [], 0
    for mz, it, pmz, z in items:
        keep = np.ones(len(mz), bool)
        if len(mz) and np.isfinite(pmz) and pmz > 0:
            try:
                keep = np.asarray(denoise(mz, it, pmz, charge=z, params=p).keep, bool)
            except Exception:
                failed += 1
        out.append(keep)
    return out, failed


def _denoise_chunk(job):
    idx, items = job
    masks, failed = _denoise_many(items)
    return idx, masks, failed


def _keep_masks(items, workers: int) -> tuple[list, int]:
    """_denoise_many over items, on up to workers processes: the heaviest precursors (the slowest to judge) spread
    across the chunks. The kernels and the loss table are made ready here first, so the workers only load them."""
    if not items:
        return [], 0
    first, f0 = _denoise_many(items[:1])
    n = len(items)
    if workers <= 1 or n < POOL_MIN:
        rest, f1 = _denoise_many(items[1:])
        return first + rest, f0 + f1
    from multiprocessing import get_context
    order = np.argsort([-(x[2] if np.isfinite(x[2]) else 0.0) for x in items[1:]], kind="stable") + 1
    n_chunks = min(len(order), 8 * workers)
    jobs = [(order[c::n_chunks], [items[i] for i in order[c::n_chunks]]) for c in range(n_chunks)]
    masks, failed = [None] * n, f0
    masks[0] = first[0]
    with get_context("spawn").Pool(min(workers, n_chunks)) as pool:
        for idx, ms, f in pool.imap_unordered(_denoise_chunk, jobs):
            failed += f
            for i, m in zip(idx, ms):
                masks[i] = m
    return masks, failed


def denoise_copies(output, workers: int = 1) -> dict:
    """write ms2_consensus_denoised.msp and ms2_spectra_denoised.msp: every record of the raw copy with only the ions
    specdenoise keeps (DENOISE: tier T0, the precursor's formula inferred from the feature m/z). A spectrum is judged
    on the feature's charge (the table's, 1 when unknown) and on its file's scale moved to the table's by the m/z
    offset in its Comment (mz_offset_ppm, 0 when absent); the written m/z values are the raw ones. A record left
    without peaks is not written. Returns {copy: (spectra, peaks before, peaks kept)}."""
    from .pipeline import log
    output = Path(output)
    tic = time.perf_counter()
    t = pd.read_csv(output / "aligned_feature_table.tsv", sep="\t", usecols=lambda c: c in ("group_id", "charge"))
    z_of = dict(zip(t["group_id"].astype(int), t["charge"].fillna(0).astype(int))) if "charge" in t else {}
    todo = []
    for raw in RAW:
        (output / SPECTRA[DENOISED[raw]]).unlink(missing_ok=True)
        src = output / SPECTRA[raw]
        if not src.exists():
            continue
        blocks = _msp_blocks(src)
        items = []
        for head, peaks in blocks:
            info = _comment(_field(head, "comment"))
            v = np.array("\t".join(x.strip() for x in peaks).split(), np.float64).reshape(-1, 2)
            sign = -1 if _field(head, "ionmode").lower().startswith("n") else 1
            z = sign * max(1, abs(int(z_of.get(int(info.get("group_id", -1)), 0))))
            off = float(info.get("mz_offset_ppm", 0.0))
            pmz = float(_field(head, "precursormz") or "nan")
            items.append((v[:, 0] * (1.0 - off * 1e-6), v[:, 1], pmz, z))
        todo.append((raw, blocks, items))
    masks, failed = _keep_masks([x for _, _, items in todo for x in items], max(1, int(workers)))
    counts, at = {}, 0
    for raw, blocks, items in todo:
        mk = masks[at:at + len(items)]
        at += len(items)
        tmp = output / (SPECTRA[DENOISED[raw]] + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for (head, peaks), keep in zip(blocks, mk):
                kept = [peaks[j] for j in np.flatnonzero(keep)]
                if not kept:
                    continue
                fh.write("".join((x + f" denoised={DENOISE_LABEL.replace(' ', '_')} n_raw={len(peaks)}"
                                  if x.lower().startswith("comment:") else x) + "\n" for x in head))
                fh.write(f"Num Peaks: {len(kept)}\n" + "".join(x + "\n" for x in kept) + "\n")
        tmp.replace(output / SPECTRA[DENOISED[raw]])
        counts[raw] = (len(items), int(sum(len(k) for k in mk)), int(sum(int(k.sum()) for k in mk)))
    if counts:
        part = ", ".join(f"{'consensus' if c == 'consensus' else 'highest TIC'} {k / max(n, 1):.0%} of {n} peaks"
                         for c, (_, n, k) in counts.items())
        log(f"ms2 denoised ({DENOISE_LABEL}): {sum(v[0] for v in counts.values())} spectra, kept {part}"
            f"{f'; {failed} could not be judged and were left whole' if failed else ''} "
            f"({time.perf_counter() - tic:.1f} s, {max(1, int(workers))} worker{'s' if workers > 1 else ''})")
    return counts


def default_workers() -> int:
    """processes for denoising outside a run (a search in the app): the CPUs this process may use, at most 8"""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    return max(1, min(8, n))


def default_copies(output) -> list[str]:
    """the copies whose best hits compete for a feature's default annotation (the higher-scoring one wins,
    annotations.default_hits): the denoised consensus and highest-TIC spectra the output holds, else the raw ones"""
    output = Path(output)
    held = lambda c: (output / SPECTRA[c]).exists() and (output / SPECTRA[c]).stat().st_size > 0
    have = [c for c in RAW if held(c)]
    return [DENOISED[c] for c in have if held(DENOISED[c])] or have or ["single"]


def default_label(copies) -> str:
    """the default copies as they read in a sentence ("... at or above 0.7 on the <label>")"""
    copies = list(copies)
    if len(copies) == 1:
        return LABEL[copies[0]]
    dn = all(c in DENOISED.values() for c in copies)
    return f"better of the {'denoised ' if dn else ''}consensus and highest-TIC spectra"


def ensure_denoised(output, workers: int = 1) -> bool:
    """denoise_copies when a raw copy has no denoised one yet, or a newer one (outputs made before the denoising);
    True when the denoised copies are there afterwards"""
    output = Path(output)
    for c in RAW:                   # a denoised copy never outlives its raw one
        if not (output / SPECTRA[c]).exists():
            (output / SPECTRA[DENOISED[c]]).unlink(missing_ok=True)
    raw = [output / SPECTRA[c] for c in RAW if (output / SPECTRA[c]).exists()]
    if not raw:
        return False
    den = [output / SPECTRA[DENOISED[c]] for c in RAW if (output / SPECTRA[c]).exists()]
    if not all(d.exists() and d.stat().st_mtime >= r.stat().st_mtime for r, d in zip(raw, den)):
        denoise_copies(output, workers)
    return any(d.exists() for d in den)


# key=value pairs of our Comment lines; file and scan are quoted (Thermo spectrum ids hold spaces and "=")
_COMMENT = re.compile(r'(\w+)=(?:"([^"]*)"|(\S+))')


def _comment(text: str) -> dict:
    return {k: a if a or not b else b for k, a, b in _COMMENT.findall(text)}


def read_representatives(output, which: str = "single") -> dict | None:
    """one copy of every feature's MS2 spectrum (which: "single" = ms2_spectra.msp, the highest-TIC spectrum;
    "consensus" = ms2_consensus.msp; "single_denoised", "consensus_denoised": their denoised copies), as arrays in
    file order: group_id, n_ms2, n_merged (spectra in the consensus,
    1 for single), vs_single (consensus only: its weighted entropy similarity to the highest-TIC spectrum; nan
    otherwise), prec_mz (the feature's m/z), rt_s (the native time of that, or the consensus's most abundant,
    spectrum), mz_offset_ppm (the spectrum's file offset from the table's m/z scale, 0 when not written), file, scan,
    ion_mode, off, mz, inten. None when the output has no such file or it is empty."""
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
            "mz_offset_ppm": np.array([float(d.get("mz_offset_ppm", 0.0)) for d in info], np.float64),
            "file": [d.get("file", "") for d in info], "scan": [d.get("scan", "") for d in info],
            "ion_mode": list(lib.ion_mode), "off": lib.off, "mz": lib.mz, "inten": lib.inten}
