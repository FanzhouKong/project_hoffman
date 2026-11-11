"""Results bundle for the HTML viewer: one binary blob built from a peak3d output folder.

Layout (little-endian): uint32 header length | header JSON (utf-8) | zero padding to 8 bytes | arrays.
The header lists every array by name with its dtype, byte offset (from the start of the array block, a
multiple of 8) and length, so the viewer maps them as typed arrays without parsing text.

Per-file values are stored log-quantised (uint16, 0 = zero; 0.04 % relative precision) and chromatograms
as uint16 fractions of each group's maximum on a common corrected-RT grid, so a report stays small and
opens instantly. The exact numbers stay in the TSV tables next to it."""
from __future__ import annotations

import gzip
import json
import math
import struct
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from .. import __version__
from ..io import Cloud, _lower_bound, _upper_bound

FORMAT = 1
BASE_COLS = ("group_id", "mz", "rt", "rt_min", "rt_max", "n_detected", "n_confirmed", "n_filled", "mz_ppm_spread",
             "rt_sd", "iso_offset", "charge", "compound_id", "role", "neutral_mass", "is_base", "n_members",
             "possible_halogen", "halogen_pattern")
C13 = 1.0033548378          # 13C - 12C
ISO_MAX_STEP = 3            # possible isotopes: M+1 .. M+3
# defaults keep a report of a few hundred files under ~15 MB
EIC_BUDGET = 3_000_000      # chromatogram values (uint16) across all groups
EIC_FILES = 16              # traces per group; larger studies show an evenly spaced subset of files
EIC_POINTS = 48             # grid points per trace
MAX_CELLS = 4_000_000       # group x file cells stored for the per-file view


# ------------------------------------------------------------------ packing ----
def pack(header: dict, arrays: dict) -> bytes:
    """header + named numpy arrays -> one blob (see module docstring)"""
    meta, chunks, pos = {}, [], 0
    for name, a in arrays.items():
        a = np.ascontiguousarray(a)
        if a.dtype.byteorder == ">":
            a = a.astype(a.dtype.newbyteorder("<"))
        meta[name] = {"dtype": a.dtype.str.lstrip("<|="), "offset": pos, "length": int(a.size)}
        b = a.tobytes()
        chunks.append(b)
        pad = (-len(b)) % 8
        chunks.append(b"\0" * pad)
        pos += len(b) + pad
    h = json.dumps({**header, "arrays": meta}, separators=(",", ":"), allow_nan=False).encode()
    lead = 4 + len(h)
    return struct.pack("<I", len(h)) + h + b"\0" * ((-lead) % 8) + b"".join(chunks)


def unpack(blob: bytes):
    """blob -> (header, {name: array}); the inverse of pack"""
    (n,) = struct.unpack_from("<I", blob, 0)
    header = json.loads(blob[4:4 + n])
    start = 4 + n + ((-(4 + n)) % 8)
    arrays = {k: np.frombuffer(blob, dtype=np.dtype(m["dtype"]).newbyteorder("<"), count=m["length"],
                               offset=start + m["offset"]) for k, m in header.pop("arrays").items()}
    return header, arrays


def _clean(x):
    """JSON-safe copy: NaN/inf -> None, numpy scalars -> Python"""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _records(df: pd.DataFrame) -> list[dict]:
    return [_clean(r) for r in df.to_dict("records")]


# ------------------------------------------------------------------ chromatograms ----
@njit(cache=True)
def eic_on_grid(off, mz, inten, rt, targets, tols, x0, dx, n_pts):
    """per group: the max-intensity EIC within targets +/- tols, on the grid x0 + j*dx (j < n_pts) of the
    scan times ``rt``. Scans falling in a grid cell are max-pooled; an empty cell interpolates linearly
    between the neighbouring scans when both hold signal, so a grid finer than the scan spacing does not
    leave holes inside a peak, and stays at zero when either neighbour is empty: a one-scan rise is drawn
    as a step, never as a ramp that looks like a shoulder the picker left out."""
    n = len(targets)
    S = len(rt)
    out = np.zeros((n, n_pts), np.float32)
    seen = np.zeros(n_pts, np.bool_)
    for g in range(n):
        lo_t = x0[g] - 0.5 * dx[g]
        hi_t = x0[g] + (n_pts - 0.5) * dx[g]
        a = max(np.searchsorted(rt, lo_t) - 1, 0)
        b = min(np.searchsorted(rt, hi_t) + 1, S)
        m = b - a
        if m <= 0:
            continue
        vals = np.zeros(m, np.float32)
        for s in range(a, b):
            lo = _lower_bound(mz, off[s], off[s + 1], targets[g] - tols[g])
            hi = _upper_bound(mz, lo, off[s + 1], targets[g] + tols[g])
            v = np.float32(0.0)
            for p in range(lo, hi):
                if inten[p] > v:
                    v = inten[p]
            vals[s - a] = v
        seen[:] = False
        for s in range(a, b):
            j = int(math.floor((rt[s] - lo_t) / dx[g]))
            if 0 <= j < n_pts:
                if (not seen[j]) or vals[s - a] > out[g, j]:
                    out[g, j] = vals[s - a]
                seen[j] = True
        for j in range(n_pts):
            if seen[j]:
                continue
            t = x0[g] + j * dx[g]
            k = np.searchsorted(rt[a:b], t)
            if k <= 0 or k >= m:
                continue
            if vals[k - 1] <= 0.0 or vals[k] <= 0.0:
                continue
            t0 = rt[a + k - 1]
            t1 = rt[a + k]
            w = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            out[g, j] = (1.0 - w) * vals[k - 1] + w * vals[k]
    return out


@njit(cache=True)
def scans_in_window(off, mz, inten, rt, targets, tols, lo, hi):
    """per group: (MS1 scans whose time ``rt`` lies in [lo, hi], those among them holding a centroid within
    targets +/- tols), int32 (n, 2). The bounds are inclusive with a small slack so the scans that set them are
    counted."""
    n = len(targets)
    out = np.zeros((n, 2), np.int32)
    eps = 1e-6
    for g in range(n):
        a = np.searchsorted(rt, lo[g] - eps)
        b = np.searchsorted(rt, hi[g] + eps)
        out[g, 0] = b - a
        for s in range(a, b):
            p = _lower_bound(mz, off[s], off[s + 1], targets[g] - tols[g])
            if p < off[s + 1] and mz[p] <= targets[g] + tols[g]:
                out[g, 1] += 1
    return out


def _eic_files(n_files: int, limit: int) -> list[int]:
    if n_files <= limit:
        return list(range(n_files))
    return sorted(set(np.linspace(0, n_files - 1, limit).round().astype(int).tolist()))


def _file_axis(output: Path, stem: str, cloud: Cloud):
    """the file's corrected scan times (float64, aligned axis) and its m/z offset per scan (ppm) or None"""
    warp = output / "rt_correction" / f"{stem}.tsv"
    moff = None
    corr = cloud.rt
    if warp.exists():
        wt = pd.read_csv(warp, sep="\t")
        c = wt["rt_corr"].to_numpy(float)
        if len(c) == cloud.n_scans:
            corr = c
            if "mz_offset_ppm" in wt:
                moff = wt["mz_offset_ppm"].to_numpy(float)
    return np.asarray(corr, np.float64), moff


def _native_mz(mz, rt, cloud: Cloud, corr, moff, offset_ppm: float):
    """consensus m/z at aligned times rt -> what the file measured: its offset at that time (a curve unless
    --mz-static), or the file's single offset when the warp table has no curve"""
    mz = np.asarray(mz, np.float64)
    if moff is None:
        return mz / (1.0 - offset_ppm * 1e-6)
    native = np.interp(np.asarray(rt, np.float64), corr, cloud.rt)
    return mz / (1.0 - np.interp(native, cloud.rt, moff) * 1e-6)


def table_stems(output, cols) -> list[str]:
    """the file columns of an aligned table (the picking manifest's stems when it is there)"""
    stems = [c for c in cols if c not in BASE_COLS]
    man = _read(Path(output) / "features" / "manifest.tsv")
    if man is not None and "stem" in man and set(man["stem"].astype(str)) <= set(stems):
        stems = [s for s in stems if s in set(man["stem"].astype(str))]   # extra columns of newer versions
    return stems


def file_offsets(output) -> dict:
    """each file's m/z offset (ppm) from the alignment summary, 0 where unknown"""
    summ = _read(Path(output) / "rt_correction" / "summary.tsv")
    if summ is None or "file" not in summ or "mz_offset_ppm" not in summ:
        return {}
    off = pd.to_numeric(summ["mz_offset_ppm"], errors="coerce").fillna(0.0)
    return {str(f): float(o) for f, o in zip(summ["file"], off)}


def ms1_pattern(output, cache_dir, stem: str, mz: float, rt: float, rt_lo: float, rt_hi: float, z: int, ppm: float,
                offset_ppm: float = 0.0) -> dict:
    """the MS1 isotope pattern of one feature in one file (peak3d.halogen.test_feature on the cached point cloud):
    the scans between the aligned peak bounds rt_lo..rt_hi (minutes), at least 3 around rt"""
    from ..halogen import test_feature
    output, cache_dir = Path(output), Path(cache_dir)
    npz = cache_dir / f"{stem}.npz"
    if not npz.exists():
        raise FileNotFoundError(f"no point-cloud cache for {stem} (the run deleted it: re-run without --no-keep-cache)")
    cloud = Cloud.load_npz(npz)
    corr, moff = _file_axis(output, stem, cloud)
    target = float(_native_mz([mz], [rt], cloud, corr, moff, offset_ppm)[0])
    s0 = int(np.searchsorted(corr, rt_lo))
    s1 = int(np.searchsorted(corr, rt_hi, side="right"))
    if s1 - s0 < 3:
        c = int(np.clip(np.searchsorted(corr, rt), 1, max(cloud.n_scans - 2, 1)))
        s0, s1 = min(s0, c - 1), max(s1, c + 2)
    s0, s1 = max(s0, 0), min(s1, cloud.n_scans)
    pat = test_feature(cloud, target, z, s0, s1, ppm)
    out = {"file": stem, "target_mz": target, "offset_ppm": float(target / mz * 1e6 - 1e6) if mz else 0.0,
           "scans": [s0, s1], "ppm": ppm, "z": max(int(z), 1)}
    if pat is None:
        out["note"] = "nothing at the feature's m/z in these scans"
        return out
    a = pat["apex_scan"]
    out.update(pat, rt_s=float(60 * corr[a]), rt_native_s=float(60 * cloud.rt[a]))
    return out


@njit(cache=True)
def eic_scans(off, mz, inten, lo, hi):
    """per scan s: the most intense centroid with lo[s] <= m/z <= hi[s] (0 where there is none)"""
    S = len(off) - 1
    out = np.zeros(S, np.float32)
    for s in range(S):
        v = np.float32(0.0)
        for p in range(_lower_bound(mz, off[s], off[s + 1], lo[s]), off[s + 1]):
            if mz[p] > hi[s]:
                break
            if inten[p] > v:
                v = inten[p]
        out[s] = v
    return out


def _source_cloud(output: Path, cache_dir: Path, stem: str) -> tuple[Cloud, str]:
    """a file's MS1 point cloud: the run's cache, else the mzML the run read (params.json), when it still exists"""
    npz = cache_dir / f"{stem}.npz"
    if npz.exists():
        return Cloud.load_npz(npz), "cache"
    params = json.loads((output / "params.json").read_text()) if (output / "params.json").exists() else {}
    for p in map(Path, params.get("input_files") or []):
        if p.stem == stem and p.suffix.lower() == ".mzml" and p.is_file():
            from ..io import load_cloud
            return load_cloud(p, (params.get("options") or {}).get("polarity")), "mzML"
    raise FileNotFoundError(f"no point-cloud cache for {stem} and its mzML file is not where the run read it")


def extract_eic(output, cache_dir, stem: str, mz: float, ppm: float, offset_ppm: float | None = 0.0,
                lo_ppm: float | None = None, hi_ppm: float | None = None) -> dict:
    """the full-run chromatogram of m/z +/- ppm in one file: per MS1 scan the most intense centroid in the window.
    offset_ppm None: m/z as given; else the target moves to what the file measured, by its m/z offset curve when the
    alignment wrote one and by offset_ppm (its single offset) otherwise, as for the report's chromatograms.
    lo_ppm / hi_ppm (both): the window from target - lo_ppm to target + hi_ppm instead (a feature's stored per-file
    window, eic_windows). Times in seconds, aligned (the file's RT correction) and native."""
    output, cache_dir = Path(output), Path(cache_dir)
    cloud, source = _source_cloud(output, cache_dir, stem)
    corr, moff = _file_axis(output, stem, cloud)
    if offset_ppm is None:
        target = np.full(cloud.n_scans, float(mz))
    elif moff is not None:
        target = float(mz) / (1.0 - moff * 1e-6)
    else:
        target = np.full(cloud.n_scans, float(mz) / (1.0 - offset_ppm * 1e-6))
    lo_ppm, hi_ppm = (float(ppm), float(ppm)) if lo_ppm is None or hi_ppm is None else (float(lo_ppm), float(hi_ppm))
    y = eic_scans(cloud.off, cloud.mz, cloud.inten, target * (1.0 - lo_ppm * 1e-6), target * (1.0 + hi_ppm * 1e-6))
    shift = target / float(mz) * 1e6 - 1e6
    return {"file": stem, "mz": float(mz), "ppm": float(ppm), "lo_ppm": lo_ppm, "hi_ppm": hi_ppm, "source": source,
            "n_scans": int(cloud.n_scans),
            "offset_ppm": [float(shift.min()), float(shift.max())] if len(shift) else [0.0, 0.0],
            "rt_s": np.round(60.0 * corr, 3).tolist(), "rt_native_s": np.round(60.0 * cloud.rt, 3).tolist(),
            "intensity": [float(f"{v:.5g}") for v in y.tolist()]}


def eic_windows(target, mz_min, mz_max, med_lo, med_hi, ppm: float):
    """per group, the m/z window (ppm below, ppm above the target) a file's chromatogram is drawn over: where the file
    has the feature, the union of target +/- ppm and the m/z span of the centroids the picker took for it (mz_min,
    mz_max; on QTOF a weak feature's centroids spread well past 5 ppm, and a +/- ppm window cut its peak); where it has
    none, the group's median span (med_lo, med_hi, ppm from the feature m/z) over the files that have it, at least +/-
    ppm; NaN spans fall back to +/- ppm"""
    target = np.asarray(target, float)
    lo, hi = np.full(len(target), float(ppm)), np.full(len(target), float(ppm))
    with np.errstate(invalid="ignore"):
        det = np.isfinite(mz_min) & np.isfinite(mz_max) & (mz_min > 0) & (mz_max >= mz_min)
        lo[det] = np.maximum(ppm, (target[det] - mz_min[det]) / target[det] * 1e6)
        hi[det] = np.maximum(ppm, (mz_max[det] - target[det]) / target[det] * 1e6)
        nd = ~det & np.isfinite(med_lo) & np.isfinite(med_hi)
        lo[nd] = np.maximum(ppm, med_lo[nd])
        hi[nd] = np.maximum(ppm, med_hi[nd])
    return lo, hi


def _feature_spans(output: Path, stems, group_ids):
    """per group x file (stems), the m/z span of the file's own feature of that group (per-file feature tables:
    mz_min, mz_max, native m/z; NaN where the file has none) and its relative extent below / above the feature m/z
    (ppm)"""
    G, F = len(group_ids), len(stems)
    mz_min, mz_max = np.full((G, F), np.nan), np.full((G, F), np.nan)
    rel_lo, rel_hi = np.full((G, F), np.nan), np.full((G, F), np.nan)
    row_of = {int(g): i for i, g in enumerate(group_ids)}
    for j, s in enumerate(stems):
        f = Path(output) / "features" / f"{s}.tsv"
        try:
            d = pd.read_csv(f, sep="\t", usecols=["group_id", "mz", "mz_min", "mz_max"])
        except (FileNotFoundError, ValueError):       # no table, or one written before the alignment added group_id
            continue
        d = d[np.isfinite(d["group_id"]) & (d["mz"] > 0)]
        idx = np.array([row_of.get(int(g), -1) for g in d["group_id"]], np.int64)
        ok = idx >= 0
        idx, d = idx[ok], d[ok]
        mz_min[idx, j], mz_max[idx, j] = d["mz_min"].to_numpy(float), d["mz_max"].to_numpy(float)
        m = d["mz"].to_numpy(float)
        rel_lo[idx, j] = (m - mz_min[idx, j]) / m * 1e6
        rel_hi[idx, j] = (mz_max[idx, j] - m) / m * 1e6
    return mz_min, mz_max, rel_lo, rel_hi


def _nanmedian_rows(x):
    """row medians ignoring NaN; NaN for a row with none"""
    out = np.full(x.shape[0], np.nan)
    for i, r in enumerate(x):
        r = r[np.isfinite(r)]
        if len(r):
            out[i] = float(np.median(r))
    return out


def _chromatograms(output: Path, cache_dir: Path, stems, table, files_info, ppm, budget, max_files, n_pts):
    """corrected-RT chromatograms of the most intense groups in up to max_files files, or (None, reason)"""
    if not cache_dir.is_dir():
        return None, "the point-cloud cache was removed after the run"
    shown = _eic_files(len(stems), max_files)
    missing = [stems[k] for k in shown if not (cache_dir / f"{stems[k]}.npz").exists()]
    if missing:
        return None, f"cache files missing for {len(missing)} files"
    G = len(table)
    if G == 0:
        return None, "no groups"
    F = len(shown)
    n_groups = int(min(G, max(1, budget // (F * n_pts))))
    order = np.argsort(-table["_max"].to_numpy(), kind="stable")[:n_groups]
    rows = np.sort(order)
    t = table.iloc[rows]
    rt_lo = t["rt_min"].to_numpy(float)
    rt_hi = t["rt_max"].to_numpy(float)
    pad = np.maximum(0.75 * (rt_hi - rt_lo), 0.1)
    x0 = rt_lo - pad
    dx = (rt_hi + pad - x0) / (n_pts - 1)
    mz_c = t["mz"].to_numpy(float)
    traces = np.zeros((n_groups, F, n_pts), np.float32)
    n_scans = np.zeros((n_groups, F, 2), np.uint16)       # MS1 scans inside [rt_min, rt_max] per file, and with signal
    offsets = {r["stem"]: (r.get("mz_offset_ppm") or 0.0) for r in files_info}
    # each file's window: its own feature's m/z span (at least +/- ppm); the group's median span where it has none
    span_lo, span_hi, rel_lo, rel_hi = _feature_spans(output, [stems[k] for k in shown], t["group_id"].to_numpy())
    med_lo, med_hi = _nanmedian_rows(rel_lo), _nanmedian_rows(rel_hi)
    win_lo = np.zeros((n_groups, F), np.float32)
    win_hi = np.zeros((n_groups, F), np.float32)
    for j, k in enumerate(shown):
        s = stems[k]
        cloud = Cloud.load_npz(cache_dir / f"{s}.npz")
        corr, moff = _file_axis(output, s, cloud)
        target = _native_mz(mz_c, t["rt"].to_numpy(float), cloud, corr, moff, offsets.get(s, 0.0))
        lo, hi = eic_windows(target, span_lo[:, j], span_hi[:, j], med_lo, med_hi, ppm)
        win_lo[:, j], win_hi[:, j] = lo, hi
        centre, tol = target * (1.0 + 0.5 * (hi - lo) * 1e-6), target * 0.5 * (lo + hi) * 1e-6
        traces[:, j, :] = eic_on_grid(cloud.off, cloud.mz, cloud.inten, corr, centre, tol, x0, dx, n_pts)
        n_scans[:, j, :] = np.minimum(scans_in_window(cloud.off, cloud.mz, cloud.inten, corr, centre, tol, rt_lo, rt_hi),
                                      65535)
    scale = traces.reshape(n_groups, -1).max(axis=1)
    q = np.zeros_like(traces, dtype=np.uint16)
    ok = scale > 0
    q[ok] = np.round(traces[ok] / scale[ok, None, None] * 65535).astype(np.uint16)
    info = {"files": shown, "points": n_pts, "n": n_groups, "ppm": ppm, "of": G,
            "selection": "most intense groups" if n_groups < G else "all groups", "window": "feature_span"}
    arrays = {"eic_row": rows.astype(np.int32), "eic_x0": x0.astype(np.float32), "eic_dx": dx.astype(np.float32),
              "eic_scale": scale.astype(np.float32), "eic_q": q.reshape(-1), "eic_ns": n_scans.reshape(-1),
              "eic_lo": win_lo.reshape(-1), "eic_hi": win_hi.reshape(-1)}
    return (info, arrays), None


# ------------------------------------------------------------------ MS2 and identifications ----
def _take(off, arrays, idx):
    """rows idx of a ragged table (off: n + 1 offsets into every array in arrays) -> (offsets, arrays)"""
    idx = np.asarray(idx, np.int64)
    lens = (off[1:] - off[:-1])[idx]
    new = np.zeros(len(idx) + 1, np.int64)
    np.cumsum(lens, out=new[1:])
    sel = np.repeat(off[idx] - new[:-1], lens) + np.arange(new[-1])
    return new.astype(np.int32), [np.asarray(a)[sel] for a in arrays]


ID_TEXT = ("name", "precursor_type", "formula", "inchikey", "smiles", "denoise_ion")
SPEC_CODE = {"consensus": 0, "single": 1, "consensus_denoised": 2, "single_denoised": 3}
# what refresh_ids swaps: these arrays and header keys
MS2_PREFIXES = ("ms2_", "ms2c_", "id_", "mk_", "dk_")
MS2_KEYS = ("ms2", "ms2c", "ids", "ann", "ms2_note")


def _keep_mask(reps, den, k):
    """per peak of the records k of reps (in that order): 1 when den, its denoised copy, holds the peak (the same m/z:
    both files carry the same peak lines), else 0; a record den lacks lost all its peaks"""
    off, doff = np.asarray(reps["off"]), np.asarray(den["off"])
    at = {int(g): j for j, g in enumerate(den["group_id"])}
    parts = []
    for i in k:
        m = reps["mz"][off[i]:off[i + 1]]
        j = at.get(int(reps["group_id"][i]))
        parts.append(np.zeros(len(m), np.uint8) if j is None else
                     np.isin(m, den["mz"][doff[j]:doff[j + 1]]).astype(np.uint8))
    return np.concatenate(parts) if parts else np.zeros(0, np.uint8)


def _spectra(reps, row_of, prefix, den=None):
    """one MS2 copy (ms2.read_representatives) as arrays keyed by table row, and its header part; with den (its
    denoised copy) also {prefix}keep, which peaks the denoising kept"""
    k = np.array([i for i, g in enumerate(reps["group_id"]) if int(g) in row_of], np.int64)
    off, (mz, inten) = _take(np.asarray(reps["off"]), (reps["mz"], reps["inten"]), k)
    arrays = {f"{prefix}row": np.array([row_of[int(reps["group_id"][i])] for i in k], np.int32),
              f"{prefix}n": np.clip(np.asarray(reps["n_ms2"])[k], 0, 65535).astype(np.uint16),
              f"{prefix}merged": np.clip(np.asarray(reps["n_merged"])[k], 0, 65535).astype(np.uint16),
              f"{prefix}files": np.clip(np.asarray(reps["n_files"])[k], 0, 65535).astype(np.uint16),
              f"{prefix}vs": np.asarray(reps["vs_single"], np.float32)[k],
              f"{prefix}prec": np.asarray(reps["prec_mz"], np.float64)[k],
              f"{prefix}rt": np.asarray(reps["rt_s"], np.float32)[k], f"{prefix}off": off,
              f"{prefix}mz": mz.astype(np.float32), f"{prefix}int": inten.astype(np.float32)}
    if den is not None:
        arrays[f"{prefix}keep"] = _keep_mask(reps, den, k)
    return arrays, {"n": int(len(k)), "file": [str(reps["file"][i]) for i in k], "scan": [str(reps["scan"][i]) for i in k],
                    "denoised": den is not None}


def _marks(output: Path, row_of, stems):
    """where the spectra of the consensus and the highest-TIC spectrum were taken, on the aligned axis (s):
    mk_row, mk_t, mk_kind (1 in the consensus, 2 highest TIC, 3 both), mk_file (index into stems, -1 unknown)"""
    mp = output / "ms2_map.tsv"
    if not mp.exists():
        return {}
    file_of = {s: i for i, s in enumerate(stems)}
    m = pd.read_csv(mp, sep="\t", dtype={"file": str, "scan": str})
    if "in_consensus" not in m:
        m["in_consensus"] = 0
    t = m["rt_aligned_s"] if "rt_aligned_s" in m else m["rt_s"]
    kind = m["in_consensus"].fillna(0).astype(int) + 2 * m["representative"].fillna(0).astype(int)
    keep = (kind > 0) & m["group_id"].isin(list(row_of))
    return {"mk_row": np.array([row_of[int(g)] for g in m.loc[keep, "group_id"]], np.int32),
            "mk_t": t[keep].to_numpy(np.float32), "mk_kind": kind[keep].to_numpy(np.uint8),
            "mk_file": np.array([file_of.get(f, -1) for f in m.loc[keep, "file"]], np.int16)}


def _denoise_masks(output: Path, hits: pd.DataFrame, row_of: dict, arrays: dict) -> dict | None:
    """the denoising search's masks (identifications_denoise.npz): id_dkey per hit (-1 none), dk_off / dk_keep the
    peaks each mask keeps over its copy's spectrum (ms2_* or ms2c_*, same peak order); a mask whose length no
    longer matches that spectrum (MS2 remapped since the search) is dropped. None for other searches"""
    from ..library import DENOISE_FILE
    p = output / DENOISE_FILE
    if "denoise_key" not in hits or not p.exists():
        return None
    with np.load(p) as z:
        gid, copy, off, keep = z["group_id"], z["copy"], z["off"], z["keep"]
    peaks = {}
    for c, prefix in ((0, "ms2c_"), (1, "ms2_")):
        if prefix + "row" in arrays:
            o = arrays[prefix + "off"]
            for k, row in enumerate(arrays[prefix + "row"].tolist()):
                peaks[(c, row)] = int(o[k + 1] - o[k])
    ok = np.array([peaks.get((int(c), row_of.get(int(g), -1)), -1) == int(off[i + 1] - off[i])
                   for i, (g, c) in enumerate(zip(gid, copy))], bool)
    d = np.nan_to_num(hits["denoise_key"].to_numpy(np.float64), nan=-1).astype(np.int64)
    d[(d >= len(ok)) | (d < -1)] = -1
    d = np.where((d >= 0) & ok[np.maximum(d, 0)], d, -1)
    return {"id_dkey": d.astype(np.int32), "dk_off": off.astype(np.int32), "dk_keep": keep.astype(np.uint8)}


def ms2_ids(output, group_ids, stems=None) -> tuple[dict, dict]:
    """both MS2 copies of every group that has MS2 (ms2.read_representatives: the highest-TIC spectrum as ms2_*, the
    consensus as ms2c_*, each with the peaks its denoised copy kept as ms2_keep / ms2c_keep when the output has one),
    where their spectra were taken (mk_*), the hits of the last identity search on all copies (identifications.tsv,
    .msp, .json; id_spec SPEC_CODE, id_lib the library record, id_lk its spectrum among id_off/id_mz/id_int, each
    library spectrum once) and the user's annotations, as header parts and
    arrays keyed by row of the aligned table (group_ids in table order; stems: its file columns, read from the table
    when not given). Header keys ms2 / ms2c / ids are None when the output has no such part; ann maps a row (as
    text) to the user's notes on it (annotations.load_user)"""
    from ..annotations import load_user, read_hits
    from ..ms2 import DENOISED, SPECTRA, _comment, read_representatives
    output = Path(output)
    row_of = {int(g): i for i, g in enumerate(np.asarray(group_ids))}
    head, arrays = {"ms2": None, "ms2c": None, "ids": None, "ann": None}, {}
    for which, key, prefix in (("single", "ms2", "ms2_"), ("consensus", "ms2c", "ms2c_")):
        reps = read_representatives(output, which)
        if reps is not None and len(reps["group_id"]):
            den = read_representatives(output, DENOISED[which])
            if den is None and (output / SPECTRA[DENOISED[which]]).exists():
                den = {"group_id": np.zeros(0, np.int64), "off": np.zeros(1, np.int64), "mz": np.zeros(0)}  # all removed
            a, head[key] = _spectra(reps, row_of, prefix, den)
            arrays.update(a)
    if head["ms2"] is not None:
        if stems is None:
            cols = pd.read_csv(output / "aligned_feature_table.tsv", sep="\t", nrows=0).columns
            stems = [c for c in cols if c not in BASE_COLS]
        arrays.update(_marks(output, row_of, stems))
    hits = read_hits(output) if head["ms2"] is not None else None
    if hits is not None:
        hits = hits.copy()                      # read_hits keeps its frame for the next caller
        meta = json.loads((output / "identifications.json").read_text()) if (output / "identifications.json").exists() else {}
        lib = None
        hits["_rec"] = -1                         # each hit's record in identifications.msp
        if (output / "identifications.msp").exists():
            from ..library import read_msp
            lib = read_msp(output / "identifications.msp")
            info = [_comment(c) for c in lib.comments]
            if info and all("library_index" in d for d in info) and "library_index" in hits:   # one per library record
                rec = {int(d["library_index"]): i for i, d in enumerate(info)}
                hits["_rec"] = [rec.get(int(v), -1) for v in hits["library_index"]]
            elif len(lib.names) == len(hits):        # older searches: one record per hit, in the table's order
                hits["_rec"] = np.arange(len(hits))
            if (hits["_rec"] < 0).any():             # written together; a mismatch means a damaged pair
                lib = None
        hits["_row"] = [row_of.get(int(g), -1) for g in hits["group_id"]]
        hits["_spec"] = hits["spectrum"].map(SPEC_CODE).fillna(1).astype(np.uint8)
        hits = hits[hits["_row"] >= 0].sort_values(["_row", "_spec", "rank"], kind="stable")
        num = lambda c, d=np.nan: hits[c].to_numpy(np.float64) if c in hits else np.full(len(hits), d)
        arrays.update(id_row=hits["_row"].to_numpy(np.int32), id_spec=hits["_spec"].to_numpy(np.uint8),
                      id_rank=np.clip(num("rank", 1), 0, 65535).astype(np.uint16),
                      id_score=num("score").astype(np.float32),
                      id_nmatch=np.clip(np.nan_to_num(num("n_matched", 0)), 0, 65535).astype(np.uint16),
                      id_ppm=num("ppm_error").astype(np.float32), id_prec=num("library_precursor_mz"),
                      id_lib=np.nan_to_num(num("library_index", -1), nan=-1).astype(np.int32))
        dk = _denoise_masks(output, hits, row_of, arrays)
        if dk is not None:
            arrays.update(dk)
        if lib is not None:
            used, lk = np.unique(hits["_rec"].to_numpy(np.int64), return_inverse=True)
            off, (mz, inten) = _take(np.asarray(lib.off), (lib.mz, lib.inten), used)
            arrays.update(id_lk=lk.astype(np.int32), id_off=off, id_mz=mz.astype(np.float32),
                          id_int=inten.astype(np.float32))
        head["ids"] = {**meta, "n": int(len(hits)), "spectra": lib is not None,
                       **{c: hits[c].fillna("").astype(str).tolist() if c in hits else [""] * len(hits)
                          for c in ID_TEXT}}
    user = load_user(output)
    if user:
        head["ann"] = {str(row_of[g]): v for g, v in user.items() if g in row_of}
    return head, arrays


def ms2_ids_blob(output) -> bytes:
    """gzip'd bundle holding only the MS2 and identification parts, for a viewer that already has the rest"""
    ids = pd.read_csv(Path(output) / "aligned_feature_table.tsv", sep="\t", usecols=["group_id"])["group_id"].to_numpy()
    head, arrays = ms2_ids(output, ids)
    return gzip.compress(pack(_clean({"format": FORMAT, "kind": "peak3d-ms2", **head}), arrays), compresslevel=6, mtime=0)


# ------------------------------------------------------------------ build ----
def possible_isotopes(mz, rt, height, iso_offset, charge, ppm: float, rt_win: float):
    """features the isotope annotation left monoisotopic that nevertheless sit k x 1.00335/z above a stronger
    feature within rt_win minutes (mass arithmetic only, no peak-shape evidence): (k per feature, 0 = none;
    row of that stronger feature, -1 = none). The viewer marks them as possible isotopes"""
    G = len(mz)
    k_out = np.zeros(G, np.int8)
    of_out = np.full(G, -1, np.int32)
    if G == 0:
        return k_out, of_out
    mz = np.asarray(mz, np.float64); rt = np.asarray(rt, np.float64); height = np.asarray(height, np.float64)
    z = np.where(np.asarray(charge) > 0, np.asarray(charge), 1).astype(np.float64)
    order = np.argsort(mz)
    mz_s = mz[order]
    tol = mz * ppm * 1e-6
    best_h = np.zeros(G)
    for k in range(1, ISO_MAX_STEP + 1):
        target = mz - k * C13 / z
        lo = np.searchsorted(mz_s, target - tol, side="left")
        hi = np.searchsorted(mz_s, target + tol, side="right")
        has = (hi > lo) & (np.asarray(iso_offset) == 0)
        for g in np.flatnonzero(has):
            cand = order[lo[g]:hi[g]]
            cand = cand[(np.abs(rt[cand] - rt[g]) <= rt_win) & (height[cand] > height[g]) & (cand != g)]
            if len(cand) == 0:
                continue
            p = cand[np.argmax(height[cand])]
            if k_out[g] == 0 or height[p] > best_h[g]:       # the nearest step wins unless a farther parent is stronger
                k_out[g] = k
                of_out[g] = p
                best_h[g] = height[p]
    return k_out, of_out


def _read(path: Path) -> pd.DataFrame | None:
    """a TSV of the output, file names (stem, file) kept as text: a file named 003.mzML must not become the number 3"""
    return pd.read_csv(path, sep="\t", dtype={"stem": str, "file": str}) if path.exists() else None


def _hist(values, lim, n_bins=61):
    bins = np.linspace(-lim, lim, n_bins + 1)
    return np.histogram(np.clip(values, -lim, lim), bins)[0].tolist()


def _mad(v):
    v = np.asarray(v, float)
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) else None


def build(output, cache_dir=None, *, eic_budget: int = EIC_BUDGET, eic_files: int = EIC_FILES,
          eic_points: int = EIC_POINTS, max_cells: int = MAX_CELLS, exclude=()) -> bytes:
    """bundle for one output folder (raw bytes; callers gzip). cache_dir defaults to <output>/.cache. exclude: file
    stems left out (subset.subset_frame: their columns, the features only they hold, their curves and anchors), named
    in the header's "excluded" """
    from . import subset
    output = Path(output)
    cache_dir = Path(cache_dir) if cache_dir else output / ".cache"
    table_path = output / "aligned_feature_table.tsv"
    if not table_path.exists():
        raise FileNotFoundError(f"not a peak3d output folder (no aligned_feature_table.tsv): {output}")
    params = json.loads((output / "params.json").read_text()) if (output / "params.json").exists() else {}
    options = params.get("options", {})
    table = pd.read_csv(table_path, sep="\t")
    man = _read(output / "features" / "manifest.tsv")
    stems = table_stems(output, table.columns)
    drop = [s for s in stems if s in {str(x) for x in exclude or ()}]
    excluded = None
    if drop:
        n_before = len(table)
        table = subset.subset_frame(output, table, table_path, stems, drop)
        stems = [s for s in stems if s not in set(drop)]
        flags = subset.file_flags(output)
        excluded = {"files": [{"stem": s, "flag": flags.get(s, "")} for s in drop], "n_groups_before": n_before}
    G, N = len(table), len(stems)
    V = table[stems].to_numpy(np.float32) if N else np.zeros((G, 0), np.float32)
    table["_max"] = V.max(axis=1) if N else 0.0

    # files: picking manifest + alignment summary, in table order
    summ = _read(output / "rt_correction" / "summary.tsv")
    files = pd.DataFrame({"stem": stems})
    if man is not None and "stem" in man:
        files = files.merge(man.drop(columns=[c for c in ("path",) if c in man]), on="stem", how="left")
    if summ is not None and "file" in summ:
        files = files.merge(summ.rename(columns={"file": "stem"}), on="stem", how="left")
    files_info = _records(files)

    # groups
    ppm = float(options.get("ppm") or 5.0)

    def col(name, dtype, default=0):
        return table[name].to_numpy(dtype) if name in table else np.full(G, default, dtype)

    nz = V > 0
    cnt = nz.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(nz, V, 0).sum(axis=1) / cnt
        var = np.where(nz, (V - mean[:, None]) ** 2, 0).sum(axis=1) / np.maximum(cnt - 1, 1)
        cv = np.where(cnt >= 2, 100 * np.sqrt(var) / mean, np.nan)
    arrays = {"g_id": col("group_id", np.int32), "g_mz": col("mz", np.float64), "g_rt": col("rt", np.float32),
              "g_rt_min": col("rt_min", np.float32), "g_rt_max": col("rt_max", np.float32),
              "g_ndet": col("n_detected", np.uint16), "g_nconf": col("n_confirmed", np.uint16),
              "g_nfill": col("n_filled", np.uint16), "g_ppm": col("mz_ppm_spread", np.float32, np.nan),
              "g_rtsd": col("rt_sd", np.float32, np.nan), "g_iso": col("iso_offset", np.int8),
              "g_z": col("charge", np.int8), "g_max": table["_max"].to_numpy(np.float32),
              "g_cv": cv.astype(np.float32)}
    # compound groups (adducts, in-source fragments, multimers and their isotopes of one compound) and the isotope
    # candidates the shape test did not confirm
    compounds = None
    if "compound_id" in table:
        roles = table["role"].fillna("").astype(str) if "role" in table else pd.Series([""] * G)
        names = sorted(set(roles) - {""})
        code = {r: i + 1 for i, r in enumerate(names)}
        nmem = col("n_members", np.uint16, 1)
        arrays.update({"g_cid": col("compound_id", np.int32, -1), "g_role": roles.map(lambda r: code.get(r, 0)).to_numpy(np.uint16),
                       "g_nm": col("neutral_mass", np.float64, np.nan), "g_base": col("is_base", np.uint8), "g_nmem": nmem})
        compounds = {"roles": [""] + names, "n_multi": int(np.unique(arrays["g_cid"][nmem > 1]).size),
                     "n_in_multi": int((nmem > 1).sum())}
    rt_win = float(params.get("rt_tol_min") or 0.05)
    k_iso, of_iso = possible_isotopes(arrays["g_mz"], arrays["g_rt"], arrays["g_max"], arrays["g_iso"], arrays["g_z"],
                                      max(ppm, 5.0), rt_win)
    arrays["g_isoq"] = k_iso
    arrays["g_isoq_of"] = of_iso
    halogen_names = None
    if "possible_halogen" in table:         # 0 not tested, 1 no, 2 maybe, 3 yes; the best hypothesis by name
        from ..halogen import HYPOTHESES, VERDICTS
        v = table["possible_halogen"].fillna("").astype(str).to_numpy()
        arrays["g_hal"] = np.array([VERDICTS.index(x) if x in VERDICTS else 0 for x in v], np.uint8)
        halogen_names = [""] + [n for n, _ in HYPOTHESES]
        b = table["halogen_pattern"].fillna("").astype(str).to_numpy() if "halogen_pattern" in table else np.full(G, "")
        arrays["g_halhint"] = np.array([halogen_names.index(x) if x in halogen_names else 0 for x in b], np.uint8)

    # per-file values and how each cell was obtained: 3 detected, 2 gap-filled with a confirmed peak,
    # 1 gap-filled (no confirmed peak), 0 nothing
    matrix = None
    if G * N and G * N <= max_cells:
        pos = V[V > 0]
        lo = math.floor(math.log10(float(pos.min()))) if len(pos) else 0
        hi = max(math.ceil(math.log10(float(pos.max()))) if len(pos) else 1, lo + 1)
        with np.errstate(divide="ignore"):
            qv = np.where(V > 0, 1 + np.round((np.log10(np.maximum(V, 1e-30)) - lo) / (hi - lo) * 65534), 0)
        status = np.where(V > 0, 1, 0).astype(np.uint8)
        pres = _read(output / "presence_mask.tsv")
        from_mask = pres is not None and all(s in pres for s in stems)
        if from_mask:
            p = pres.set_index("group_id").reindex(arrays["g_id"])[stems].fillna(0).to_numpy(np.int8)
            status = np.where(p == 2, 3, np.where(p == 1, 2, status)).astype(np.uint8)
        arrays["m_q"] = qv.astype(np.uint16).reshape(-1)
        arrays["m_status"] = status.reshape(-1)
        # presence: the status tells detected cells (3) from gap-filled ones (1, 2); without presence_mask.tsv
        # (external kernels) every cell with a value is 1
        matrix = {"log10_lo": lo, "log10_hi": hi, "presence": bool(from_mask)}
    matrix_note = None if matrix or not G * N else f"{G} groups x {N} files is above the {max_cells:,} cell limit"

    # RT correction: per-file shift curves (downsampled) and anchor residuals
    dx_, dy_, doff = [], [], [0]
    for s in stems:
        w = _read(output / "rt_correction" / f"{s}.tsv")
        if w is None or not len(w):
            doff.append(doff[-1])
            continue
        idx = np.unique(np.linspace(0, len(w) - 1, min(len(w), 400)).round().astype(int))
        x = w["rt_native"].to_numpy(float)[idx]
        dx_.append(x.astype(np.float32))
        dy_.append((60 * (w["rt_corr"].to_numpy(float)[idx] - x)).astype(np.float32))
        doff.append(doff[-1] + len(idx))
    arrays["drift_x"] = np.concatenate(dx_) if dx_ else np.zeros(0, np.float32)
    arrays["drift_y"] = np.concatenate(dy_) if dy_ else np.zeros(0, np.float32)
    arrays["drift_off"] = np.asarray(doff, np.int32)
    residuals = None
    anc = _read(output / "rt_correction" / "anchors.tsv")
    if anc is not None and drop and "file" in anc:
        anc = anc[~anc["file"].astype(str).isin(drop)]
    if anc is not None and len(anc):
        b = anc["residual_before_s"].to_numpy(float)
        a = anc.loc[anc["is_inlier"].astype(bool), "residual_after_s"].to_numpy(float)
        both = np.abs(np.concatenate([b, a]))
        lim = float(np.percentile(both[np.isfinite(both)], 99)) if len(both) else 1.0
        lim = lim if lim > 0 else 1.0
        residuals = {"lim": lim, "before": _hist(b, lim), "after": _hist(a, lim), "n_before": int(len(b)),
                     "n_after": int(len(a)), "mad_before": _mad(b), "mad_after": _mad(a)}

    eic, eic_note = None, None
    if G and N and eic_budget > 0:
        got, eic_note = _chromatograms(output, cache_dir, stems, table, files_info, ppm, eic_budget, eic_files,
                                       eic_points)
        if got:
            eic, extra = got
            arrays.update(extra)

    try:
        ms2_head, extra = ms2_ids(output, arrays["g_id"], stems)
        arrays.update(extra)
    except Exception as e:   # a damaged MS2 or identification file never costs the rest of the results
        ms2_head = {**{k: None for k in MS2_KEYS}, "ms2_note": f"{type(e).__name__}: {e}"}

    header = {"format": FORMAT, "kind": "peak3d-results", "title": output.resolve().name,
              "kernel": params.get("kernel", "peak3d"), "kernel_label": params.get("kernel_label") or params.get("kernel", "peak3d"),
              "kernel_version": params.get("kernel_version") or params.get("peak3d_version"), **ms2_head,
              "created": time.strftime("%Y-%m-%d %H:%M"), "viewer_version": __version__,
              "quant": options.get("quant", "height"), "n_groups": G, "stems": stems, "files": files_info,
              "params": _clean(params), "matrix": matrix, "matrix_note": matrix_note, "eic": eic,
              "eic_note": eic_note, "residuals": residuals, "compounds": compounds,
              "iso_rt_win": rt_win, "halogen_names": halogen_names, "excluded": excluded}
    return pack(_clean(header), arrays)
