"""Pipeline stages behind the CLI: pick files (one worker per file), align, group, gap-fill, write.

Worker pools use the ``spawn`` start method: numba's thread pool is not fork-safe once the parent
has run a parallel kernel (the JIT warm-up), and forked children deadlock. Spawned workers load
the compiled kernels from the on-disk numba cache instead."""
from __future__ import annotations

import json
import os
import pickle
import shutil
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .align import AlignParams, AlignResult, FileAlignment, align_run, anchors_table, summary_table
from .features import COLUMNS, build_features, precise_mz, write_features
from .gapfill import fill_file
from .grouping import consensus, group_features, merge_complementary
from .io import Cloud, load_cloud
from .pick import pick_cloud
from .warp import FitDiag, TableWarp

RESERVED = {"group_id", "mz", "rt", "rt_min", "rt_max", "n_detected", "n_filled", "mz_ppm_spread", "rt_sd",
            "iso_offset", "charge"}


def log(msg):
    print(f"[peak3d] {msg}", flush=True)


# ------------------------------------------------------------------ picking ----
def _warm_jit():
    from .synth import Peak, make_cloud
    cloud, _ = make_cloud([Peak(300.0, 0.5, 1e5, 0.05), Peak(400.0, 0.7, 5e4, 0.05, iso=(0.1,))],
                          n_scans=200, dt=0.005, floor=50.0, n_noise=2000, thresh=20.0)
    b = pick_cloud(cloud)
    build_features(cloud, b)
    from .gapfill import fill_file as _ff
    _ff(cloud, TableWarp.identity(cloud.rt), np.array([300.0]), np.array([0.5]), np.array([0.05]), 5.0)


def _load_model(path):
    if path is None:
        return None
    with open(path, "rb") as fh:
        model = pickle.load(fh)
    if not hasattr(model, "predict_proba") or not hasattr(model, "feature_columns"):
        raise ValueError("re-scorer must expose predict_proba() and feature_columns")
    return model


def _apply_model(feats, model):
    X = feats[list(model.feature_columns)].values
    feats["score_model"] = model.predict_proba(X)[:, 1]
    feats["score"] = feats["score_model"]
    return feats


def _pick_one(job):
    path, features_dir, cache_dir, min_score, keep_rejected, polarity, threads, model_path = job
    import numba
    numba.set_num_threads(max(1, int(threads)))
    t = {}
    tic = time.perf_counter()
    cloud = load_cloud(path, polarity)
    t["parse"] = time.perf_counter() - tic
    stem = cloud.name
    tic = time.perf_counter()
    cloud.save_npz(Path(cache_dir) / f"{stem}.npz")
    t["cache"] = time.perf_counter() - tic
    tic = time.perf_counter()
    b = pick_cloud(cloud)
    t.update({f"pick_{k}": v for k, v in b.timings.items()})
    t["pick"] = time.perf_counter() - tic
    tic = time.perf_counter()
    feats, rej = build_features(cloud, b, min_score, keep_rejected)
    model = _load_model(model_path)
    if model is not None:
        feats = _apply_model(feats, model)
        feats = feats[feats["score"] >= min_score].reset_index(drop=True)
    t["features"] = time.perf_counter() - tic
    write_features(feats, Path(features_dir) / f"{stem}.tsv")
    if rej is not None:
        precise_mz(rej).to_csv(Path(features_dir) / f"{stem}.rejected.tsv", sep="\t", index=False, float_format="%.6g")
    params = b.params.to_dict()
    params.update(n_scans=int(cloud.n_scans), n_centroids=int(cloud.n_points), polarity=cloud.polarity,
                  n_basins_initial=b.n_basins_initial, n_after_merge=b.n_after_merge,
                  n_candidates=int(len(b.basin_id)), n_features=int(len(feats)),
                  timings_s={k: round(float(v), 4) if isinstance(v, float) else v for k, v in t.items()},
                  peak3d_version=__version__)
    (Path(features_dir) / f"{stem}.params.json").write_text(json.dumps(params, indent=1))
    return dict(stem=stem, path=str(path), n_features=int(len(feats)), n_centroids=int(cloud.n_points),
                n_scans=int(cloud.n_scans), parse_s=round(t["parse"], 3), pick_s=round(t["pick"], 3),
                features_s=round(t["features"], 3))


def pick_files(files, output: Path, args, cache_dir: Path | None = None):
    output = Path(output)
    features_dir = output / "features"
    cache_dir = Path(cache_dir or getattr(args, "cache_dir", None) or output / ".cache")
    features_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = [Path(f) for f in files]
    stems = [f.stem for f in files]
    if len(set(stems)) != len(stems):
        raise ValueError("input file stems must be unique")
    for s in stems:
        if s in RESERVED:
            raise ValueError(f"file stem {s!r} collides with a reserved column name")
    n_workers = max(1, min(len(files), args.cores))
    threads = max(1, args.cores // n_workers)
    tic = time.perf_counter()
    _warm_jit()
    log(f"JIT warm-up {time.perf_counter() - tic:.1f} s; picking {len(files)} files with {n_workers} workers x {threads} threads")
    jobs = [(str(f), str(features_dir), str(cache_dir), args.min_score, args.keep_rejected, args.polarity, threads,
             str(args.model) if args.model else None) for f in files]
    tic = time.perf_counter()
    if n_workers == 1:
        results = [_pick_one(j) for j in jobs]
    else:
        with get_context("spawn").Pool(n_workers) as pool:
            results = pool.map(_pick_one, jobs, chunksize=1)
    man = pd.DataFrame(results)
    man.to_csv(features_dir / "manifest.tsv", sep="\t", index=False)
    log(f"picked {len(files)} files in {time.perf_counter() - tic:.1f} s: "
        f"{man['n_features'].sum()} features, median {man['pick_s'].median():.2f} s pick / {man['parse_s'].median():.2f} s parse per file")
    return results


# ------------------------------------------------------------------ align ----
def _read_features(features_dir: Path):
    feats = {}
    for f in sorted(Path(features_dir).glob("*.tsv")):
        if f.name == "manifest.tsv" or f.name.endswith(".rejected.tsv"):
            continue
        df = pd.read_csv(f, sep="\t")
        keep = list(COLUMNS)
        feats[f.stem] = df[keep] if all(c in df.columns for c in COLUMNS) else df
    return feats


def _scan_rt(cache_dir: Path, stem: str):
    with np.load(Path(cache_dir) / f"{stem}.npz", allow_pickle=False) as z:
        return z["rt"]


def _identity_result(feats, grids, dt_med, ppm):
    files = {s: FileAlignment(stem=s, warp=TableWarp.identity(grids[s]), role="none",
                              diag=FitDiag(flag="identity:disabled")) for s in feats}
    return AlignResult(files, None, True, ppm, 2 * dt_med, dt_med, float("nan"), 0, ["rt correction disabled"])


def _fill_one(job):
    stem, cache_dir, native, corr, g_mz, g_rt, g_half, ppm_tol, miss_idx, threads, noise_ref, h_ref, fwhm_scans = job
    import numba
    numba.set_num_threads(max(1, int(threads)))
    cloud = Cloud.load_npz(Path(cache_dir) / f"{stem}.npz")
    warp = TableWarp(native, corr)
    h, a, c = fill_file(cloud, warp, g_mz, g_rt, g_half, ppm_tol, noise_ref, h_ref, fwhm_scans)
    return stem, miss_idx, h, a, c


def align_and_group(output: Path, cache_dir: Path, args, stage_times: dict | None = None):
    output = Path(output)
    cache_dir = Path(cache_dir)
    features_dir = output / "features"
    stage_times = stage_times if stage_times is not None else {}
    feats = _read_features(features_dir)
    if not feats:
        raise FileNotFoundError(f"no feature tables in {features_dir}")
    stems = list(feats)
    grids = {s: _scan_rt(cache_dir, s) for s in stems}
    dts = [float(np.median(np.diff(g))) if len(g) > 1 else 0.01 for g in grids.values()]
    dt_med = float(np.median(dts))
    P = AlignParams(ppm=args.ppm, rt_tol_max=args.rt_tol_max, holdout=args.holdout)

    tic = time.perf_counter()
    if args.no_rt_correction:
        res = _identity_result(feats, grids, dt_med, args.ppm)
    else:
        res = align_run(feats, grids, P)
    stage_times["align_s"] = round(time.perf_counter() - tic, 3)
    for note in res.notes:
        log(note)
    log(f"alignment: medoid={res.medoid} iterations={res.n_iter_done} ppm_tol={res.ppm_tol:.2f} "
        f"rt_tol={60 * res.rt_tol:.2f} s pooled_mad={res.pooled_mad_s:.2f} s ({stage_times['align_s']} s)")

    # apply warps, write per-file correction tables
    rtc_dir = output / "rt_correction"
    rtc_dir.mkdir(exist_ok=True)
    for s in stems:
        fa = res.files[s]
        df = feats[s]
        df["rt_corr"] = fa.warp.forward(df["rt"].values)
        df["rt_min_corr"] = fa.warp.forward(df["rt_min"].values)
        df["rt_max_corr"] = fa.warp.forward(df["rt_max"].values)
        is_anchor = np.zeros(len(df), bool)
        is_anchor[fa.anchor_idx] = True
        df["is_anchor"] = is_anchor
        df["mz_corr"] = df["mz"].values * (1.0 - fa.mz_offset_ppm * 1e-6)
        pd.DataFrame({"scan": np.arange(len(fa.warp.native)), "rt_native": fa.warp.native, "rt_corr": fa.warp.corr}) \
            .to_csv(rtc_dir / f"{s}.tsv", sep="\t", index=False, float_format="%.6f")
    summary_table(res).to_csv(rtc_dir / "summary.tsv", sep="\t", index=False, float_format="%.4g")
    anchors_table(res).to_csv(rtc_dir / "anchors.tsv", sep="\t", index=False, float_format="%.6g")
    if args.holdout:
        rows = [dict(file=s, **fa.holdout) for s, fa in res.files.items() if fa.holdout]
        (output / "qc").mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(output / "qc" / "holdout.tsv", sep="\t", index=False, float_format="%.4g")

    # grouping on the corrected axis. Tolerances: m/z per pair from each feature's own
    # intensity-dependent sigma (its file's error model); RT = the larger of 3x the anchor residual
    # MAD and half the study's median peak width (apexes of one compound in two files cannot be
    # further apart than that if the peaks overlap), floored at 2 scans, capped at --rt-tol-max.
    tic = time.perf_counter()
    parts = []
    fwhms = []
    fwhm_scans = {}
    for k, s in enumerate(stems):
        df = feats[s]
        pj = features_dir / f"{s}.params.json"
        fp = json.loads(pj.read_text()) if pj.exists() else {}
        a_, b_ = fp.get("sig_a", 0.5), fp.get("sig_b", 0.0)
        fwhm_scans[s] = float(fp.get("fwhm_scans", 0.0))
        if np.isfinite(fp.get("fwhm_med", np.nan)):
            fwhms.append(fp["fwhm_med"])
        parts.append(pd.DataFrame({"mz": df["mz_corr"].values, "rt": df["rt_corr"].values, "rt_min": df["rt_min_corr"].values,
                                   "rt_max": df["rt_max_corr"].values, "height": df["height"].values,
                                   "area": df["area"].values, "noise": df["noise"].values, "file": k, "row": np.arange(len(df)),
                                   "iso_offset": df["iso_offset"].values.astype(np.int64),
                                   "charge": df["charge"].values.astype(np.int64),
                                   "sig": np.sqrt(a_ ** 2 + b_ ** 2 / np.maximum(df["height"].values, 1e-9))}))
    A = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    N = len(stems)
    fwhm_study = float(np.median(fwhms)) if fwhms else float("nan")
    rt_tol = res.rt_tol
    if np.isfinite(fwhm_study):
        rt_tol = float(np.clip(max(res.rt_tol, 0.5 * fwhm_study), 2 * dt_med, args.rt_tol_max))
    res.rt_tol = rt_tol
    offs = np.array([fa.mz_offset_ppm for fa in res.files.values()])
    log(f"grouping tolerances: rt >= {60 * rt_tol:.2f} s or half the narrower peak width (study FWHM {60 * fwhm_study:.2f} s), "
        f"m/z >= {res.ppm_tol:.1f} ppm per-pair sigma; per-file m/z offsets median {np.median(np.abs(offs)):.2f} ppm, max {np.abs(offs).max():.2f} ppm")
    gid, cons = group_features(A["mz"].values, A["rt"].values, A["rt_min"].values, A["rt_max"].values,
                               A["height"].values, A["file"].values, N, res.ppm_tol, rt_tol, sig=A["sig"].values)
    G0 = len(cons["mz"])
    if N > 1:
        # per-group sigma: that of its strongest member (the seed), via a height-weighted pick
        g_sig = np.zeros(G0)
        order_h = np.argsort(A["height"].values)           # ascending: the last write per group wins = strongest
        g_sig[gid[order_h]] = A["sig"].values[order_h]
        gid = merge_complementary(gid, cons, A["file"].values, N, res.ppm_tol, rt_tol, g_sig)
        cons = consensus(gid, A["mz"].values, A["rt"].values, A["rt_min"].values, A["rt_max"].values, A["height"].values)
    G = len(cons["mz"])
    log(f"complementary merge: {G0} -> {G} groups")
    A["gid"] = gid
    for k, s in enumerate(stems):
        sel = A["file"].values == k
        g = np.empty(len(feats[s]), np.int64)
        g[A["row"].values[sel]] = gid[sel]
        feats[s]["group_id"] = g
        write_features(feats[s], features_dir / f"{s}.tsv")
    stage_times["group_s"] = round(time.perf_counter() - tic, 3)
    log(f"grouped {len(A)} features from {N} files into {G} groups ({stage_times['group_s']} s)")

    # matrices
    H = np.zeros((G, N), np.float32)
    AR = np.zeros((G, N), np.float32)
    detected = np.zeros((G, N), bool)
    H[gid, A["file"].values] = A["height"].values
    AR[gid, A["file"].values] = A["area"].values
    detected[gid, A["file"].values] = True

    # gap filling, with confirmation: a missing cell whose smoothed EIC holds a real peak at the group's
    # position counts as presence of the group in that file
    n_filled = np.zeros(G, np.int64)
    confirmed = np.zeros((G, N), bool)
    if not args.no_gap_fill and N > 0:
        tic = time.perf_counter()
        half = np.maximum(res.rt_tol, 0.5 * (cons["rt_max"] - cons["rt_min"]))
        # per group: reference height (strongest member) and that member's cell noise
        order_h = np.argsort(A["height"].values)
        g_h = np.zeros(G)
        g_noise = np.zeros(G)
        g_h[gid[order_h]] = A["height"].values[order_h]
        g_noise[gid[order_h]] = A["noise"].values[order_h]
        n_workers = max(1, min(N, args.cores))
        threads = max(1, args.cores // n_workers)
        jobs = []
        for k, s in enumerate(stems):
            miss = np.flatnonzero(~detected[:, k])
            if len(miss) == 0:
                continue
            fa = res.files[s]
            jobs.append((s, str(cache_dir), fa.warp.native, fa.warp.corr, cons["mz"][miss], cons["rt"][miss],
                         half[miss], res.ppm_tol, miss, threads, g_noise[miss], g_h[miss], fwhm_scans.get(s, 0.0)))
        if n_workers == 1 or len(jobs) <= 1:
            out = [_fill_one(j) for j in jobs]
        else:
            with get_context("spawn").Pool(n_workers) as pool:
                out = pool.map(_fill_one, jobs, chunksize=1)
        idx = {s: k for k, s in enumerate(stems)}
        for s, miss, h, a, c in out:
            k = idx[s]
            H[miss, k] = h
            AR[miss, k] = a
            confirmed[miss, k] = c > 0
        n_filled = ((~detected) & (H > 0)).sum(axis=1)
        stage_times["gapfill_s"] = round(time.perf_counter() - tic, 3)
        log(f"gap-filled {int((~detected).sum())} cells in {len(jobs)} files, {int(confirmed.sum())} hold a confirmed peak "
            f"({stage_times['gapfill_s']} s)")
    n_confirmed = confirmed.sum(axis=1)
    # presence rule: a group must be detected or confirmed in >= min_presence files (capped at the file count)
    need = max(1, min(int(getattr(args, "min_presence", 2)), N))
    present = cons["n_detected"].astype(int) + n_confirmed
    keep_g = present >= need
    log(f"presence rule (>= {need} of {N} files, detected or confirmed): {int(keep_g.sum())} of {G} groups kept, "
        f"{int((~keep_g).sum())} dropped")
    for k, s in enumerate(stems):
        feats[s]["group_kept"] = keep_g[feats[s]["group_id"].values]
        write_features(feats[s], features_dir / f"{s}.tsv")

    # group-level isotope labels: the members' majority iso_offset (0 = monoisotopic or unknown) and charge
    def _mode(values, n_levels):
        counts = np.bincount(gid * n_levels + np.clip(values, 0, n_levels - 1), minlength=G * n_levels).reshape(G, n_levels)
        return counts.argmax(axis=1)
    g_iso = _mode(A["iso_offset"].values, 4)
    g_z = _mode(A["charge"].values, 4)
    # tables
    base = pd.DataFrame({"group_id": np.arange(G), "mz": cons["mz"], "rt": cons["rt"], "rt_min": cons["rt_min"],
                         "rt_max": cons["rt_max"], "n_detected": cons["n_detected"].astype(int),
                         "n_confirmed": n_confirmed.astype(int),
                         "n_filled": n_filled.astype(int), "mz_ppm_spread": cons["mz_ppm_spread"],
                         "rt_sd": cons["rt_sd"], "iso_offset": g_iso.astype(int), "charge": g_z.astype(int)})
    # presence of every group (before the rule) in every file: 2 detected, 1 confirmed peak, 0 absent
    pres = np.where(detected, 2, np.where(confirmed, 1, 0)).astype(np.int8)
    precise_mz(pd.concat([base[["group_id", "mz", "rt", "n_detected", "n_confirmed"]], pd.DataFrame(pres, columns=stems)], axis=1)) \
        .to_csv(output / "presence_mask.tsv", sep="\t", index=False)
    main_is_height = getattr(args, "quant", "height") == "height"
    main = pd.concat([base, pd.DataFrame(H if main_is_height else AR, columns=stems)], axis=1)[keep_g]
    other = pd.concat([base, pd.DataFrame(AR if main_is_height else H, columns=stems)], axis=1)[keep_g]
    precise_mz(main).to_csv(output / "aligned_feature_table.tsv", sep="\t", index=False, float_format="%.6g")
    precise_mz(other).to_csv(output / ("aligned_feature_area.tsv" if main_is_height else "aligned_feature_height.tsv"),
                             sep="\t", index=False, float_format="%.6g")
    pd.concat([base[["group_id"]], pd.DataFrame((~detected).astype(np.int8), columns=stems)], axis=1)[keep_g] \
        .to_csv(output / "filled_mask.tsv", sep="\t", index=False)
    stage_times["n_groups_all"] = int(G)
    stage_times["n_groups_kept"] = int(keep_g.sum())
    return res, cons, stage_times


def _write_params(output: Path, args, res: AlignResult | None, stage_times: dict, files, extra=None):
    import numba
    import scipy
    p = dict(peak3d_version=__version__, numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__,
             pandas=pd.__version__, command=" ".join(sys.argv), options={k: (str(v) if isinstance(v, Path) else v)
                                                                            for k, v in vars(args).items() if k != "cmd"},
             input_files=[str(f) for f in files], stage_times_s=stage_times)
    if res is not None:
        p.update(alignment=("skipped (single file)" if res.skipped and len(res.files) == 1 else
                            ("disabled" if args.no_rt_correction else "ok")),
                 medoid=res.medoid, n_iterations=res.n_iter_done, ppm_tol=res.ppm_tol, rt_tol_min=res.rt_tol,
                 pooled_anchor_mad_s=res.pooled_mad_s, notes=res.notes)
    if extra:
        p.update(extra)
    (output / "params.json").write_text(json.dumps(p, indent=1, default=str))


def process(files, args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output / ".cache"
    stage = {}
    tic = time.perf_counter()
    results = pick_files(files, output, args, cache_dir)
    stage["pick_s"] = round(time.perf_counter() - tic, 3)
    res, cons, stage = align_and_group(output, cache_dir, args, stage)
    tic = time.perf_counter()
    try:
        from . import qc
        qc.write_qc(output, res, cons)
    except Exception as e:  # QC plots never fail the run
        log(f"qc plots skipped: {e!r}")
    stage["qc_s"] = round(time.perf_counter() - tic, 3)
    _write_params(output, args, res, stage, files, extra=dict(n_files=len(files),
                                                                n_features_total=int(sum(r["n_features"] for r in results))))
    if not args.keep_cache and cache_dir == output / ".cache":
        shutil.rmtree(cache_dir, ignore_errors=True)


def align_existing(args):
    output = Path(args.output)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output / ".cache"
    if not cache_dir.exists():
        sys.exit(f"cache directory {cache_dir} not found (run process with --keep-cache)")
    stage = {}
    res, cons, stage = align_and_group(output, cache_dir, args, stage)
    try:
        from . import qc
        qc.write_qc(output, res, cons)
    except Exception as e:
        log(f"qc plots skipped: {e!r}")
    files = sorted(str(p) for p in (output / "features").glob("*.tsv") if not p.name.endswith("rejected.tsv")
                   and p.name != "manifest.tsv")
    _write_params(output, args, res, stage, files)
