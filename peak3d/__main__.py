"""Command line: peak3d {process,pick,align,demo,report,gui} ...   (or python -m peak3d ...)

process: pick every mzML in --input, align the runs on their own credible features, group them
         into one table, gap-fill, write QC and report.html -> --output
pick:    per-file feature tables only
align:   re-run alignment/grouping/gap-fill from an existing output (features/ + .cache/)
demo:    write three small synthetic mzML runs and process them (checks an installation)
report:  (re)write report.html for an existing output
gui:     local web app: choose folders, run, and browse results in the browser
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from . import __version__


def _positive(kind):
    def conv(v):
        x = kind(v)
        if x <= 0:
            raise argparse.ArgumentTypeError(f"must be > 0, got {v}")
        return x
    return conv


def _unit(v):
    x = float(v)
    if not 0.0 <= x <= 1.0:
        raise argparse.ArgumentTypeError(f"must be in [0, 1], got {v}")
    return x


def build_parser():
    ap = argparse.ArgumentParser(prog="peak3d", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"peak3d {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--cores", type=_positive(int), default=1, help="worker processes (default 1)")
        p.add_argument("--min-score", type=_unit, default=0.5, help="feature quality threshold (default 0.5)")
        p.add_argument("--polarity", choices=["pos", "neg"], default=None,
                       help="keep only this polarity (required for polarity-switching files)")
        p.add_argument("--model", type=Path, default=None, help="pickled re-scorer (optional)")
        p.add_argument("--keep-rejected", action="store_true", help="also write rejected candidates")

    def align_opts(p):
        p.add_argument("--ppm", type=_positive(float), default=5.0, help="m/z tolerance for anchors/grouping")
        p.add_argument("--rt-tol-max", type=_positive(float), default=0.25, help="cap on the grouping RT tolerance (min)")
        p.add_argument("--no-rt-correction", action="store_true", help="group on native RT (identity warps)")
        p.add_argument("--no-gap-fill", action="store_true")
        p.add_argument("--min-presence", type=_positive(int), default=2,
                       help="a group must be detected, or confirmed by a peak in its gap-filled chromatogram, in at least "
                            "this many files (capped at the number of files; 1 = off; default 2)")
        p.add_argument("--holdout", action="store_true", help="held-out anchor check per file")
        p.add_argument("--quant", choices=["height", "area"], default="height",
                       help="value in the main aligned table (default height; the other is written too)")

    p = sub.add_parser("process", help="pick + align + group + gap-fill")
    p.add_argument("--input", type=Path, required=True, help="directory of .mzML files")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=None, help="where per-file point clouds are cached (default <output>/.cache)")
    p.add_argument("--keep-cache", action="store_true")
    p.add_argument("--no-report", action="store_true", help="do not write report.html")
    common(p)
    align_opts(p)

    p = sub.add_parser("pick", help="per-file feature tables only")
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=None)
    common(p)

    p = sub.add_parser("align", help="re-run alignment/grouping/gap-fill from an existing output")
    p.add_argument("--output", type=Path, required=True, help="existing output with features/ and .cache/")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--cores", type=_positive(int), default=1)
    p.add_argument("--no-report", action="store_true", help="do not write report.html")
    align_opts(p)

    p = sub.add_parser("demo", help="process three synthetic runs (checks an installation)")
    p.add_argument("--output", type=Path, required=True, help="writes OUT/demo_mzml/ and OUT/peak3d_out/")
    p.add_argument("--cores", type=_positive(int), default=1)

    p = sub.add_parser("report", help="(re)write report.html for an existing output")
    p.add_argument("output", type=Path, help="a peak3d output folder")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="point-cloud cache for chromatograms (default <output>/.cache, if it still exists)")
    p.add_argument("--path", type=Path, default=None, help="write here instead of <output>/report.html")
    p.add_argument("--eic-budget", type=_positive(int), default=None,
                   help="chromatogram values to store (default 3,000,000; more = larger file, more features)")
    p.add_argument("--eic-files", type=_positive(int), default=None, help="files per chromatogram plot (default 16)")

    p = sub.add_parser("gui", help="local web app: run peak3d and browse results in the browser")
    p.add_argument("--host", default="127.0.0.1", help="interface to listen on (default 127.0.0.1: this computer only)")
    p.add_argument("--port", type=int, default=8765, help="port (default 8765; the next free one is used if taken)")
    p.add_argument("--root", type=Path, default=None, help="limit folder browsing, inputs and outputs to this folder")
    p.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    p.add_argument("--token", default=os.environ.get("PEAK3D_TOKEN") or "auto",
                   help="access token (default: random per start; env PEAK3D_TOKEN)")
    p.add_argument("--no-token", action="store_true", help="no access token (only on a machine you alone use)")
    p.add_argument("--allow-host", action="append", default=[], metavar="NAME",
                   help="extra Host name to accept, e.g. a reverse proxy's (repeatable)")
    return ap


def write_demo(out: Path) -> Path:
    """Three synthetic centroid runs: 60 compounds with an M+1 isotope, noise, and RT drift between runs."""
    import numpy as np

    from .synth import Peak, make_cloud, write_mzml
    data = out / "demo_mzml"
    data.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    base = list(zip(rng.uniform(150, 900, 60), rng.uniform(0.6, 3.4, 60), 10 ** rng.uniform(4.5, 6.5, 60)))
    for k, shift in enumerate((0.0, 0.05, -0.04)):
        peaks = [Peak(float(m), float(r + shift + 0.01 * r * k), float(h), 0.05, iso=(0.15,)) for m, r, h in base]
        cloud, _ = make_cloud(peaks, n_scans=800, dt=0.005, thresh=200.0, floor=100.0, n_noise=20000, seed=k,
                              name=f"demo{k + 1}")
        write_mzml(cloud, data / f"demo{k + 1}.mzML")
    return data


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "gui":
        from .web.server import serve
        serve(args.host, args.port, args.root, None if args.no_token else args.token, not args.no_browser,
              args.allow_host)
        return
    if args.cmd == "report":
        from .web.report import write_report
        limits = {k: v for k, v in (("eic_budget", args.eic_budget), ("eic_files", args.eic_files)) if v}
        path = write_report(args.output, args.cache_dir, args.path, **limits)
        print(f"[peak3d] report: {path} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)
        return
    os.environ.setdefault("NUMBA_NUM_THREADS", str(max(1, args.cores)))
    from . import pipeline
    t0 = time.perf_counter()
    if args.cmd == "demo":
        data = write_demo(args.output)
        print(f"[peak3d] demo: 3 synthetic runs in {data}", flush=True)
        args = parser.parse_args(["process", "--input", str(data), "--output", str(args.output / "peak3d_out"),
                                  "--cores", str(args.cores)])
    if args.cmd == "process":
        if not args.input.is_dir():
            sys.exit(f"--input must be a directory of .mzML files: {args.input}")
        files = sorted(p for p in args.input.iterdir() if p.suffix.lower() == ".mzml")
        if not files:
            sys.exit(f"no .mzML files in {args.input}")
        pipeline.process(files, args)
        print(f"[peak3d] {len(files)} files -> {args.output / 'aligned_feature_table.tsv'}", flush=True)
    elif args.cmd == "pick":
        pipeline.pick_files(args.files, args.output, args)
    elif args.cmd == "align":
        pipeline.align_existing(args)
    print(f"[peak3d] done in {time.perf_counter() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
