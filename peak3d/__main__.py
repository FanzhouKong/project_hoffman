"""Command line: peak3d {process,pick,align,demo} ...   (or python -m peak3d ...)

process: pick every mzML in --input, align the runs on their own credible features, group them
         into one table, gap-fill, write QC -> --output
pick:    per-file feature tables only
align:   re-run alignment/grouping/gap-fill from an existing output (features/ + .cache/)
demo:    write three small synthetic mzML runs and process them (checks an installation)
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
    align_opts(p)

    p = sub.add_parser("demo", help="process three synthetic runs (checks an installation)")
    p.add_argument("--output", type=Path, required=True, help="writes OUT/demo_mzml/ and OUT/peak3d_out/")
    p.add_argument("--cores", type=_positive(int), default=1)
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
