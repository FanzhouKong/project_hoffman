# diagnostic: MassCube per-file feature detection on the MassCube-benchmark files under three settings, to separate
# a version change from our workflow. Writes every detected feature (m/z, RT, scan length) per file to
# results/diag_double/masscube_versions/<MODE>/<stem>.tsv.
#   A  MassCube 1.0.22 (the paper's version, run in its own environment): workflows.feature_detection, all ROIs
#   B  MassCube 1.2.22 process_single_file with the 1.0.22 segmentation settings (Gaussian sigma 1.2, prominence
#      ratio 0.1, distance 5) in place of 1.2.22's (0.6, 0.03, 3)
#   C  MassCube 1.2.22 process_single_file as installed (the per-file stage of our benchmark run)
#   Ae as A on the eight EIC files, also writing each feature's first and last RT (rt_lo, rt_hi: its peak edges)
# usage: masscube_version_runs.py MODE   (run with the interpreter of the matching MassCube version)
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.runs import MC_EXP, RUNS  # noqa: E402

mode = sys.argv[1]
out = ROOT / f"results/diag_double/masscube_versions/{mode}"
out.mkdir(parents=True, exist_ok=True)
files = [RUNS["MC_SYNTH"]["files"][0]] * (mode != "Ae") + [RUNS[f"MC_{s}"]["files"][0] for s in MC_EXP]
cols = ["mz", "rt", "length"] + ["rt_lo", "rt_hi"] * (mode == "Ae")

if mode in ("A", "Ae"):
    import masscube
    from masscube.workflows import feature_detection

    def detect(path):
        d = feature_detection(path, anno_isotope=False, anno_adduct=False, anno_in_source_fragment=False)
        if mode == "Ae":
            print("ROI attributes:", [a for a in vars(d.rois[0])], flush=True)
            return [(r.mz, r.rt, r.length, float(r.rt_seq[0]), float(r.rt_seq[-1])) for r in d.rois]
        return [(r.mz, r.rt, r.length) for r in d.rois]
else:
    import masscube
    import masscube.feature_detection as fd
    from masscube.workflows import process_single_file
    if mode == "B":
        fd.gaussian_filter_and_prominence_method.__defaults__ = (1.2, 0.1, 5)

    def detect(path):
        d = process_single_file(path, segment_feature=True, group_features=False, annotate_ms2=False, return_data=True)
        return [(f.mz, f.rt, f.length) for f in d.features]

print(f"mode {mode}: masscube {masscube.__version__ if hasattr(masscube, '__version__') else '?'} from {masscube.__file__}", flush=True)
for path in files:
    tic = time.perf_counter()
    rows = detect(path)
    stem = Path(path).stem
    pd.DataFrame(rows, columns=cols).to_csv(out / f"{stem}.tsv", sep="\t", index=False)
    print(f"{stem}: {len(rows)} features, {time.perf_counter() - tic:.0f} s", flush=True)
