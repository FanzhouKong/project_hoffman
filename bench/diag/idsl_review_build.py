# build the data files of the IDSL003 label review page: for every one of the 20,000 labels, the EIC window
# (+/-0.5 min) in the labelled file 003 (10 and 30 ppm) and in the 30 random injections of label_review/files.txt
# (10 ppm, drift-corrected start), plus the per-label statistics from idsl_label_audit.py / idsl_label_verdict.py
# run on those same 30 injections. Output: OUT/meta.json, OUT/rt.bin, OUT/stats.bin, OUT/c000.bin ... (gzip).
# usage: idsl_review_build.py REVIEW_DIR OUT_DIR   (reads 31 mzML -> srun)
import sys, os, re, json, gzip, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.msdata import load_run  # noqa: E402
from bench.diag.idsl_label_audit import eic_win  # noqa: E402

D, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
HALF, NW, CHUNK, RTPAD = 0.5, 99, 200, 1280
L16, L8 = float(np.log1p(1e9)), float(np.log1p(1e8))
F003 = ROOT / "data/raw/IDSL_IPA/003.mzML"
REPDIR = ROOT / "data/mzml/MTBLS1684"
A = pd.read_csv(D / "verdict.tsv", sep="\t")
INFO = {i["file"]: i for i in json.load(open(D / "replicates.json"))}
REPS = open(D / "files.txt").read().strip().split(",")
MZ, RT = A["m/z"].values.astype(float), A["RT(min)"].values.astype(float)


def stamp(path):
    with open(path, "rb") as fh:
        m = re.search(rb'startTimeStamp="([^"]+)"', fh.read(20000))
    return m.group(1).decode() if m else ""


def q16(x):
    return np.round(np.log1p(x) / L16 * 65535).clip(0, 65535).astype(np.uint16)


def q8(x):
    return np.round(np.log1p(x) / L8 * 255).clip(0, 255).astype(np.uint8)


def work(args):
    name, path = args
    t0 = time.time()
    run = load_run(path)
    n = len(MZ)
    starts = np.zeros(n, np.uint16)
    shift_ms = np.zeros(n, np.int16)
    is003 = name == "003z"
    t10 = np.zeros((n, NW), np.uint16 if is003 else np.uint8)
    t30 = np.zeros((n, NW), np.uint16) if is003 else None
    if is003:
        kx, ky = np.array([0.0, 14.0]), np.array([0.0, 0.0])
    else:
        kx = np.array(INFO[name]["drift_knots_min"])
        ky = np.array(INFO[name]["drift_knots_s"]) / 60
    for i in range(n):
        sh = float(np.interp(RT[i], kx, ky))
        s0 = int(np.searchsorted(run.rt, RT[i] + sh - HALF))
        s0 = min(max(s0, 0), run.n_scans - NW)
        starts[i] = s0
        shift_ms[i] = int(round(sh * 60000))
        e = eic_win(run, MZ[i], 10.0, s0, s0 + NW)
        t10[i] = q16(e) if is003 else q8(e)
        if is003:
            t30[i] = q16(eic_win(run, MZ[i], 30.0, s0, s0 + NW))
    rt = np.full(RTPAD, np.nan, np.float32)
    rt[:run.n_scans] = run.rt
    return dict(name=name, starts=starts, shift_ms=shift_ms, t10=t10, t30=t30, rt=rt, n_scans=run.n_scans,
                stamp=stamp(path), secs=round(time.time() - t0, 1))


jobs = [("003z", F003)] + [(s, REPDIR / f"{s}.mzML") for s in REPS]
with Pool(min(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")), len(jobs))) as pool:
    res = {r["name"]: r for r in pool.imap_unordered(work, jobs)}
z = res["003z"]
order = ["003z"] + REPS
print("extracted", {k: res[k]["secs"] for k in order}, flush=True)

# ---- rt.bin: Float32 [31 x RTPAD], row 0 = labelled 003, rows 1.. = REPS order
rt = np.stack([res[k]["rt"] for k in order])
open(OUT / "rt.bin", "wb").write(gzip.compress(rt.astype("<f4").tobytes(), 6))

# ---- chunks
n = len(MZ)
starts = np.stack([res[k]["starts"] for k in order], 1).astype("<u2")          # n x 31
shifts = np.stack([res[k]["shift_ms"] for k in REPS], 1).astype("<i2")         # n x 30
reps = np.stack([res[k]["t10"] for k in REPS], 1)                               # n x 30 x NW (uint8)
sizes = []
for c in range(0, n, CHUNK):
    sl = slice(c, min(c + CHUNK, n))
    blob = b"".join([starts[sl].tobytes(), shifts[sl].tobytes(), z["t10"][sl].astype("<u2").tobytes(),
                     z["t30"][sl].astype("<u2").tobytes(), np.ascontiguousarray(reps[sl]).tobytes()])
    gz = gzip.compress(blob, 9)
    open(OUT / f"c{c // CHUNK:03d}.bin", "wb").write(gz)
    sizes.append(len(gz))
print(f"chunks: {len(sizes)}, gz total {sum(sizes) / 1e6:.1f} MB, max {max(sizes) / 1e6:.2f} MB", flush=True)

# ---- presence bits per label and replicate (strict peak test of idsl_label_audit / idsl_label_verdict)
R = np.load(D / "replicates.npz")
rf = list(R["files"])
g = lambda k: R[f"tgt_{k}"]
ok = np.isfinite(g("pbr"))
strict = ok & (g("pbr") >= 3) & (g("n_pts") >= 4) & (g("r2") >= 0.6) & (g("flank") == 0) & (g("prom") >= 0.5)
lenient = ok & (g("pbr") >= 2) & (g("n_pts") >= 3) & (g("flank") == 0)
rows = [rf.index(s) for s in REPS]
bits = np.zeros(n, np.int64)
for j, r in enumerate(rows):
    bits |= strict[r].astype(np.int64) << j
rep_n = strict[rows].sum(0)
rep_l = lenient[rows].sum(0)
rep_m = ok[rows].sum(0)

# ---- stats (columnar JSON)
CLASSES = ["clear_peak", "peak", "flank", "mz_off", "no_local_peak", "nothing"]
VERDICTS = sorted(A.verdict.unique())
pub = ["IDSL.IPA", "XCMS", "MZMINE", "MSDIAL"]
pubmask = sum(A[c].isin(["TP", "FP"]).values.astype(int) << k for k, c in enumerate(pub))


def col(x, nd):
    x = np.asarray(x, float)
    return [None if not np.isfinite(v) else (int(round(v)) if nd == 0 else round(float(v), nd)) for v in x]


S = dict(
    id=A["id"].astype(int).tolist(), mz=[round(float(v), 5) for v in MZ], rt=[round(float(v), 2) for v in RT],
    lab=(A["Manual Curation"] == "TP").astype(int).tolist(), pub=[int(v) for v in pubmask],
    cls=[CLASSES.index(c) for c in A.cls003], h=col(A.h, 0), pbr=col(A.pbr, 1), npt=col(A.n_pts, 0),
    r2=col(A.r2, 2), prom=col(A.prom, 2), fwhm=col(A.fwhm, 0), flank=A.flank.astype(int).tolist(),
    drt=col(A.drt_s, 1), dppm=col(A.d_ppm, 1), mzsd=col(A.mz_sd_ppm, 1), dppmd=col(A.d_ppm_dom, 1),
    domr=col(A.dom_ratio, 1), cp1=col(A.corr_p1, 2), rp1=col(A.r_p1, 2), cm1=col(A.corr_m1, 2), rm1=col(A.r_m1, 2),
    repn=rep_n.astype(int).tolist(), repl=rep_l.astype(int).tolist(), repm=rep_m.astype(int).tolist(),
    decoy=col(A.decoy, 2), nbtp=A.nb_tp_10.astype(int).tolist(), nbtn=A.nb_tn_10.astype(int).tolist(),
    ver=[VERDICTS.index(v) for v in A.verdict], ptp=col(A.p_tp, 2), pres=[int(v) for v in bits])
open(OUT / "stats.bin", "wb").write(gzip.compress(json.dumps(S, separators=(",", ":")).encode(), 9))

files = [dict(name="003", role="labelled file (Zenodo 003.mzML)", acquired=z["stamp"], n_scans=int(z["n_scans"]))]
for s in REPS:
    i = INFO[s]
    files.append(dict(name=s, acquired=res[s]["stamp"], n_scans=int(res[s]["n_scans"]), n_anchor=i["n_anchor"],
                      shift_med_s=round(i["shift_med_s"], 2), resid_mad_s=round(i["resid_mad_s"], 2)))
meta = dict(n=n, chunk=CHUNK, nw=NW, half_min=HALF, rtpad=RTPAD, l16=L16, l8=L8, n_chunks=len(sizes),
            files=files, classes=CLASSES, verdicts=VERDICTS, tools=["IDSL.IPA", "XCMS", "MZmine", "MS-DIAL"],
            built=time.strftime("%Y-%m-%d"))
json.dump(meta, open(OUT / "meta.json", "w"), indent=1)
print("stats.bin", round((OUT / "stats.bin").stat().st_size / 1e6, 2), "MB; rt.bin",
      round((OUT / "rt.bin").stat().st_size / 1e6, 2), "MB; done")
