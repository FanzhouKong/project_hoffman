# blind visual-check sheets for the IDSL003 labels, drawn from the review page's data files (label_review/page_data:
# 003 at 10 / 30 ppm and the 30 random injections). Each panel shows only a code: blue 003 at 10 ppm (dots, label
# scan = red dot), grey 003 at 30 ppm, orange = median (line) and 25-75 % band of the 30 injections at 10 ppm
# (drift-corrected), red line = label RT,
# shaded band = the +/-0.1 min scoring window. The y axis is scaled to the blue trace; grey / orange may run off the top.
# usage: idsl_visual_sheets.py PAGE_DATA OUT_DIR PREFIX [--ids FILE] [--per 36] [--cols 6] [--half 0.4] [--w 2.8 --h 2.3]
#   writes OUT_DIR/sheets/PREFIX_###.png and OUT_DIR/PREFIX_key.tsv (code -> label index); run in srun (CPU-bound)
import sys, os, json, gzip, base64, argparse
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("page_data"); ap.add_argument("out"); ap.add_argument("prefix")
ap.add_argument("--ids", help="file with label indices (0-based, one per line) to draw; default all")
ap.add_argument("--per", type=int, default=36); ap.add_argument("--cols", type=int, default=6)
ap.add_argument("--half", type=float, default=0.4); ap.add_argument("--w", type=float, default=2.8)
ap.add_argument("--h", type=float, default=2.3); ap.add_argument("--seed", type=int, default=5)
A = ap.parse_args()
P, OUT = Path(A.page_data), Path(A.out)
(OUT / "sheets").mkdir(parents=True, exist_ok=True)


def load(name):
    b = (P / name).read_bytes()
    if name.endswith(".txt"):
        b = base64.b64decode(b)
    return gzip.decompress(b) if b[:2] == b"\x1f\x8b" else b


META = json.load(open(P / "meta.json"))
ST = json.loads(load("stats.txt" if (P / "stats.txt").exists() else "stats.bin"))
NF, NW, CH, PAD = len(META["files"]), META["nw"], META["chunk"], META["rtpad"]
NR = NF - 1
RT = np.frombuffer(load("rt.txt" if (P / "rt.txt").exists() else "rt.bin"), "<f4").reshape(NF, PAD)
L16, L8 = META["l16"], META["l8"]
_cache = {}


def chunk(c):
    if c not in _cache:
        name = f"c{c:03d}.txt" if (P / f"c{c:03d}.txt").exists() else f"c{c:03d}.bin"
        b = load(name)
        n = min(CH, META["n"] - c * CH)
        o = 0
        st = np.frombuffer(b, "<u2", n * NF, o).reshape(n, NF); o += n * NF * 2
        sh = np.frombuffer(b, "<i2", n * NR, o).reshape(n, NR); o += n * NR * 2
        t10 = np.frombuffer(b, "<u2", n * NW, o).reshape(n, NW); o += n * NW * 2
        t30 = np.frombuffer(b, "<u2", n * NW, o).reshape(n, NW); o += n * NW * 2
        reps = np.frombuffer(b, "u1", n * NR * NW, o).reshape(n, NR, NW)
        _cache.clear()
        _cache[c] = (st, sh, t10, t30, reps)
    return _cache[c]


def traces(i):
    st, sh, t10, t30, reps = chunk(i // CH)
    k = i % CH
    x = RT[0, st[k, 0]:st[k, 0] + NW].astype(float)
    y10 = np.expm1(t10[k] / 65535 * L16); y30 = np.expm1(t30[k] / 65535 * L16)
    R = [(RT[r + 1, st[k, r + 1]:st[k, r + 1] + NW] - sh[k, r] / 60000, np.expm1(reps[k, r] / 255 * L8)) for r in range(NR)]
    return x, y10, y30, R


def sheet(args):
    s, idxs = args
    rows = int(np.ceil(len(idxs) / A.cols))
    fig, axes = plt.subplots(rows, A.cols, figsize=(A.w * A.cols, A.h * rows), dpi=100, squeeze=False)
    for p, (ax, i) in enumerate(zip(axes.ravel(), idxs)):
        x, y10, y30, R = traces(int(i))
        t0 = ST["rt"][i]
        v = (x >= t0 - A.half) & (x <= t0 + A.half)
        top = max(float(y10[v].max()) if v.any() else 0.0, 50.0) * 1.15
        ax.axvspan((t0 - 0.1) * 60, (t0 + 0.1) * 60, color="#1d5fc4", alpha=0.09, lw=0)
        Y = np.vstack([np.interp(x, rx, ry, left=np.nan, right=np.nan) for rx, ry in R])
        q25, q50, q75 = np.nanpercentile(Y, [25, 50, 75], axis=0)
        ax.fill_between(x * 60, q25, q75, color="#d08a1e", alpha=0.22, lw=0)
        ax.plot(x * 60, q50, color="#d08a1e", lw=1.1)
        ax.plot(x * 60, y30, color="#9aa3ad", lw=0.8)
        ax.plot(x * 60, y10, color="#1d5fc4", lw=1.0, marker=".", ms=2.2)
        j = int(np.argmin(np.abs(x - t0)))
        ax.axvline(t0 * 60, color="#c3303a", lw=0.6)
        ax.plot(x[j] * 60, y10[j], "o", color="#c3303a", ms=3.5)
        ax.set_xlim((t0 - A.half) * 60, (t0 + A.half) * 60); ax.set_ylim(0, top)
        ax.set_title(f"{A.prefix}{s:03d}-{p + 1:02d}", fontsize=8, loc="left", pad=2)
        ax.tick_params(labelsize=5, length=2, pad=1)
        ax.yaxis.get_offset_text().set_fontsize(5)
    for ax in axes.ravel()[len(idxs):]:
        ax.axis("off")
    fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.3)
    fig.savefig(OUT / "sheets" / f"{A.prefix}{s:03d}.png")
    plt.close(fig)
    return s


if __name__ == "__main__":
    idx = np.arange(META["n"]) if not A.ids else np.array([int(l) for l in open(A.ids) if l.strip()])
    rng = np.random.default_rng(A.seed)
    idx = idx[rng.permutation(len(idx))]
    groups = [(s + 1, idx[k:k + A.per]) for s, k in enumerate(range(0, len(idx), A.per))]
    with open(OUT / f"{A.prefix}key.tsv", "w") as fh:
        fh.write("code\tidx\tid\n")
        for s, g in groups:
            for p, i in enumerate(g):
                fh.write(f"{A.prefix}{s:03d}-{p + 1:02d}\t{i}\t{ST['id'][i]}\n")
    # sort each worker's sheets by chunk to reuse decoded chunks less randomly: simple per-sheet decode is fine
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))) as pool:
        for k, _ in enumerate(pool.imap_unordered(sheet, groups, chunksize=4)):
            if (k + 1) % 100 == 0:
                print(k + 1, "sheets", flush=True)
    print(f"{len(groups)} sheets of {A.per} -> {OUT / 'sheets'}; key {OUT / (A.prefix + 'key.tsv')}")
