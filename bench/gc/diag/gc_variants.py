"""peak3d gc + mrh_score on a picker output under several DeconvParams, each in its own folder of symlinks"""
import os, sys, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from peak3d.gc import pipeline as G
from peak3d.gc.deconv import DeconvParams
from bench.gc import mrh_score as M


def main():
    R = Path("/quobyte/metabolomicsgrp/fanzhou/hoffmann/results/gc_mrh")
    tag = sys.argv[1]
    variants = {"v2": dict(delay=False), "v2_min3": dict(min_features=3, delay=False), "v3": {}}
    want = sys.argv[2:] or list(variants)
    for mode, src in (("lc", "peak3d"), ("gc", "peak3d_gc")):
        for name in want:
            out = R / "variants" / f"{src}_{tag}_{name}"
            out.mkdir(parents=True, exist_ok=True)
            for p in (R / src).iterdir():
                if p.name not in ("gc", "DONE") and not (out / p.name).exists():
                    (out / p.name).symlink_to(p)
            if (out / "gc").exists():
                shutil.rmtree(out / "gc")
            res = G.run(out, cores=11, deconv=DeconvParams(**variants[name]))
            print(f"##### {mode} {name}: gc done", {k: v for k, v in res.items() if isinstance(v, (int, float, str))}, flush=True)
            M.main(["--output", str(out), "--truth", "/quobyte/metabolomicsgrp/fanzhou/hoffmann/data/raw/MRH_GC_ORBITRAP", "--out", str(R / "variants" / f"score_{src}_{tag}_{name}")])


if __name__ == "__main__":     # the GC stage starts worker processes that import this file
    main()
