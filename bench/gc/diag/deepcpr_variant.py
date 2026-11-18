"""peak3d gc on the DeepCPR benchmark picker outputs under other DeconvParams, each in its own folder of symlinks,
then the paper-style table (bench/gc/deepcpr_paper_table.py) and the component counts.

    python bench/gc/diag/deepcpr_variant.py TAG [name=value ...] [--src mosa_dsk] [--sets aa,fa]"""
import contextlib
import io
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from peak3d.gc import pipeline as G                      # noqa: E402
from peak3d.gc.deconv import DeconvParams                # noqa: E402
from bench.gc import deepcpr_paper_table as T            # noqa: E402

R = Path("/quobyte/metabolomicsgrp/fanzhou/hoffmann/results/gc_deepcpr")
RAW = Path("/quobyte/metabolomicsgrp/fanzhou/hoffmann/data/raw/DEEPCPR_GC/data")
LEVELS = {"aa": "LMJ-05=5,LMJ-10=10,LMJ-20=20,LMJ-30=30,LMJ-50=50",
          "fa": "zhi50-5vs1=0.2,zhi40-5vs1=0.4,zhi30-5vs1=0.6,zhi20-5vs1=0.8,zhi10-5vs1=1.0"}


def main():
    args = sys.argv[1:]
    src, sets = "mosa_dsk", ["aa", "fa"]
    if "--src" in args:
        src = args.pop(args.index("--src") + 1); args.remove("--src")
    if "--sets" in args:
        sets = args.pop(args.index("--sets") + 1).split(","); args.remove("--sets")
    tag, kv = args[0], {}
    types = {k: type(v) for k, v in DeconvParams().__dict__.items()}
    for item in args[1:]:
        k, v = item.split("=")
        kv[k] = (v.lower() in ("1", "true")) if types[k] is bool else types[k](v)
    for s in sets:
        out = R / "variants" / tag / s
        out.mkdir(parents=True, exist_ok=True)
        for p in (R / src / s).iterdir():
            if p.name not in ("gc", "DONE") and not (out / p.name).exists():
                (out / p.name).symlink_to(p)
        if (out / "gc").exists():
            shutil.rmtree(out / "gc")
        with contextlib.redirect_stdout(io.StringIO()):
            res = G.run(out, cores=5, deconv=DeconvParams(**kv))
        print(f"##### {tag} {s} {kv}: {res.get('n_components')} components in 5 files, deconvolution "
              f"{res.get('stage_times_s', {}).get('deconvolution_s')} s", flush=True)
        sys.argv = ["x", "--raw", str(RAW / ("amino acids" if s == "aa" else "fatty acids")), "--levels", LEVELS[s],
                    "--deepcpr", str(R / "deepcpr" / s), "--mosa", str(out), "--out", str(R / "variants" / tag / f"paper_table_{s}.tsv")]
        T.main()


if __name__ == "__main__":
    main()
