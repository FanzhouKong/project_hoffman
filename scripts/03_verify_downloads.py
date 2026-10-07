#!/usr/bin/env python3
"""Compare data/raw against manifest/manifest.tsv; write manifest/download_status.tsv.

Exact byte sizes are checked where the repository reports them (GNPS, Zenodo,
Workbench). MetaboLights listings give rounded sizes (e.g. 703M), so those are
checked to within 1 MB per 100 MB.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((ROOT / "manifest/manifest.tsv").open(), delimiter="\t", lineterminator="\n"))
out, bad = [], 0
for r in rows:
    p = ROOT / "data/raw" / r["dataset"] / r["relpath"]
    exp = int(r["size_bytes"] or 0)
    partial = p.with_name(p.name + ".aria2").exists()
    if not p.exists():
        status = "missing"
    elif partial:
        status = "partial"
    else:
        got = p.stat().st_size
        if exp == 0:
            status = "ok_unsized"
        elif r["source"] == "metabolights":
            status = "ok" if abs(got - exp) <= max(1.1e6, exp * 0.01) else f"size_mismatch:{got}"
        else:
            status = "ok" if got == exp else f"size_mismatch:{got}"
    bad += not status.startswith("ok")
    out.append(dict(dataset=r["dataset"], relpath=r["relpath"], status=status))

with (ROOT / "manifest/download_status.tsv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["dataset", "relpath", "status"], delimiter="\t", lineterminator="\n")
    w.writeheader()
    w.writerows(out)

summary = {}
for o in out:
    k = (o["dataset"], o["status"].split(":")[0])
    summary[k] = summary.get(k, 0) + 1
for (d, s), n in sorted(summary.items()):
    print(f"{d}\t{s}\t{n}")
print(f"{len(out) - bad}/{len(out)} files OK")
