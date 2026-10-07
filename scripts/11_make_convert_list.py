#!/usr/bin/env python3
"""Build manifest/convert_list.tsv: one row per vendor file to convert to centroided mzML.

kinds
  thermo_zip   Thermo .raw stored inside a zip archive (BM21, HZV029)
  thermo_file  plain Thermo .raw (MTBLS733)
  agilent_dzip Agilent .d folder zipped on its own (MTBLS1684)

Row N (1-based) is processed by array task N of 12_convert.sbatch.
"""
import csv
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
rows = []

for ds, archive in [("BM21", "BM21/ST002454_BloodyMary21.zip"),
                    ("HZV029", "HZV029/ST002233_HZV029_HILICpos.zip")]:
    with zipfile.ZipFile(RAW / archive) as z:
        for m in sorted(z.namelist()):
            if m.startswith("__MACOSX") or not m.lower().endswith(".raw"):
                continue
            rows.append((ds, "thermo_zip", archive, m, Path(m).stem))

def complete(p):  # aria2c leaves a .aria2 control file next to partial downloads
    return not p.with_name(p.name + ".aria2").exists()


for p in sorted(q for q in (RAW / "LI2018_QE/FILES").glob("*.raw") if complete(q)):
    rows.append(("LI2018_QE", "thermo_file", str(p.relative_to(RAW)), "-", p.stem))

for p in sorted(q for q in (RAW / "MTBLS1684/FILES").glob("*.d.zip") if complete(q)):
    rows.append(("MTBLS1684", "agilent_dzip", str(p.relative_to(RAW)), "-",
                 p.name.removesuffix(".d.zip")))

out = ROOT / "manifest/convert_list.tsv"
with out.open("w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t", lineterminator="\n")
    w.writerow(["task", "dataset", "kind", "source", "member", "out_stem"])
    for i, r in enumerate(rows, 1):
        w.writerow([i, *r])

by = {}
for r in rows:
    by[r[0]] = by.get(r[0], 0) + 1
print(by, "total", len(rows), "->", out)
