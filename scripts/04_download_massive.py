#!/usr/bin/env python3
"""Sequential, rate-limit-aware downloader for MassIVE (gnps) rows of the manifest.

MassIVE answers bursts with HTTP 429, which aria2c treats as fatal, so these files
are fetched one at a time with a pause between files and exponential backoff on
429/5xx. Files whose size already matches the manifest are skipped.
usage: 04_download_massive.py [DATASET ...]
"""
import csv, sys, time, urllib.error, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
want = set(sys.argv[1:])
rows = [r for r in csv.DictReader((ROOT / "manifest/manifest.tsv").open(), delimiter="\t")
        if r["source"] == "gnps" and (not want or r["dataset"] in want)]
UA = {"User-Agent": "peakbench/0.1 (UC Davis; sequential download)"}
done = fail = 0
for i, r in enumerate(rows, 1):
    dest = ROOT / "data/raw" / r["dataset"] / r["relpath"]
    size = int(r["size_bytes"] or 0)
    if dest.exists() and (size == 0 or dest.stat().st_size == size):
        continue
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    wait = 30
    for attempt in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request(r["url"], headers=UA), timeout=300) as resp, \
                 part.open("wb") as fh:
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
            if size and part.stat().st_size != size:
                raise IOError(f"size {part.stat().st_size} != {size}")
            part.rename(dest)
            done += 1
            break
        except (urllib.error.HTTPError, urllib.error.URLError, IOError, TimeoutError) as e:
            code = getattr(e, "code", None)
            print(f"[{i}/{len(rows)}] {r['relpath']}: {e}; retry in {wait}s", flush=True)
            time.sleep(wait)
            wait = min(wait * 2, 900)
    else:
        fail += 1
        print(f"FAILED {r['dataset']}/{r['relpath']}", flush=True)
    if i % 50 == 0:
        print(f"[{i}/{len(rows)}] downloaded {done}, failed {fail}", flush=True)
    time.sleep(1)
print(f"finished: downloaded {done}, failed {fail}")
sys.exit(1 if fail else 0)
