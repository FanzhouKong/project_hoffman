#!/usr/bin/env python3
"""List every file of every registered dataset and write manifest/manifest.tsv
plus an aria2c input file. Only calls repository listing APIs (light enough for
a login node); the actual download runs as a Slurm job (02_download.sbatch).

usage: python3 scripts/01_build_manifest.py [--only ID[,ID...]]
"""
import argparse, csv, html, json, re, sys, time
import urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.datasets import DATASETS  # noqa: E402

UA = {"User-Agent": "peakbench-manifest/0.1 (UC Davis metabolomics)"}
COLS = ["dataset", "priority", "source", "accession", "relpath", "url",
        "size_bytes", "md5"]


def get(url, retries=4):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            if i == retries - 1:
                raise
            print(f"  retry {url}: {e}", file=sys.stderr)
            time.sleep(5 * (i + 1))


def head_size(url):
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(r.headers.get("Content-Length") or 0)
    except Exception:  # noqa: BLE001
        return 0


HUMAN = {"K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}  # Apache listings are 1024-based


def parse_human(s):
    s = s.strip()
    if not s or s == "-":
        return 0
    if s[-1] in HUMAN:
        return int(float(s[:-1]) * HUMAN[s[-1]])
    return int(float(s))


# --- sources ---------------------------------------------------------------

def list_metabolights(ds):
    base = f"https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/{ds['accession']}/"
    rows = []

    def walk(url, rel, depth):
        page = get(url)
        for href, size in re.findall(
                r'<a href="([^"?/][^"]*)">[^<]*</a>\s*</td><td[^>]*>[^<]*</td><td[^>]*>\s*([^<]*)</td>',
                page):
            href = html.unescape(href)
            name = urllib.parse.unquote(href)
            if href.endswith("/"):
                # Agilent .d / Bruker .d folders and FILES/ subfolders
                if depth < 6 and name not in ("HASHES/", "METADATA_REVISIONS/"):
                    walk(url + href, rel + name, depth + 1)
            else:
                rows.append((rel + name, url + href, parse_human(size), ""))

    walk(base, "", 0)
    return rows


def list_mwb(ds):
    acc = ds["accession"]
    page = get("https://www.metabolomicsworkbench.org/data/DRCCStudySummary.php"
               f"?Mode=SetupRawDataDownload&StudyID={acc}")
    rows = []
    for p in sorted(set(re.findall(r'(/studydownload/[^"\']+)', page))):
        url = "https://www.metabolomicsworkbench.org" + p
        rows.append((Path(p).name, url, head_size(url), ""))
    # machine-readable study metadata
    for kind in ("summary", "analysis", "factors"):
        rows.append((f"metadata/{acc}_{kind}.json",
                     f"https://www.metabolomicsworkbench.org/rest/study/study_id/{acc}/{kind}",
                     0, ""))
    return rows


def list_gnps(ds):
    acc = ds["accession"]
    sql = f"select filepath,size from filename where dataset='{acc}'"
    rows, offset = [], 0
    while True:
        q = urllib.parse.quote(f"{sql} order by filepath limit 1000 offset {offset}")
        batch = json.loads(get(
            f"https://datasetcache.gnps2.org/datasette/database.json?sql={q}&_shape=array"))
        if not batch:
            break
        for r in batch:
            fp = r["filepath"]
            inc = ds.get("include_prefix")
            if inc and not any(fp.startswith(p) for p in inc):
                continue
            url = ("https://massive.ucsd.edu/ProteoSAFe/DownloadResultFile?file=f."
                   + urllib.parse.quote(f"{acc}/{fp}") + "&forceDownload=true")
            rows.append((fp, url, int(r.get("size") or 0), ""))
        offset += len(batch)
    return rows


def list_zenodo(ds):
    rec = json.loads(get(f"https://zenodo.org/api/records/{ds['accession']}"))
    rows = []
    for f in rec["files"]:
        if f["key"] in ds.get("exclude_names", []):
            continue
        md5 = f.get("checksum", "").removeprefix("md5:")
        rows.append((f["key"], f["links"]["self"], int(f["size"]), md5))
    return rows


def list_github(ds):
    repo = ds["accession"]
    meta = json.loads(get(f"https://api.github.com/repos/{repo}"))
    br = meta["default_branch"]
    url = f"https://github.com/{meta['full_name']}/archive/refs/heads/{br}.zip"
    return [(f"{Path(repo).name}-{br}.zip", url, 0, "")]


LISTERS = dict(metabolights=list_metabolights, mwb=list_mwb, gnps=list_gnps,
               zenodo=list_zenodo, github=list_github)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated dataset ids")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    out_dir = ROOT / "manifest"
    out_dir.mkdir(exist_ok=True)
    tsv = out_dir / "manifest.tsv"
    old = []
    if only and tsv.exists():  # keep other datasets' rows on a partial rebuild
        with tsv.open() as fh:
            old = [r for r in csv.DictReader(fh, delimiter="\t", lineterminator="\n")
                   if r["dataset"] not in only]

    rows = list(old)
    for ds in sorted(DATASETS, key=lambda d: d["priority"]):
        if only and ds["id"] not in only:
            continue
        print(f"[{ds['id']}] listing {ds['source']}:{ds['accession']}", file=sys.stderr)
        files = LISTERS[ds["source"]](ds)
        for rel, url, size, md5 in files:
            rows.append(dict(dataset=ds["id"], priority=ds["priority"],
                             source=ds["source"], accession=ds["accession"],
                             relpath=rel, url=url, size_bytes=size, md5=md5))
        gb = sum(int(f[2]) for f in files) / 1e9
        print(f"  {len(files)} files, {gb:.1f} GB", file=sys.stderr)

    rows.sort(key=lambda r: (int(r["priority"]), r["dataset"], r["relpath"]))
    with tsv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    # aria2c input file: one URL per entry, indented per-download options
    with (out_dir / "aria2_input.txt").open("w") as fh:
        for r in rows:
            dest = Path("data/raw") / r["dataset"] / r["relpath"]
            fh.write(f"{r['url']}\n  dir={dest.parent}\n  out={dest.name}\n")
            if r["md5"]:
                fh.write(f"  checksum=md5={r['md5']}\n")

    by = {}
    for r in rows:
        n, b = by.get(r["dataset"], (0, 0))
        by[r["dataset"]] = (n + 1, b + int(r["size_bytes"] or 0))
    print("\ndataset\tfiles\tGB")
    for k, (n, b) in by.items():
        print(f"{k}\t{n}\t{b / 1e9:.1f}")
    print(f"TOTAL\t{sum(v[0] for v in by.values())}\t"
          f"{sum(v[1] for v in by.values()) / 1e9:.1f}")


if __name__ == "__main__":
    main()
