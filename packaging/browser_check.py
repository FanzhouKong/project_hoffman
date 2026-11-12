"""Use the Mosa app's downloads in a real browser (Playwright): open a results folder in the running GUI, then
Download table, Download candidates and the Download menu's feature table and report, and the report's own
Download candidates from the saved report.html; check what each saved file holds against the results folder.

    python packaging/browser_check.py --url "http://127.0.0.1:8799/?token=..." --results OUT --save DIR [--channel msedge]

Needs `pip install playwright` and a browser: --channel msedge / chrome use the installed one (the Windows build
check uses Edge), none the Playwright Chromium (`python -m playwright install chromium-headless-shell`). Exits 1
on the first failed check; screenshots and the downloads stay in --save."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


def fail(msg):
    print(f"FAILED: {msg}", flush=True)
    sys.exit(1)


def download(page, click, save: Path, what: str) -> Path:
    with page.expect_download(timeout=60_000) as d:
        click()
    dl = d.value
    if dl.failure():
        fail(f"{what}: the browser reports {dl.failure()}")
    path = save / dl.suggested_filename
    dl.save_as(path)
    print(f"{what}: {dl.suggested_filename} ({path.stat().st_size} bytes)", flush=True)
    return path


# a MassWiki search record (the data of a row of its job database): key order and value types, as profiled from a
# whole job (5543 records); a set allows either type (entropy is the int 0 for a one-peak spectrum, the RT model's
# confidences are null where it gives none)
S, I, F, B, N, L = "str", "int", "float", "bool", "null", "list"
CAND = {"db": S, "id": S, "name": S, "adduct": S, "rt": S, "precursor_mz": S, "smiles": S, "user_name": S, "spec_uid": S,
        "predicted_rt_hilic": S, "predicted_rt_rp": S, "rt_confidence_level_hilic": S, "rt_confidence_level_rp": S,
        "anno_delta_rt": S, "splash": S, "index": I, "entropy_similarity": F, "library_wiki_id": S,
        "delta_predicted_rt": N, "rt_confidence_level": N, "confidence_in_domain": B, "confidence_pct": {F, N},
        "ensemble_sd_pct": {F, N}}
SEARCHES = {"identity_search": [CAND], "open_search": L, "neutral_loss_search": L, "hybrid_search": L}
RECORD = {"analysis": {"reference_library": SEARCHES,
                       "experimental_library": {"similar_spectra": L, "cluster_peaks": L},
                       "annotation_library": {k: L for k in SEARCHES}, "internal_library": {"rt_similar": L, "ms2_similar": L}},
          "spectrum": {"id": S, "name": S, "searchmode": S, "formula": S, "inchikey": S, "smiles": S, "precursor_type": S,
                       "charge": I, "scan": I, "precursor_mz": F, "peaks_clean": L, "peaks_source": S, "entropy": {I, F},
                       "peaks": L, "rt_normalized": F, "rt": F, "spec_uid": I},
          "wiki_id": S, "status": S}
TYPE = {str: S, int: I, float: F, bool: B, type(None): N, list: L}


def shape_errors(obj, want, path="") -> list[str]:
    """where obj departs from the template want: dict keys and their order, leaf types; a [template] list checks its
    items"""
    if isinstance(want, dict):
        if not isinstance(obj, dict):
            return [f"{path}: {TYPE.get(type(obj), type(obj).__name__)}, not an object"]
        if list(obj) != list(want):
            return [f"{path}: keys {list(obj)} != {list(want)}"]
        return [e for k in want for e in shape_errors(obj[k], want[k], f"{path}.{k}")]
    if isinstance(want, list):
        if not isinstance(obj, list):
            return [f"{path}: not a list"]
        return [e for i, x in enumerate(obj) for e in shape_errors(x, want[0], f"{path}[{i}]")]
    got = TYPE.get(type(obj), type(obj).__name__)
    return [] if got in (want if isinstance(want, set) else {want}) else [f"{path}: {got}, not {want}"]


def sample_template(db: Path) -> dict:
    """the template above checked against a real MassWiki job database: its first record with identity hits"""
    import sqlite3
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    for (blob,) in con.execute("select data from main"):
        rec = json.loads(blob)
        if rec.get("spectrum") and rec["analysis"]["reference_library"]["identity_search"]:
            errs = shape_errors(rec, RECORD)
            if errs:
                fail(f"the template does not fit the sample {db.name}: {errs[:3]}")
            print(f"  template matches a record of {db.name} ({rec['wiki_id']})", flush=True)
            return rec
    fail(f"no record with identity hits in {db}")


def check_candidates(path: Path, results: Path) -> list:
    """the JSON array holds MassWiki search records (keys, order, types), floats stay floats, peaks_clean and entropy
    follow from peaks, and each feature's identity_search is identifications.tsv on the annotation copy (best first,
    at most 100)"""
    raw = path.read_text(encoding="utf-8")
    doc = json.loads(raw)
    if not isinstance(doc, list) or not doc:
        fail(f"{path.name}: not a non-empty JSON array")
    meta = json.loads((results / "identifications.json").read_text(encoding="utf-8"))
    # the annotation copy of each feature (1.2.6): the default copy whose best hit scores higher, the consensus on a
    # tie (default_spectra; a search before 1.2.6 names one copy, its kind on both spectra)
    dn = str(meta.get("default_spectrum", "")).endswith("_denoised")
    copies = meta.get("default_spectra") or (["consensus_denoised", "single_denoised"] if dn else ["consensus", "single"])
    with open(results / "identifications.tsv", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    best: dict[str, tuple] = {}
    for r in rows:
        if r["spectrum"] in copies and int(r["rank"]) == 1:
            k = (float(r["score"]), -copies.index(r["spectrum"]))
            if r["group_id"] not in best or k > best[r["group_id"]][0]:
                best[r["group_id"]] = (k, r["spectrum"])
    spec_of = {g: sp for g, (_, sp) in best.items()}
    spec = "/".join(sorted(set(spec_of.values()))) or copies[0]
    want: dict[str, list] = {}
    for r in rows:
        if r["spectrum"] == spec_of.get(r["group_id"], copies[0]):
            want.setdefault(r["group_id"], []).append((int(r["rank"]), r["name"], round(float(r["score"]), 4)))
    n_hits = n_named = 0
    for rec in doc:
        errs = shape_errors(rec, RECORD)
        if errs:
            fail(f"{path.name}: {rec.get('wiki_id')}: {errs[:3]}")
        s = rec["spectrum"]
        gid = s["id"]
        if not rec["wiki_id"].endswith(f"/{gid}"):
            fail(f"{path.name}: wiki_id {rec['wiki_id']} for spectrum {gid}")
        if s["scan"] != int(gid) or s["rt"] != s["rt_normalized"] or rec["status"] != "success" or not s["peaks"]:
            fail(f"{path.name}: {gid}: scan / rt / status / peaks")
        pc = s["peaks_clean"]
        if not all(isinstance(v, float) for p in s["peaks"] + pc for v in p):
            fail(f"{path.name}: {gid}: a peak value is not a float (1.0 written as 1)")
        if pc:
            if abs(sum(v for _, v in pc) - 1) > 1e-6 or any(pc[i][0] >= pc[i + 1][0] for i in range(len(pc) - 1)):
                fail(f"{path.name}: {gid}: peaks_clean does not sum to 1 in m/z order")
            if max(m for m, _ in pc) >= s["precursor_mz"] - 1.6 + 1e-9:
                fail(f"{path.name}: {gid}: peaks_clean holds the precursor region")
            ent = -sum(v * math.log(v) for _, v in pc if v > 0)
            if abs(ent - s["entropy"]) > 1e-6:
                fail(f"{path.name}: {gid}: entropy {s['entropy']} != {ent}")
        hits = rec["analysis"]["reference_library"]["identity_search"]
        got = [(c["name"], c["entropy_similarity"]) for c in hits]
        exp = [(n, sc) for _, n, sc in sorted(want.get(gid, []))][:100]
        if [n for n, _ in got] != [n for n, _ in exp] or any(abs(a[1] - b[1]) > 2e-4 for a, b in zip(got, exp)):
            fail(f"{path.name}: {gid} lists {got[:3]}..., identifications.tsv {exp[:3]}...")
        if any(float(c["precursor_mz"]) <= 0 for c in hits):
            fail(f"{path.name}: {gid}: a candidate precursor_mz is not a number")
        n_hits += len(hits)
        n_named += s["name"] != "Unknown"
    print(f"  {len(doc)} MassWiki records, {n_hits} candidates on {spec}, {n_named} with an annotation as name; "
          "keys, order and types as MassWiki's; every list matches identifications.tsv", flush=True)
    return doc


def open_results(page, results: Path):
    page.locator(".nav-item").nth(1).click()
    page.get_by_role("button", name="Choose folder").click()
    status = page.locator("#browse .dialog-actions .body-small")
    page.wait_for_function("e => e.textContent.length > 0", arg=status.element_handle(), timeout=30_000)
    box = page.locator("#browse .dialog-head input")
    box.fill(str(results))
    box.press("Enter")
    page.get_by_text("Mosa results here", exact=True).wait_for(timeout=30_000)
    for _ in range(3):                    # headless Chromium sometimes needs a second click to close the dialog
        page.locator("#browse .dialog-actions .btn.filled").click()
        try:
            page.locator(".tabs").wait_for(timeout=10_000)
            break
        except Exception:
            pass
    page.locator('[data-tab="features"]').click()
    page.get_by_role("button", name="Download table").wait_for(timeout=60_000)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="the app's link, with its token")
    ap.add_argument("--results", required=True, type=Path, help="a results folder with an identity search")
    ap.add_argument("--save", required=True, type=Path, help="folder for the downloads and screenshots")
    ap.add_argument("--channel", default=None, help="msedge / chrome: the installed browser")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--sample", type=Path, help="a MassWiki job database (.db) to check the record template against")
    a = ap.parse_args(argv)
    a.save.mkdir(parents=True, exist_ok=True)
    results = a.results.resolve()
    if a.sample:
        sample_template(a.sample)
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=a.channel, headless=not a.headed)
        ctx = browser.new_context(accept_downloads=True, viewport={"width": 1500, "height": 950})
        page = ctx.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(a.url)
        try:
            open_results(page, results)
        except Exception as e:
            page.screenshot(path=str(a.save / "open_failed.png"))
            fail(f"opening {results} in the app: {e}")
        page.screenshot(path=str(a.save / "features.png"))
        # the toolbar: the listed rows as a table, their candidates as JSON
        t = download(page, page.get_by_role("button", name="Download table").click, a.save, "Download table")
        head = t.read_text(encoding="utf-8").splitlines()
        if not head or not head[0].startswith("group_id\tmz\trt_s") or len(head) < 2:
            fail(f"{t.name}: not the feature table")
        c = download(page, page.get_by_role("button", name="Download candidates").click, a.save, "Download candidates")
        app_doc = check_candidates(c, results)
        # the Download menu: whole files from the server (plain links)
        for label, check in (("Feature table", lambda b: b.startswith(b"group_id\t")), ("Report", lambda b: b"peak3d-data" in b)):
            page.get_by_role("button", name="Download", exact=True).click()
            item = page.locator("a.menu-item", has_text=label).first
            f = download(page, item.click, a.save, f"menu: {label}")
            if not check(f.read_bytes()):
                fail(f"{f.name}: unexpected content")
            if not f.name.startswith(results.name):
                if not (label == "Report" and results.name in f.name):
                    fail(f"{f.name}: the name does not carry the folder name {results.name!r}")
        page.screenshot(path=str(a.save / "after.png"))
        # the report on its own (file://, no server): its Download candidates
        rp = page.context.new_page()
        rp.on("pageerror", lambda e: errors.append("report: " + str(e)))
        rp.goto((results / "report.html").as_uri())
        rp.locator('[data-tab="features"]').click()
        sub = a.save / "report"
        sub.mkdir(exist_ok=True)
        r = download(rp, rp.get_by_role("button", name="Download candidates").click, sub, "report: Download candidates")
        rep_doc = check_candidates(r, results)
        if [r["analysis"] for r in rep_doc] != [r["analysis"] for r in app_doc]:
            fail("the report's candidates differ from the app's")
        browser.close()
    if errors:
        fail("page errors: " + " | ".join(errors))
    print("browser check passed", flush=True)


if __name__ == "__main__":
    main()
