"""Reliability of specdenoise.smiles (2026-11-03): its formula vs the libraries' own, record by record.

    python smiles_check.py OUT_DIR LIBRARY.msp [LIBRARY.msp ...]

For every record with both a SMILES and a formula, the formula read from the SMILES is compared with the library's
(unique (SMILES, formula) pairs). A disagreement is arbitrated by the record's measured precursor m/z: the ion
formula of each side (search.ion_formula: the named adduct, else the one that fits) is checked against it. "ours
wrong" = the library's formula fits the precursor and ours does not: the case that matters, since a wrong formula
that also fits the m/z would denoise with the wrong structure. GNPS exports give Formula_smiles (a formula computed
from the SMILES by another toolkit) instead of Formula; that is read with a line scanner. Records without a formula
(most of MoNA) report how many get an ion formula from the SMILES that fits their precursor m/z.
Writes OUT_DIR/smiles_check.md and smiles_disagree.tsv."""
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from specdenoise import formula as fm
from specdenoise.search import _neutral, ion_formula
from specdenoise.smiles import parse, smiles_formula

ISO = re.compile(r"\[\d")
SYMS = re.compile(r"\[(?:\d+)?([A-Z][a-z]?|se|as|te|[bcnops])")


def why_refused(smi: str) -> str:
    if ISO.search(smi):
        return "isotope label"
    if "*" in smi:
        return "wildcard"
    if parse(smi) is None:
        return "syntax"
    bad = {s for s in SYMS.findall(smi) if s not in fm.IDX and s not in ("b", "c", "n", "o", "p", "s") and s != "H"}
    bad |= {"B"} if re.search(r"B(?!r)", re.sub(r"\[[^\]]*\]", "", smi)) else set()
    return "element " + ",".join(sorted(bad)) if bad else "element"


def gnps_records(path):
    """(smiles, formula_smiles, precursor m/z, adduct) of a GNPS .msp, read line by line"""
    rec = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                if rec:
                    yield rec.get("smiles", ""), rec.get("formula_smiles", ""), rec.get("precursormz", ""), rec.get("adduct", ""), rec.get("ionmode", "")
                rec = {}
                continue
            k, _, v = line.partition(":")
            k = k.strip().lower().replace("_", "") if k.strip().lower() != "formula_smiles" else "formula_smiles"
            if k in ("smiles", "formula_smiles", "precursormz", "adduct", "ionmode"):
                rec[k] = v.strip()
    if rec:
        yield rec.get("smiles", ""), rec.get("formula_smiles", ""), rec.get("precursormz", ""), rec.get("adduct", ""), rec.get("ionmode", "")


def records(path):
    """(smiles, formula, precursor m/z, adduct, mode code) of every record"""
    if "gnps" in Path(path).name.lower():
        for smi, f, pmz, add, mode in gnps_records(path):
            try:
                pmz = float(pmz.split()[0]) if pmz else float("nan")
            except ValueError:
                pmz = float("nan")
            m = mode.lower()
            yield smi, f, pmz, add, 1 if m.startswith("p") else 2 if m.startswith("n") else 0
        return
    from peak3d.library import read_msp
    lib = read_msp(path, comments=False)
    for i in range(len(lib)):
        yield lib.smiles[i], lib.formula[i], float(lib.precursor_mz[i]), lib.precursor_type[i], int(lib.mode[i])


def check(path):
    t = time.perf_counter()
    pairs = {}                 # (smiles, formula) -> first record's (pmz, adduct, mode), count
    nof = {}                   # smiles of records without a formula -> (pmz, adduct, mode) list (first 3)
    n_rec = n_both = n_smiles_only = 0
    for smi, f, pmz, add, mode in records(path):
        n_rec += 1
        smi, f = (smi or "").strip(), (f or "").strip()
        if smi and f:
            n_both += 1
            k = (smi, f)
            if k in pairs:
                pairs[k][1] += 1
            else:
                pairs[k] = [(pmz, add, mode), 1]
        elif smi:
            n_smiles_only += 1
            nof.setdefault(smi, []).append((pmz, add, mode))
    cat = Counter()
    rows = []
    for (smi, f), ((pmz, add, mode), cnt) in pairs.items():
        lib_f = _neutral(f)
        ours = smiles_formula(smi)
        if lib_f is None:
            cat["library formula unreadable"] += 1
            continue
        if ours is None:
            cat["refused: " + why_refused(smi)] += 1
            continue
        if (ours[0] == lib_f).all():
            cat["agree"] += 1
            continue
        big = smiles_formula(smi, "largest")
        dh = int(ours[0][fm.H] - lib_f[fm.H])
        heavy_same = (np.delete(ours[0], fm.H) == np.delete(lib_f, fm.H)).all()
        kind = ("largest component" if big is not None and (big[0] == lib_f).all() else
                f"H {dh:+d}" if heavy_same else "heavy atoms")
        a = ion_formula(f, add, mode, pmz)
        b = ion_formula("", add, mode, pmz, smiles=smi)
        if a is None and b is None:
            verdict = "neither fits"
        elif b is None:
            verdict = "ours wrong"
        elif a is None:
            verdict = "library wrong"
        else:
            verdict = "both fit" if (a[0] == b[0]).all() else "both fit, differ"
        cat[f"disagree: {kind}: {verdict}"] += 1
        rows.append(dict(library=Path(path).name, smiles=smi, library_formula=f, ours=fm.format(ours[0]),
                         ours_charge=ours[1], kind=kind, verdict=verdict, precursor_mz=pmz, adduct=add, records=cnt))
    # records without a formula: does the SMILES give an ion formula that fits the precursor?
    cov = Counter()
    fail_add = Counter()
    fail_ex = []
    for smi, recs in nof.items():
        for pmz, add, mode in recs:
            cov["records"] += 1
            if smiles_formula(smi) is None:
                cov["refused"] += 1
            elif ion_formula("", add, mode, pmz, smiles=smi) is not None:
                cov["ion formula fits m/z"] += 1
            else:
                cov["no adduct fits m/z"] += 1
                fail_add[add or "(none)"] += 1
                if len(fail_ex) < 40:
                    fail_ex.append(dict(library=Path(path).name, smiles=smi, ours=fm.format(smiles_formula(smi)[0]),
                                        charge=smiles_formula(smi)[1], precursor_mz=pmz, adduct=add, mode=mode))
    return dict(library=Path(path).name, records=n_rec, with_both=n_both, unique_pairs=len(pairs),
                smiles_only_records=n_smiles_only, cat=cat, cov=cov, fail_add=fail_add, fail_ex=fail_ex,
                seconds=round(time.perf_counter() - t, 1)), rows


def main(out_dir, *libs):
    out_dir = Path(out_dir)
    lines = ["# SMILES formula check\n", "Unique (SMILES, formula) pairs per library; a disagreement is arbitrated by "
             "the record's precursor m/z (ion formula of each side within 0.02 Da).\n"]
    allrows = []
    failx = []
    for path in libs:
        res, rows = check(path)
        allrows += rows
        c = res["cat"]
        n = res["unique_pairs"]
        ag = c.get("agree", 0)
        lines.append(f"## {res['library']}\n\n{res['records']:,} records, {res['with_both']:,} with SMILES and formula "
                     f"= {n:,} unique pairs ({res['seconds']} s)\n")
        lines.append("| outcome | pairs | % |\n|---|---|---|")
        for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {k} | {v:,} | {100 * v / max(n, 1):.3f} |")
        cmp = n - sum(v for k, v in c.items() if k.startswith("refused") or k.startswith("library formula"))
        wrong = sum(v for k, v in c.items() if k.endswith("ours wrong") or k.endswith("both fit, differ"))
        lines.append(f"\nagree {ag:,} of {cmp:,} compared ({100 * ag / max(cmp, 1):.3f} %); ours contradicted by the "
                     f"precursor m/z or fitting it with another formula: {wrong:,}\n")
        if res["cov"]:
            cv = res["cov"]
            lines.append(f"records without a formula: {cv['records']:,}; SMILES gives an ion formula fitting the "
                         f"precursor: {cv.get('ion formula fits m/z', 0):,} "
                         f"({100 * cv.get('ion formula fits m/z', 0) / max(cv['records'], 1):.1f} %), refused "
                         f"{cv.get('refused', 0):,}, no adduct fits {cv.get('no adduct fits m/z', 0):,}\n")
            if res["fail_add"]:
                lines.append("adducts of the records where none fits: " + ", ".join(
                    f"{k} {v:,}" for k, v in res["fail_add"].most_common(15)) + "\n")
                failx.extend(res["fail_ex"])
        print("\n".join(lines[-12:]), flush=True)
    pd.DataFrame(allrows).to_csv(out_dir / "smiles_disagree.tsv", sep="\t", index=False)
    pd.DataFrame(failx).to_csv(out_dir / "smiles_nofit_examples.tsv", sep="\t", index=False)
    (out_dir / "smiles_check.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
