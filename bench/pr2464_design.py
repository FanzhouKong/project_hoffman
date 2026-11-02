#!/usr/bin/env python3
"""Export the PR002464 benchmark design (1-doc/DesignSpreadsheet_20230824.xlsx) to plain TSV.

Every study package ships the same design spreadsheet; this checks that and writes, under
data/raw/PR002464/design/:
  standards.tsv           the 96 spiked standards: mixture (4mix), relative amount in spike A and
                          spike B (spikeA_vol, spikeB_vol), log2FC = log2(A/B), formula, SMILES ...
  internal_standards.tsv  the labelled internal standards (constant in every vial but solvent blanks)
  vial_contents.tsv       relative amount of each component per vial (sheet "Vial Contents")
Needs openpyxl (pandas.read_excel); 05_download_pr002464.sbatch provides it.
usage: pr2464_design.py STUDY [STUDY ...]
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/raw/PR002464"
XLSX = "1-doc/DesignSpreadsheet_20230824.xlsx"


def main(studies):
    paths = [BASE / st / XLSX for st in studies]
    digests = {p: hashlib.md5(p.read_bytes()).hexdigest() for p in paths}
    if len(set(digests.values())) > 1:
        for p, d in digests.items():
            print(d, p)
        sys.exit("design spreadsheets differ between studies; export them separately")
    book = pd.read_excel(paths[0], sheet_name=None)
    out = BASE / "design"
    out.mkdir(exist_ok=True)

    s = book["Standards"]
    s = s[s["standard.number"].notna()]
    std = pd.DataFrame({
        "std": s["standard.number"].astype(int),
        "name": s["pubchem.name"],
        "cas": s["ChromadexCAS#"],
        "pubchem_cid": s["standard.cid"].astype("Int64"),
        "mixture": s["mixture"].astype(int),
        "mix_conc_ug_ml": s["[4mix] ug/mL"],
        "spikeA_vol": s["spike A_vols"],
        "spikeB_vol": s["spike B_vols"],
        "log2FC": s["log2FC"],
        "formula": s["MolecularFormula"],
        "monoisotopic_mass": s["MonoisotopicMass"],
        "charge": s["Charge"].astype(int),
        "xlogp": s["XLogP"],
        "inchikey": s["InChIKey"],
        "smiles": s["IsomericSMILES"],
    }).sort_values("std")
    assert len(std) == 96 and std["std"].is_unique
    std.to_csv(out / "standards.tsv", sep="\t", index=False)

    i = book["Internal Standards"]
    i = i[i["Compound"].notna()]
    pd.DataFrame({
        "compound": i["Compound"].str.strip(),
        "vial": i["Vial"].astype(int),
        "final_conc_ug_ml": i["target Final concentration (ug/mL)"],
    }).to_csv(out / "internal_standards.tsv", sep="\t", index=False)

    v = book["Vial Contents"]
    v.columns = v.iloc[0].astype(str).str.strip()   # row 0 holds the real header
    v = v.iloc[1:]
    v = v[pd.to_numeric(v["Vial#"], errors="coerce").notna()]
    vial = pd.DataFrame({
        "vial": v["Vial#"].astype(int),
        "group": v["figure reference"],
        **{c: v[c].astype(float) for c in
           ["SRM 1950", "SRM 8231", "Yeast 12C", "Yeast 13C", "[spikeA]", "[spikeB]", "[IS]"]},
    })
    vial.to_csv(out / "vial_contents.tsv", sep="\t", index=False)
    print(f"wrote {out}: {len(std)} standards, {len(i)} internal standards, {len(vial)} vials")


if __name__ == "__main__":
    main(sys.argv[1:] or ["ST004500"])
