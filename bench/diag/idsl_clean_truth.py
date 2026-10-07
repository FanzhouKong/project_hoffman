# cleaned IDSL003 truth: keep only the curated labels that Claude's blind visual check (first look) agreed with
# outright (results/diag_idsl/visual_check/merged_VX.tsv, status1 == agree), and drop the TN member of any remaining
# TP-TN pair within the scoring tolerance (10 ppm, 0.1 min), which a tool reporting the TP peak could not avoid hitting.
# Everything the check flagged (contradicted, questionable, split, undecided) is removed, as the user decided
# (2026-10-05). Output keeps the original CSV columns plus the visual call.
# usage: idsl_clean_truth.py   -> data/truth/IDSL003/labels_clean.csv, data/truth/IDSL003/params.json
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/truth/IDSL003"
OUT.mkdir(parents=True, exist_ok=True)
lab = pd.read_csv(ROOT / "data/raw/IDSL_IPA/idslipa_benchmarking_dataset.csv", encoding="utf-8-sig")
M = pd.read_csv(ROOT / "results/diag_idsl/visual_check/merged_VX.tsv", sep="\t", keep_default_na=False)
keep = M[M.status1 == "agree"].sort_values("idx")
L = lab.iloc[keep.idx.values].copy()
L["visual_call"] = keep.call.values
mz, rt = L["m/z"].values, L["RT(min)"].values
tp = (L["Manual Curation"] == "TP").values
o = np.argsort(mz)
drop = set()
for j in range(len(o)):
    hi = np.searchsorted(mz[o], mz[o[j]] * (1 + 10e-6), side="right")
    for k in range(j + 1, hi):
        a, b = o[j], o[k]
        if abs(rt[a] - rt[b]) <= 0.1 and tp[a] != tp[b]:
            drop.add(a if not tp[a] else b)
L = L.drop(index=L.index[sorted(drop)])
L = L.rename(columns={L.columns[0]: "id"})
L.to_csv(OUT / "labels_clean.csv", index=False)
info = dict(
    source="data/raw/IDSL_IPA/idslipa_benchmarking_dataset.csv (Zenodo 6302236, 20,000 labels)",
    method="blind two-pass visual EIC check of every label (results/diag_idsl/visual_check/SUMMARY.md); kept labels "
           "whose first blind call agreed with the curated label (TP: peak apex inside +/-0.1 min at 10 ppm; "
           "TN: no peak / tail only / neighbouring ion), minus the TN member of TP-TN pairs within 10 ppm and 0.1 min",
    created="2026-10-05", n_original=int(len(lab)), n_first_look_agree=int(len(keep)),
    n_dropped_conflict_tn=len(drop), n=int(len(L)), n_tp=int((L["Manual Curation"] == "TP").sum()),
    n_tn=int((L["Manual Curation"] == "TN").sum()),
    removed=dict(flagged_for_review=int(M.final.isin(["disagree", "split", "unsure"]).sum()),
                 first_look_questionable_then_settled=int(((M.status1 == "unsure") & (M.final == "agree")).sum())))
json.dump(info, open(OUT / "params.json", "w"), indent=1)
print(json.dumps(info, indent=1))
