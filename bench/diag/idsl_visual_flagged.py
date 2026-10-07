# build flagged.json for the IDSL003 label decision page from the merged two-pass visual check
# (idsl_visual_merge.py VISUAL_DIR V X -> merged_VX.tsv): every label whose final status is disagree, split or unsure,
# with both blind calls and notes. Order: disagree, split, unsure; curated TP before TN; then label index.
# usage: idsl_visual_flagged.py VISUAL_DIR OUT_JSON
import sys, json
from pathlib import Path
import pandas as pd

V, OUT = Path(sys.argv[1]), Path(sys.argv[2])
M = pd.read_csv(V / "merged_VX.tsv", sep="\t", keep_default_na=False)
F = M[M.final.isin(["disagree", "split", "unsure"])].copy()
# "disagree": both looks contradict the label; "lean": first look unsure, second contradicts
F.loc[(F.final == "disagree") & (F.status1 == "unsure"), "final"] = "lean"
F["o1"] = F.final.map({"disagree": 0, "lean": 1, "split": 2, "unsure": 3})
F["o2"] = (F.label == "TN").astype(int)
F = F.sort_values(["o1", "o2", "idx"])
s = lambda v: str(v).strip()
items = [dict(i=int(r.idx), c1=s(r.call), n1=s(r.note), c2=s(r.call2), n2=s(r.note2), st=r.final) for r in F.itertuples()]
cnt = F.groupby(["final", "label"]).size().to_dict()
n_agree = int((M.final == "agree").sum())
about = (f"Every one of the 20,000 labels was drawn blind and called by Claude: is there a real chromatographic peak "
         f"with its apex inside the ±0.1 min window at 10 ppm? {n_agree:,} calls agreed with the curated label and are not "
         f"shown. The {len(items):,} listed here had a second, independent blind look on larger panels: "
         f"{sum(v for (f, l), v in cnt.items() if f == 'disagree'):,} where both looks contradict the label, "
         f"{sum(v for (f, l), v in cnt.items() if f == 'lean'):,} where the first look was unsure and the second contradicts it, "
         f"{sum(v for (f, l), v in cnt.items() if f == 'split'):,} where the two looks disagree with each other, and "
         f"{sum(v for (f, l), v in cnt.items() if f == 'unsure'):,} that stayed undecidable.")
json.dump(dict(items=items, about=about, counts={f"{f}_{l}": int(v) for (f, l), v in cnt.items()}),
          open(OUT, "w"), separators=(",", ":"))
print(len(items), "items;", about)
