# merge the blind visual calls on the IDSL003 label sheets (idsl_visual_sheets.py) with the curated labels.
# call P = peak apex inside the +/-0.1 min window, W = questionable, O = only a tail of a peak outside, N = no peak,
# M = peak only at a neighbouring m/z (30 ppm). Label TP agrees with P; label TN agrees with N / O / M.
# Flags: "disagree" (TP called N/O/M, TN called P) and "unsure" (W). With a pass-2 call file the final status
# combines both passes. Also compares with the earlier 260 blind calls (label_audit/blind) and the automated verdicts.
# usage: idsl_visual_merge.py VISUAL_DIR PREFIX [PREFIX2]   -> VISUAL_DIR/merged_PREFIX.tsv, flagged ids, log on stdout
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
V = Path(sys.argv[1])
P1 = sys.argv[2]
P2 = sys.argv[3] if len(sys.argv) > 3 else None
pd.set_option("display.width", 220, "display.max_columns", 20)
A = pd.read_csv(ROOT / "results/diag_idsl/label_review/verdict.tsv", sep="\t")
lab = np.where(A["Manual Curation"] == "TP", "TP", "TN")


def calls(prefix):
    K = pd.read_csv(V / "keys" / f"{prefix}key.tsv", sep="\t")
    parts = []
    for f in sorted((V / "calls").glob(f"{prefix}[0-9][0-9][0-9].tsv")):
        c = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False)
        c.columns = [x.strip().lower() for x in c.columns]
        parts.append(c[["code", "call"] + (["note"] if "note" in c.columns else [])])
    C = pd.concat(parts) if parts else pd.DataFrame(columns=["code", "call", "note"])
    C["call"] = C["call"].str.strip().str.upper().str[:1]
    C = C.drop_duplicates("code", keep="last")
    M = K.merge(C, on="code", how="left")
    bad = M.call.notna() & ~M.call.isin(list("PWONM"))
    if bad.any():
        print(f"{prefix}: {bad.sum()} calls outside P/W/O/N/M set to W:", M[bad].call.value_counts().to_dict())
        M.loc[bad, "call"] = "W"
    print(f"{prefix}: {M.call.notna().sum()} of {len(M)} panels called "
          f"({M.code.str[:len(prefix) + 3][M.call.notna()].nunique()} sheets)")
    return M


def status(label, call):
    if pd.isna(call):
        return "missing"
    if call == "W":
        return "unsure"
    if label == "TP":
        return "agree" if call == "P" else "disagree"
    return "disagree" if call == "P" else "agree"


M1 = calls(P1)
M1["label"] = lab[M1.idx]
M1["verdict"] = A.verdict.values[M1.idx]
M1["status1"] = [status(l, c) for l, c in zip(M1.label, M1.call)]
print("\n== pass 1: label x call")
print(pd.crosstab(M1.label, M1.call.fillna("-"), margins=True))
print("\n== pass 1 status by label")
print(pd.crosstab(M1.label, M1.status1, margins=True))

# earlier blind calls (old rubric: T = flank / tail of a peak whose apex is > 3 s away, inside or outside the window)
B = pd.read_csv(ROOT / "results/diag_idsl/label_audit/blind/scored.tsv", sep="\t")[["id", "call"]].rename(columns={"call": "blind260"})
J = M1.merge(B, on="id")
if len(J):
    print(f"\n== agreement with the earlier 260 blind calls (n = {J.call.notna().sum()} called)")
    print(pd.crosstab(J.blind260, J.call.fillna("-")))
    core = J[J.blind260.isin(list("PNM")) & J.call.isin(list("PNMO"))]
    same = ((core.blind260 == "P") == (core.call == "P")).mean() if len(core) else np.nan
    print(f"  peak / no-peak agreement where both calls are definite: {same:.3f} (n = {len(core)})")

print("\n== automated verdict x pass-1 call")
print(pd.crosstab(M1.verdict, M1.call.fillna("-")))

out = M1.copy()
if P2:
    M2 = calls(P2)[["idx", "call"] + (["note"] if "note" in calls(P2).columns else [])]
    M2 = M2.rename(columns={"call": "call2", "note": "note2"})
    out = out.merge(M2, on="idx", how="left")
    out["status2"] = [status(l, c) if isinstance(c, str) else "" for l, c in zip(out.label, out.call2)]

    def final(r):
        # pass 2 re-checks every pass-1 flag: a W settled for the label is accepted, a W settled against it is a
        # disagreement, a disagreement the second look does not confirm is "split", two W's stay "unsure"
        s1, s2 = r.status1, r.status2
        if s1 == "agree" or not s2:
            return s1
        if s1 == "unsure":
            return s2
        if s2 == "disagree":
            return "disagree"
        return "split" if s2 == "agree" else "unsure"
    out["final"] = out.apply(final, axis=1)
    print("\n== pass 2 among flagged: status1 x status2")
    f = out[out.status1.isin(["disagree", "unsure"])]
    print(pd.crosstab(f.status1, f.status2, margins=True))
    print("\n== final status by label")
    print(pd.crosstab(out.label, out.final, margins=True))
else:
    out["final"] = out.status1
flag = out[out.final.isin(["disagree", "unsure", "split"])].sort_values("idx")
tag = P1 + (P2 or "")
out.to_csv(V / f"merged_{tag}.tsv", sep="\t", index=False)
flag.idx.to_csv(V / f"flagged_{tag}.txt", index=False, header=False)
print(f"\nflagged for review: {len(flag)} -> {V / f'flagged_{tag}.txt'}")
