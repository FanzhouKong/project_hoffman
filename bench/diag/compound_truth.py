"""Does Mosa's compound grouping join the ions of one spiked standard?

Truth: data/truth/PR002464/<study>/spiked_truth.tsv (bench/spiketruth.py). Every truth ion carries its standard
and a class: formula_ion / adduct (predicted adducts of the standard), isotope, fragment (co-eluting ion below the
standard's mass with the standard's A/B concentration signature) or unannotated (co-eluting, above). The ions of
one standard are one truth cluster.

Per study: truth ions -> nearest aligned row (ppm, RT tolerance) -> compound_id. Scores:
  linked recall      share of a standard's non-main matched ions in the same compound as its main ion, per class;
                     also over the co-eluting ones only (consensus apexes within 2 grouping windows), since an ion
                     Mosa placed far from the main ion cannot be grouped by any rule
  completeness       share of a standard's matched ions inside its largest compound
  pair precision     among pairs of matched truth rows sharing a compound: same standard (or shared_with) / all
  co-elution baseline  the same precision when every co-eluting pair (|dRT| <= window) is joined
  extras             members of the truth standards' compounds that match no truth ion of any tier (junk proxy:
                     unknown, but the fewer the better at equal recall)
  role accuracy      predicted role == truth ion name, for formula ions
A sweep re-clusters compound_pairs.tsv at other shape cuts and intensity vetoes, so the rule is judged on truth.

usage: python bench/diag/compound_truth.py --out <peak3d_out> --study ST004501 [--ppm 5] [--rt-tol 0.1]
       [--tier A|all] [--report DIR]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.score import nearest  # noqa: E402
from peak3d.compounds import _star_kernel  # noqa: E402

TRUTH = ROOT / "data/truth/PR002464"
CLASSES = ["formula_ion", "adduct", "isotope", "fragment", "unannotated"]
META = {"group_id", "mz", "rt", "rt_min", "rt_max", "n_detected", "n_confirmed", "n_filled", "mz_ppm_spread", "rt_sd",
        "iso_offset", "charge", "compound_id", "role", "neutral_mass", "is_base", "n_members"}


def load_truth(study, tier):
    t = pd.read_csv(TRUTH / study / "spiked_truth.tsv", sep="\t", keep_default_na=False, na_values=[""])
    t["mz_q"] = t["mz_obs"].fillna(t["mz"])
    t["rt_q"] = t["rt_apex"]
    t["flags"] = t["flags"].fillna("")
    t["shared_with"] = t["shared_with"].fillna("").astype(str)
    all_t = t.copy()
    if tier != "all":
        t = t[t["tier"] == tier]
    t = t[~t["flags"].str.contains("ft_artifact")]
    return t.reset_index(drop=True), all_t


def shared_sets(t):
    out = {}
    for s, sw in zip(t["std"], t["shared_with"]):
        if sw and sw != "nan":
            out.setdefault(int(s), set()).update(int(float(x)) for x in sw.split(";") if x)
    return out


def main_ions(t):
    f = t[t["source"] == "formula"]
    if f.empty:
        f = t
    return f.sort_values("height_dom", ascending=False).groupby("std").head(1).set_index("std")


def cluster_metrics(t, cid, mains, shared, rt_win, label, coel):
    """cid: compound id per truth row (-1 unmatched); coel: bool per truth row, consensus apex within 2 windows of
    its standard's main ion"""
    m = cid >= 0
    linked = {c: [0, 0, 0, 0] for c in CLASSES}      # linked, n, linked among co-eluting, n co-eluting
    comp_frac, rows = [], []
    for s, g in t[m].groupby("std"):
        if s not in mains.index or cid[mains.loc[s, "_i"]] < 0:
            continue
        mi = mains.loc[s, "_i"]
        cm = cid[mi]
        others = g.index[g.index != mi]
        for i in others:
            c = t.loc[i, "annotation_class"]
            linked[c][1] += 1
            linked[c][0] += int(cid[i] == cm)
            if coel[i]:
                linked[c][3] += 1
                linked[c][2] += int(cid[i] == cm)
        vals, cnt = np.unique(cid[g.index], return_counts=True)
        comp_frac.append(cnt.max() / len(g))
        rows.append(dict(std=int(s), n_matched=len(g), n_linked_to_main=int((cid[others] == cm).sum()),
                         n_others=len(others), completeness=cnt.max() / len(g), n_compounds=len(vals)))
    per_std = pd.DataFrame(rows)
    idx = np.flatnonzero(m)
    _, first = np.unique(t["row"].values[idx], return_index=True)
    idx = idx[first]
    std = t["std"].values[idx].astype(int); c = cid[idx]; rt = t["rt_q"].values[idx]
    same_c = c[:, None] == c[None, :]
    same_s = std[:, None] == std[None, :]
    amb = np.zeros_like(same_s)
    for a, bs in shared.items():
        for b in bs:
            amb |= (std[:, None] == a) & (std[None, :] == b)
            amb |= (std[:, None] == b) & (std[None, :] == a)
    upper = np.triu(np.ones_like(same_c), 1)
    joined = same_c & upper & ~amb
    tp = int((joined & same_s).sum()); fp = int((joined & ~same_s).sum())
    coelp = (np.abs(rt[:, None] - rt[None, :]) <= rt_win) & upper & ~amb
    ctp = int((coelp & same_s).sum()); cfp = int((coelp & ~same_s).sum())
    L = sum(v[0] for v in linked.values()); Ln = sum(v[1] for v in linked.values())
    Lc = sum(v[2] for v in linked.values()); Lcn = sum(v[3] for v in linked.values())
    summary = dict(label=label, n_matched=int(m.sum()), n_std_scored=len(per_std),
                   linked_all=L / max(1, Ln), linked_coeluting=Lc / max(1, Lcn), n_coeluting=Lcn,
                   completeness_mean=float(np.mean(comp_frac)) if comp_frac else np.nan,
                   pair_tp=tp, pair_fp=fp, pair_precision=tp / (tp + fp) if tp + fp else np.nan,
                   coelution_precision=ctp / (ctp + cfp) if ctp + cfp else np.nan)
    for k, (a, b, a2, b2) in linked.items():
        summary[f"linked_{k}"] = a / b if b else np.nan
        summary[f"coel_linked_{k}"] = a2 / b2 if b2 else np.nan
        summary[f"n_{k}"] = b
    return summary, per_std


def extras(cid_all, t_rows_all_tiers, cid_truth_rows):
    """for the compounds holding matched tier-scored truth rows: members that match no truth ion of any tier"""
    truth_rows = set(t_rows_all_tiers.tolist())
    comps = np.unique(cid_truth_rows[cid_truth_rows >= 0])
    n_mem = n_truth = 0
    sizes = []
    for c in comps:
        mem = np.flatnonzero(cid_all == c)
        n_mem += len(mem)
        n_truth += sum(1 for r in mem if r in truth_rows)
        sizes.append(len(mem))
    return dict(n_truth_compounds=len(comps), members=int(n_mem), members_truth=int(n_truth),
                extras=int(n_mem - n_truth), extras_per_compound=(n_mem - n_truth) / max(1, len(comps)),
                size_median=float(np.median(sizes)) if sizes else np.nan)


def recluster(pairs, G, g_h, eligible, shape_cut, rel_cut, int_cut):
    r = pairs["r_shape"].values
    ok = np.where(pairs["relation"].values, r >= rel_cut, r >= shape_cut) & np.isfinite(r)
    if int_cut is not None:
        ri = pairs["r_int"].values
        ok &= ~np.isfinite(ri) | (ri >= int_cut)
    ei, ej = pairs["gid_a"].values[ok], pairs["gid_b"].values[ok]
    src = np.concatenate([ei, ej]); dst = np.concatenate([ej, ei])
    o = np.argsort(src, kind="stable")
    indptr = np.zeros(G + 1, np.int64)
    np.cumsum(np.bincount(src, minlength=G), out=indptr[1:])
    cid, C = _star_kernel(np.argsort(-g_h, kind="stable"), indptr, dst[o], eligible)
    loose = np.flatnonzero(cid < 0)
    cid[loose] = C + np.arange(len(loose))
    return cid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="peak3d_out directory")
    ap.add_argument("--study", required=True)
    ap.add_argument("--ppm", type=float, default=5.0)
    ap.add_argument("--rt-tol", type=float, default=0.1)
    ap.add_argument("--tier", default="A")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--no-sweep", action="store_true")
    a = ap.parse_args()
    rep = a.report or a.out.parent / "compound_truth"
    rep.mkdir(parents=True, exist_ok=True)

    t, t_all = load_truth(a.study, a.tier)
    T = pd.read_csv(a.out / "aligned_feature_table.tsv", sep="\t")
    params = json.loads((a.out / "params.json").read_text())
    info = params.get("stage_times_s", {}).get("compounds", {})
    rt_win = float(params.get("rt_tol_min", 0.05))
    feats = pd.DataFrame({"mz": T["mz"], "rt": T["rt"]})
    t["row"] = nearest(feats, t["mz_q"].values, t["rt_q"].values, a.ppm, a.rt_tol)
    t["_i"] = np.arange(len(t))
    rows_all_tiers = nearest(feats, t_all["mz_q"].values, t_all["rt_q"].values, a.ppm, a.rt_tol)
    rows_all_tiers = rows_all_tiers[rows_all_tiers >= 0]
    shared = shared_sets(t)
    mains = main_ions(t)
    hit = t["row"].values >= 0
    row = np.maximum(t["row"].values, 0)
    cid = np.where(hit, T["compound_id"].values[row], -1)
    t["compound_id"] = cid
    t["role_pred"] = np.where(hit, T["role"].values[row], "")
    t["group_id"] = np.where(hit, T["group_id"].values[row], -1)
    t["rt_mosa"] = np.where(hit, T["rt"].values[row], np.nan)
    t["n_members"] = np.where(hit, T["n_members"].values[row], 0)
    main_c = {s: cid[i] for s, i in zip(mains.index, mains["_i"])}
    main_rt = {s: t["rt_mosa"].values[i] for s, i in zip(mains.index, mains["_i"])}
    t["linked_to_main"] = [bool(c >= 0 and main_c.get(s, -2) == c) for s, c in zip(t["std"], cid)]
    t["drt_main_s"] = [60 * abs(r - main_rt.get(s, np.nan)) for s, r in zip(t["std"], t["rt_mosa"])]
    coel = (t["drt_main_s"].values <= 2 * 60 * rt_win)

    summary, per_std = cluster_metrics(t, cid, mains, shared, rt_win, "pipeline", coel)
    f = t[(t["source"] == "formula") & hit]
    summary["role_n"] = len(f)
    summary["role_correct"] = float(np.mean(f["ion"].values == f["role_pred"].values)) if len(f) else np.nan
    summary["role_named"] = float(np.mean([str(r).startswith("[") for r in f["role_pred"]])) if len(f) else np.nan
    summary.update(extras(T["compound_id"].values, rows_all_tiers, cid))
    summary.update({f"info_{k}": v for k, v in info.items()})

    # per class: was the main<->ion pair tested, and what did it score
    pairs = pd.read_csv(a.out / "compound_pairs.tsv", sep="\t")
    key = {}
    for k, (ga, gb) in enumerate(zip(pairs["gid_a"].values, pairs["gid_b"].values)):
        key[(min(ga, gb), max(ga, gb))] = k
    rows = []
    for s, g in t[hit].groupby("std"):
        if s not in mains.index or cid[mains.loc[s, "_i"]] < 0:
            continue
        gm = int(t["group_id"].values[mains.loc[s, "_i"]])
        for i in g.index:
            gi = int(t["group_id"].values[i])
            if gi == gm:
                continue
            k = key.get((min(gm, gi), max(gm, gi)))
            rec = dict(std=s, cls=t.loc[i, "annotation_class"], linked=bool(t.loc[i, "linked_to_main"]),
                       coel=bool(coel[i]), drt_s=t.loc[i, "drt_main_s"], in_pairs=k is not None)
            if k is not None:
                p = pairs.iloc[k]
                rec.update(r_shape=p["r_shape"], r_int=p["r_int"], relation=bool(p["relation"]), edge=bool(p["edge"]))
            rows.append(rec)
    D = pd.DataFrame(rows)
    D.to_csv(rep / f"{a.study}_main_pairs.tsv", sep="\t", index=False)
    g = D.groupby("cls")
    cls_table = pd.DataFrame({"n": g.size(), "linked": g["linked"].mean(), "coeluting": g["coel"].mean(),
                              "in_pairs_file": g["in_pairs"].mean(), "drt_med_s": g["drt_s"].median(),
                              "r_shape_med": g["r_shape"].median(), "r_shape>=0.6": g.apply(lambda x: (x["r_shape"] >= 0.6).mean()),
                              "r_int_med": g["r_int"].median(),
                              "r_int<cut": g.apply(lambda x: (x["r_int"] < info.get("int_cut", np.nan)).mean()),
                              "relation": g["relation"].mean(), "edge": g["edge"].mean()})

    rows = [summary]
    if not a.no_sweep:
        pres = pd.read_csv(a.out / "presence_mask.tsv", sep="\t", usecols=["group_id"])
        G = int(pres["group_id"].max()) + 1
        g_h = np.zeros(G)
        sample_cols = [c for c in T.columns if c not in META]
        g_h[T["group_id"].values] = T[sample_cols].max(axis=1).values
        comp_all = pd.read_csv(a.out / "compounds.tsv", sep="\t")
        child = np.zeros(G, bool)
        kept_child = T["role"].astype(str).str.contains(r" M\+\d| iso$").values
        child[T["group_id"].values[kept_child]] = True
        eligible = ~child
        gid_t = t["group_id"].values
        base_of = comp_all.set_index("compound_id")["base_group_id"]
        child_rows = np.flatnonzero((gid_t >= 0) & child[np.maximum(gid_t, 0)])
        child_base = np.array([base_of.get(int(T["compound_id"].values[t["row"].values[k]]), -1) for k in child_rows], np.int64)
        vetoes = [("none", None)] + [(q, info.get(f"int_null_{q}")) for q in ("q90", "q95", "q99")]
        for sc in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            for vname, veto in vetoes:
                if veto is not None and not (isinstance(veto, float) and np.isfinite(veto)):
                    continue
                c_all = recluster(pairs, G, g_h, eligible, sc, max(0.5, sc - 0.1), veto)
                c_t = np.where(gid_t >= 0, c_all[np.maximum(gid_t, 0)], -1)
                ok = child_base >= 0
                c_t[child_rows[ok]] = c_all[child_base[ok]]
                s2, _ = cluster_metrics(t, c_t, mains, shared, rt_win, f"shape>={sc} veto={vname}", coel)
                c_rows = np.full(len(T), -1, np.int64)
                c_rows[:] = c_all[T["group_id"].values]
                s2.update(extras(c_rows, rows_all_tiers, c_t))
                rows.append(s2)
    sweep = pd.DataFrame(rows)
    t.drop(columns=["_i"]).to_csv(rep / f"{a.study}_ions.tsv", sep="\t", index=False)
    per_std.to_csv(rep / f"{a.study}_per_standard.tsv", sep="\t", index=False)
    sweep.to_csv(rep / f"{a.study}_summary.tsv", sep="\t", index=False)
    cls_table.to_csv(rep / f"{a.study}_by_class.tsv", sep="\t")
    pd.set_option("display.width", 250)
    print(f"{a.study}: {len(t)} truth ions (tier {a.tier}), {int(hit.sum())} matched at {a.ppm} ppm / {a.rt_tol} min; "
          f"{summary['n_std_scored']} standards scored; role accuracy {summary['role_correct']:.2f} "
          f"(named {summary['role_named']:.2f}) on {summary['role_n']} formula ions; shape cut {info.get('shape_cut', float('nan')):.2f}, "
          f"int veto {'on' if info.get('int_active') else 'off'} at {info.get('int_cut', float('nan')):.2f}")
    print("main<->ion pairs by class:")
    print(cls_table.round(3).to_string())
    cols = ["label", "linked_all", "linked_coeluting", "coel_linked_formula_ion", "coel_linked_isotope", "coel_linked_fragment",
            "coel_linked_unannotated", "completeness_mean", "pair_precision", "pair_fp", "coelution_precision",
            "extras_per_compound", "size_median"]
    print(sweep[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"written: {rep}")


if __name__ == "__main__":
    main()
