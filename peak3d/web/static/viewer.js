// Results viewer: Overview, Features (table + chromatograms), Alignment and Run details for one peak3d
// output. Mounted by report.html (standalone) and by the GUI. Reads the decoded bundle only.
(function (P3) {
  "use strict";
  const { h, icon, fixed, int, compact, sci, duration } = P3;
  const S1 = "--viz-series-1", S2 = "--viz-series-2";
  const SUB = { 0: "₀", 1: "₁", 2: "₂", 3: "₃", 4: "₄", 5: "₅", 6: "₆", 7: "₇", 8: "₈", 9: "₉" };
  const sub = (t) => String(t).replace(/\d/g, (d) => SUB[d]);
  const hypName = (name) => (name === "none" ? "no halogen" : name ? sub(name) : "–");   // Cl2 -> Cl₂
  const STATUS = ["not found", "gap-filled", "gap-filled (peak confirmed)", "detected"];
  const STARS = ["red", "blue", "yellow"];          // the user's own marks for features (annotations.STARS)
  const MARK = { red: ["mark_circle", "circle"], blue: ["mark_diamond", "diamond"], yellow: ["star", "star"] };   // icon, shape name
  const markIcon = (c, on) => icon(on ? MARK[c][0] : MARK[c][0] + "_outline");
  // the MS2 copies a search scores (bundle.SPEC_CODE): consensus / highest TIC, each raw and denoised
  const SPEC = { consensus: 0, single: 1, consensus_denoised: 2, single_denoised: 3 };
  const SPEC_NAME = ["consensus", "single", "consensus_denoised", "single_denoised"];
  const specOf = (copy, dn) => (copy === "consensus" ? 0 : 1) + (dn ? 2 : 0);
  // params.json stage_times_s keys (peak3d process, then peak3d external for the other kernels) and their labels
  const STAGE_KEYS = [["pick_s", "Picking"], ["align_s", "Alignment"], ["group_s", "Grouping"], ["gapfill_s", "Gap filling"], ["qc_s", "QC figures"],
    ["kernel_s", "{k}"], ["read_s", "Reading files"], ["ms2_s", "MS2 mapping"], ["identify_s", "Identity search"]];

  // ---------------------------------------------------------------- data access
  class Results {
    constructor({ header, a }) {
      this.h = header;
      this.a = a;
      this.G = header.n_groups;
      this.stems = header.stems;
      this.N = this.stems.length;
      this.files = header.files || [];
      this.params = header.params || {};
      this.opts = this.params.options || {};
      this.quant = header.quant === "area" ? "Area" : "Height";
      this.logMax = new Float32Array(this.G);
      for (let g = 0; g < this.G; g++) this.logMax[g] = a.g_max[g] > 0 ? Math.log10(a.g_max[g]) : 0;
      this.eicAt = new Map();
      if (header.eic) a.eic_row.forEach((row, k) => this.eicAt.set(row, k));
      this.rtS = Float32Array.from(a.g_rt, (v) => v * 60);       // retention times are shown in seconds
      // compound groups (adducts, in-source fragments, multimers and their isotopes of one compound; null when the
      // results predate them) and the isotope candidates the shape test did not confirm
      this.cid = a.g_cid || null;
      this.compounds = header.compounds || null;
      this.roles = this.compounds ? this.compounds.roles : [""];
      this.isoq = a.g_isoq || null;
      this.isoqOf = a.g_isoq_of || null;
      this.members = null;                                        // cid -> rows, built on first use
      this.kernel = header.kernel || "peak3d";
      this.kernelLabel = this.kernel === "peak3d" ? "Mosa" : header.kernel_label || this.kernel;
      this.kernelVersion = header.kernel_version || this.params.peak3d_version || "";
      this.setMs2({ header, a });
    }
    // both MS2 copies of each group (the consensus of all its spectra, m2c; the highest-TIC spectrum, m2) with the
    // peaks their denoising kept (ms2c_keep, ms2_keep), where their spectra were taken (marks), the hits of the last
    // identity search on every copy, raw and denoised (every hit above 0.1 on either, so each hit has a partner on the
    // other) and the user's annotations, all by table row; a new search hands in a bundle with only these parts
    setMs2({ header, a }) {
      this.m2 = header.ms2 ? { h: header.ms2, a } : null;
      this.m2c = header.ms2 && header.ms2c ? { h: header.ms2c, a } : null;
      this.id = header.ms2 && header.ids ? { h: header.ids, a } : null;
      this.ms2At = new Map();
      this.ms2cAt = new Map();
      this.idsAt = SPEC_NAME.map(() => new Map());   // by copy (SPEC): row -> hit indices, best first
      this.marks = new Map();                   // row -> [{t (aligned s), kind: 1 consensus, 2 highest TIC, 3 both}]
      this.nms2 = new Uint16Array(this.G);
      if (this.m2) a.ms2_row.forEach((row, k) => { this.ms2At.set(row, k); this.nms2[row] = a.ms2_n[k]; });
      if (this.m2c) a.ms2c_row.forEach((row, k) => this.ms2cAt.set(row, k));
      if (this.m2 && a.mk_row) a.mk_row.forEach((row, k) => {
        const l = this.marks.get(row), m = { t: a.mk_t[k], kind: a.mk_kind[k] };
        if (l) l.push(m); else this.marks.set(row, [m]);
      });
      if (this.id) a.id_row.forEach((row, j) => {         // sorted by row, copy, rank: the first of each is the best
        const m = this.idsAt[a.id_spec ? a.id_spec[j] : 1], l = m.get(row);
        if (l) l.push(j); else m.set(row, [j]);
      });
      // which copies have a denoised version to show, and whether the search scored them. A denoising search
      // (id_dkey) denoised the spectrum against each candidate's formula: the mask comes with each hit (dk_*)
      this.dkey = this.id && a.id_dkey && a.dk_off ? a.id_dkey : null;
      this.dn = this.dkey ? { consensus: !!this.m2c, single: !!this.m2 } :
        { consensus: !!(this.m2c && a.ms2c_keep), single: !!(this.m2 && a.ms2_keep) };
      this.dnHits = !!this.id && (this.idsAt[2].size > 0 || this.idsAt[3].size > 0);
      const d = this.id && this.id.h.default_spectrum;   // the copy whose best hit is the default annotation
      this.defSpec = d != null && SPEC[d] != null ? SPEC[d] : this.m2c ? 0 : 1;
      this.minScore = this.id && this.id.h.min_score != null ? this.id.h.min_score : 0.7;
      this.setAnnotations(header.ann || {});
    }
    // the user's notes ({row: {annotation?, comment?, stars?}}; an older file's {row: annotation} is read too) and what
    // follows for every row: annotated or not, the score shown
    setAnnotations(notes) {
      this.notes = new Map();
      for (const [r, v] of Object.entries(notes || {})) if (v) this.notes.set(+r, v.status ? { annotation: v } : v);
      this.annotated = new Uint8Array(this.G);
      this.annScore = new Float32Array(this.G).fill(NaN);
      this.nAnnotated = 0;
      this.nByUser = 0;
      if (!this.m2) return;
      for (let g = 0; g < this.G; g++) {
        if (!this.nms2[g] && !this.notes.has(g)) continue;
        const x = this.annotation(g);
        if (x && !x.unknown) { this.annotated[g] = 1; this.annScore[g] = x.score; this.nAnnotated++; }
        else this.annScore[g] = this.bestScore(g);
        if (x && x.source === "user") this.nByUser++;
      }
    }
    setNotes(g, entry) {
      const all = {};
      for (const [k, v] of this.notes) all[k] = v;
      if (entry) all[g] = entry; else delete all[g];
      this.setAnnotations(all);
    }
    stars(g) { const e = this.notes.get(g); return (e && e.stars) || []; }
    role(g) { return this.cid ? this.roles[this.a.g_role[g]] || "" : ""; }
    nMembers(g) { return this.cid ? this.a.g_nmem[g] : 1; }
    // the rows of g's compound, g included, by m/z
    siblings(g) {
      if (!this.cid) return [g];
      if (!this.members) {
        this.members = new Map();
        for (let r = 0; r < this.G; r++) {
          const c = this.cid[r];
          if (c < 0) continue;
          const l = this.members.get(c);
          if (l) l.push(r); else this.members.set(c, [r]);
        }
        for (const l of this.members.values()) l.sort((p, q) => this.a.g_mz[p] - this.a.g_mz[q]);
      }
      return this.members.get(this.cid[g]) || [g];
    }
    // what the feature is within its compound: a confirmed isotope, a possible one (mass arithmetic only), a named
    // ion, the base ion, or the m/z difference to the base; null for a feature on its own
    ion(g) {
      const a = this.a, iso = a.g_iso[g], role = this.role(g), n = this.nMembers(g);
      if (iso > 0) {
        const series = / M\+\d+$/.test(role) ? role.replace(/ M\+\d+$/, "") : "";
        const named = series && series !== "base";          // "[M+Na]+ M+1" names the series; "base M+1" does not
        return { text: named ? role : `M+${iso}`, cls: "iso", title: `Isotope: M+${iso} of its monoisotopic feature, peak shape confirmed${named ? ` (${series} series)` : ""}` };
      }
      if (this.isoq && this.isoq[g] > 0) {
        const p = this.isoqOf[g];
        return { text: `M+${this.isoq[g]}?`, cls: "maybe", title: `Possible isotope: m/z and RT fit a +${this.isoq[g]} × 1.00335 step from feature #${a.g_id[p]} (m/z ${a.g_mz[p].toFixed(4)}), but the peak shapes did not confirm it` };
      }
      if (!this.cid || n <= 1) return null;
      if (role.startsWith("[")) return { text: role, cls: "adduct", title: `${role}: an ion of the feature group's neutral mass${isFinite(a.g_nm[g]) ? " " + a.g_nm[g].toFixed(4) : ""}` };
      if (role === "base") return { text: "M", cls: "base", title: "Base ion of its feature group (the strongest monoisotopic member); no mass relation names it" };
      if (/^[+-]\d/.test(role)) return { text: `Δ ${role}`, cls: "delta", title: "m/z difference to the feature group's base ion; no known adduct or fragment relation" };
      return role ? { text: role, cls: "", title: "" } : null;
    }
    comment(g) { const e = this.notes.get(g); return (e && e.comment) || ""; }
    reviewed(g) { const e = this.notes.get(g); return !!(e && e.annotation); }
    hits(g, spec) { return (this.id && this.idsAt[spec].get(g)) || []; }
    nWithMs2() { return Math.max(this.ms2At.size, this.ms2cAt.size); }
    bestScore(g) { const l = this.hits(g, this.defSpec); return l.length ? this.id.a.id_score[l[0]] : NaN; }
    // the annotation in force: the user's choice, else the best hit on the default copy at or above min_score
    annotation(g) {
      const e = this.notes.get(g), u = e && e.annotation;
      if (u) return u.status === "unknown" ? { source: "user", unknown: true } : { source: "user", confirmed: !!u.confirmed, name: u.name,
        adduct: u.precursor_type, formula: u.formula, inchikey: u.inchikey, smiles: u.smiles, score: u.score, spectrum: u.spectrum, rank: u.rank,
        lib: u.library_index != null ? u.library_index : -1 };
      const l = this.hits(g, this.defSpec);
      if (!l.length) return null;
      const b = this.hit(l[0]);
      return b.score >= this.minScore ? Object.assign({ source: "library" }, b) : null;
    }
    // are the two copies of row g one and the same spectrum (one MS2, or a consensus of one that is the highest-TIC
    // spectrum)? Then the viewer shows one
    oneSpectrum(g) {
      const kc = this.ms2cAt.get(g), ks = this.ms2At.get(g);
      if (kc == null || ks == null) return true;
      if (this.nms2[g] <= 1) return true;
      return this.m2c.a.ms2c_merged[kc] === 1 && this.m2c.h.file[kc] === this.m2.h.file[ks] && this.m2c.h.scan[kc] === this.m2.h.scan[ks];
    }
    // one copy's spectrum: which = "consensus" (k into m2c) or "single" (k into m2); dn: only the peaks the denoising
    // kept (in a denoising search those of hit j's mask: the whole spectrum when j has none)
    spectrum(which, k, dn, j) {
      const c = which === "consensus", a = (c ? this.m2c : this.m2).a, p = c ? "ms2c_" : "ms2_";
      const o0 = a[p + "off"][k], o1 = a[p + "off"][k + 1];
      const mz = a[p + "mz"].subarray(o0, o1), inten = a[p + "int"].subarray(o0, o1), keep = dn ? this.mask(which, k, j) : null;
      if (!keep) return { mz, inten };
      const m = [], v = [];
      for (let i = 0; i < o1 - o0; i++) if (keep[i]) { m.push(mz[i]); v.push(inten[i]); }
      return { mz: Float32Array.from(m), inten: Float32Array.from(v) };
    }
    // the denoising's keep flags over one copy's spectrum, or null: the per-spectrum denoised copy's, or in a
    // denoising search hit j's (null when j has none: its record gave no usable formula)
    mask(which, k, j) {
      const c = which === "consensus", a = (c ? this.m2c : this.m2).a, p = c ? "ms2c_" : "ms2_";
      if (this.dkey) {
        const d = j != null ? this.dkey[j] : -1;
        if (d < 0) return null;
        const ia = this.id.a;
        return ia.dk_keep.subarray(ia.dk_off[d], ia.dk_off[d + 1]);
      }
      const keep = a[p + "keep"];
      return keep ? keep.subarray(a[p + "off"][k], a[p + "off"][k + 1]) : null;
    }
    // [peaks kept by the denoising, peaks] of one copy's spectrum (null when it has no denoised version; j: the hit
    // whose mask counts, in a denoising search)
    kept(which, k, j) {
      const c = which === "consensus", a = (c ? this.m2c : this.m2).a, p = c ? "ms2c_" : "ms2_", keep = this.mask(which, k, j);
      if (!keep) return null;
      let n = 0;
      for (let i = 0; i < keep.length; i++) n += keep[i];
      return [n, a[p + "off"][k + 1] - a[p + "off"][k]];
    }
    // the same library record among row g's hits on copy spec (the other score of a hit), or null
    partner(g, j, spec) {
      if (j == null || !this.id || !this.id.a.id_lib) return null;
      const lib = this.id.a.id_lib[j];
      if (lib < 0) return null;
      for (const t of this.hits(g, spec)) if (this.id.a.id_lib[t] === lib) return t;
      return null;
    }
    hit(j) {
      const a = this.id.a, h = this.id.h, k = a.id_lk ? a.id_lk[j] : j;     // id_lk: each library spectrum is sent once
      const spec = h.spectra ? { mz: a.id_mz.subarray(a.id_off[k], a.id_off[k + 1]), inten: a.id_int.subarray(a.id_off[k], a.id_off[k + 1]) } : null;
      return { j, rank: a.id_rank[j], score: a.id_score[j], nmatch: a.id_nmatch[j], ppm: a.id_ppm[j], prec: a.id_prec[j],
        spectrum: SPEC_NAME[a.id_spec ? a.id_spec[j] : 1], lib: a.id_lib ? a.id_lib[j] : -1,
        name: h.name[j], adduct: h.precursor_type[j], formula: h.formula[j], inchikey: h.inchikey[j], smiles: h.smiles[j],
        dion: h.denoise_ion ? h.denoise_ion[j] : "", spec };
    }
    value(g, f) {
      const q = this.a.m_q[g * this.N + f];
      if (!q) return 0;
      const m = this.h.matrix;
      return Math.pow(10, m.log10_lo + (q - 1) / 65534 * (m.log10_hi - m.log10_lo));
    }
    status(g, f) { return this.a.m_status ? this.a.m_status[g * this.N + f] : null; }
    // the run's halogen check: 0 not tested, 1 no, 2 possible, 3 halogen; and the best hypothesis's name
    hal(g) { return this.a.g_hal ? this.a.g_hal[g] : 0; }
    halHint(g) { return this.a.g_halhint && this.h.halogen_names ? this.h.halogen_names[this.a.g_halhint[g]] || "" : ""; }
    eic(g) {
      const k = this.eicAt.get(g);
      if (k == null) return null;
      const e = this.h.eic, P = e.points, F = e.files.length, a = this.a;
      const x = new Float32Array(P);
      for (let j = 0; j < P; j++) x[j] = 60 * (a.eic_x0[k] + j * a.eic_dx[k]);
      const traces = e.files.map((f, j) => {
        const y = new Float32Array(P), off = (k * F + j) * P, sc = a.eic_scale[k] / 65535;
        for (let p = 0; p < P; p++) y[p] = a.eic_q[off + p] * sc;
        // MS1 scans of this file inside the peak window, and those holding the ion (older bundles: unknown)
        const ns = a.eic_ns ? { all: a.eic_ns[(k * F + j) * 2], signal: a.eic_ns[(k * F + j) * 2 + 1] } : null;
        // the m/z window of this file's trace, ppm below / above the feature (older bundles: +/- the run's ppm)
        const lo = a.eic_lo ? a.eic_lo[k * F + j] : e.ppm, hi = a.eic_hi ? a.eic_hi[k * F + j] : e.ppm;
        return { file: f, y, ns, lo, hi };
      });
      return { x, traces };
    }
    medianFeatures() {
      const v = this.files.map((f) => f.n_features).filter((x) => x != null).sort((p, q) => p - q);
      return v.length ? { med: v[v.length >> 1], lo: v[0], hi: v[v.length - 1] } : null;
    }
  }

  // ---------------------------------------------------------------- small pieces
  const card = (cls, title, sub, ...body) => h("section.card" + cls, title ? h("div.card-head", h("h3.title-medium", title)) : null, sub ? h("p.card-sub.body-small", sub) : null, ...body);
  const tile = (label, value, note) => h("div.tile", h("div.label", label), h("div.value.num", value), note ? h("div.note", note) : null);
  const kv = (pairs) => h("dl.kv.body-medium", pairs.filter((p) => p && p[1] != null && p[1] !== "").flatMap(([k, v]) => [h("dt", k), h("dd", v)]));
  const banner = (text, kind) => h("div.banner" + (kind === "error" ? ".error" : ""), icon(kind === "error" ? "error" : "info"), h("div.body-medium", text));
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const base = (p) => String(p || "").split(/[\\/]/).filter(Boolean).pop() || "";
  // a labelled box like the GUI's form fields (label above), for the identity search controls
  const field = (label, control, trail, width) => h("div.field", { style: width ? { width } : null }, h("div.box", control, h("label", label), trail || null));

  // the listed features as a tab-separated table, from the values stored in the bundle (rounded to 0.04 %); retention
  // times in seconds, the files whose value was gap-filled (after n_filled, when the bundle tells detected cells from
  // filled ones), plus the MS2 count and best library hit when the results have them
  const TABLE_COLS = ["group_id", "mz", "rt_s", "rt_min_s", "rt_max_s", "n_detected", "n_confirmed", "n_filled", "mz_ppm_spread",
    "rt_sd_s", "iso_offset", "charge"];
  const COMPOUND_COLS = ["compound_id", "role", "neutral_mass", "is_base", "n_members", "possible_isotope"];
  const HAL_COLS = ["possible_halogen", "halogen_pattern"];
  const FILLED_COL = "gap_filled_files";
  const HAL_TEXT = ["", "no", "maybe", "yes"];
  const MS2_COLS = ["n_ms2", "n_consensus", "annotation", "annotation_source", "reviewed", "annotation_score", "annotation_adduct",
    "annotation_formula", "annotation_inchikey", "best_hit", "best_score", "stars", "comment"];
  // the RT correction's warnings a file can carry (its flag in rt_correction/summary.tsv): the app lets the user leave
  // such files out of the downloaded tables and the report
  const FILE_WARN = [
    ["few_anchors_matched", "few anchors matched", "matched fewer than 80 % of the typical file's RT anchors, so their RT correction rests on little evidence"],
    ["rt_residual_high", "RT not corrected", "still have features far from the other files after RT correction (at least 4 x the study's median): their retention changed in a way the correction could not follow"],
  ];
  const hasFlag = (f, k) => String((f && f.flag) || "").split(";").includes(k);
  function bundleTable(R, rows) {
    const a = R.a, num = (v, d) => (isFinite(v) ? v.toFixed(d) : ""), clean = (t) => String(t || "").replace(/[\t\r\n]+/g, " ");
    const notes = !!(R.m2 || R.notes.size);
    // gap-filled: a cell with a value whose status is not "detected" (1 filled, 2 filled with a confirmed peak)
    const fill = !!(a.m_q && a.m_status && R.h.matrix && R.h.matrix.presence), N = R.N, at = TABLE_COLS.indexOf("n_filled") + 1;
    const filledFiles = (g) => {
      const l = [];
      for (let f = 0; f < N; f++) { const st = a.m_status[g * N + f]; if ((st === 1 || st === 2) && a.m_q[g * N + f]) l.push(R.stems[f]); }
      return l.join(";");
    };
    const head = TABLE_COLS.slice();
    if (fill) head.splice(at, 0, FILLED_COL);
    const parts = [head.concat(R.cid ? COMPOUND_COLS : [], a.g_hal ? HAL_COLS : [], notes ? MS2_COLS : [], a.m_q ? R.stems : []).join("\t") + "\n"];
    let buf = [];
    for (const g of rows) {
      const row = [a.g_id[g], a.g_mz[g].toFixed(6), num(60 * a.g_rt[g], 2), num(60 * a.g_rt_min[g], 2), num(60 * a.g_rt_max[g], 2),
        a.g_ndet[g], a.g_nconf[g], a.g_nfill[g], num(a.g_ppm[g], 3), num(60 * a.g_rtsd[g], 3), a.g_iso[g], a.g_z[g]];
      if (fill) row.splice(at, 0, filledFiles(g));
      if (R.cid) row.push(a.g_cid[g], R.role(g), num(a.g_nm[g], 6), a.g_base[g], a.g_nmem[g], R.isoq && R.isoq[g] ? `M+${R.isoq[g]}` : "");
      if (a.g_hal) row.push(HAL_TEXT[R.hal(g)] || "", R.halHint(g));
      if (notes) {
        const x = R.m2 ? R.annotation(g) : null, kc = R.ms2cAt.get(g), l = R.hits(g, R.defSpec), b = l.length ? R.hit(l[0]) : null;
        const named = x && !x.unknown;
        row.push(R.nms2[g], kc != null ? R.m2c.a.ms2c_merged[kc] : 0, x ? (x.unknown ? "unknown" : clean(x.name)) : "", x ? x.source : "",
          R.reviewed(g) ? "yes" : "no", named ? num(x.score, 4) : "", named ? clean(x.adduct) : "", named ? clean(x.formula) : "",
          named ? clean(x.inchikey) : "", b ? clean(b.name) : "", b ? b.score.toFixed(4) : "", R.stars(g).join(";"), clean(R.comment(g)));
      }
      if (a.m_q) for (let f = 0; f < R.N; f++) { const v = R.value(g, f); row.push(v ? String(+v.toPrecision(5)) : "0"); }
      buf.push(row.join("\t"));
      if (buf.length >= 2000) { parts.push(buf.join("\n") + "\n"); buf = []; }
    }
    if (buf.length) parts.push(buf.join("\n") + "\n");
    return new Blob(parts, { type: "text/tab-separated-values" });
  }

  // the identity-search candidates of the listed features that have MS2, as a JSON array of MassWiki search records
  // (the "data" of each row of a MassWiki job database): the same keys in the same order, the same value types, ""
  // and null where MassWiki has no value, and empty lists for the searches Mosa does not run (open, neutral loss,
  // hybrid, the annotation, experimental and internal libraries). spectrum is the query: the copy the annotations
  // come from (the consensus or the highest-TIC spectrum) as measured, peaks_clean that cleaned as for scoring
  // (cleanPeaks), its spectral entropy, the feature's RT in seconds; a feature with an annotation in force carries it
  // as its name, formula, InChIKey and SMILES (searchmode identity_search), as a MassWiki upload with known spectra
  // does. reference_library.identity_search: every library spectrum the last search listed on the annotation copy,
  // best first, at most 100 (MassWiki's cap), entropy_similarity the score there (the search's own similarity,
  // weighted entropy unless chosen otherwise); id and index the record's position in the library file, db its name.
  // Returns { blob, n features, k candidates }; one record per line
  const MW_MAX = 100;
  const PRECURSOR_MARGIN = 1.6, NOISE = 0.01, MIN_CENTROID_DA = 0.05;     // similarity.py
  // similarity.clean (as ms_entropy's, which MassWiki builds on): peaks from 1.6 Da below the precursor up dropped,
  // peaks within max(0.05, 2 x tol) Da merged strongest first, those under 1 % of the strongest dropped, intensities
  // summing to 1; m/z ascending
  function cleanPeaks(mz, inten, prec, tol) {
    const cut = prec > 0 ? prec - PRECURSOR_MARGIN : Infinity;
    let pk = [];
    for (let i = 0; i < mz.length; i++) if (inten[i] > 0 && isFinite(inten[i]) && isFinite(mz[i]) && mz[i] < cut) pk.push([+mz[i], +inten[i]]);
    pk.sort((p, q) => p[0] - q[0]);
    const d = Math.max(MIN_CENTROID_DA, 2 * tol);
    while (pk.length > 1 && pk.some((p, i) => i > 0 && p[0] - pk[i - 1][0] <= d)) {
      const work = pk.map((p) => p[1]), out = [];
      for (const idx of work.map((_, i) => i).sort((a, b) => work[b] - work[a])) {
        if (work[idx] <= 0) continue;
        let lo = idx - 1, hi = idx + 1, s = 0, sw = 0;
        while (lo >= 0 && pk[idx][0] - pk[lo][0] <= d) lo--;
        while (hi < pk.length && pk[hi][0] - pk[idx][0] <= d) hi++;
        for (let j = lo + 1; j < hi; j++) { s += work[j]; sw += work[j] * pk[j][0]; work[j] = 0; }
        out.push([sw / s, s]);
      }
      pk = out.sort((p, q) => p[0] - q[0]);
    }
    if (!pk.length) return pk;
    const top = Math.max(...pk.map((p) => p[1]));
    pk = pk.filter((p) => p[1] >= NOISE * top);
    const sum = pk.reduce((s, p) => s + p[1], 0);
    return pk.map(([m, v]) => [m, v / sum]);
  }
  // JSON writes 1.0 as 1, which Python reads back as an int; the fields MassWiki holds as floats keep their ".0"
  const FLOAT_MARK = "⁣f";
  const fl = (v) => (Number.isInteger(v) ? FLOAT_MARK + v : v);
  const FLOAT_RE = new RegExp(`"${FLOAT_MARK}(-?\\d+)"`, "g");
  function candidatesJson(R, rows) {
    const a = R.a, ih = R.id.h, ia = R.id.a;
    const txt = (s) => (s == null ? "" : String(s));
    const none = (s) => txt(s) || "None";
    const tol = ih.fragment_tol || 0.02;
    const pols = new Set([R.opts.polarity, ...R.files.map((f) => f.polarity)].filter(Boolean).map((p) => (/^(pos|\+)/i.test(p) ? 1 : /^(neg|-)/i.test(p) ? -1 : 0)));
    const runSign = pols.size === 1 ? [...pols][0] : 0;
    const signOf = (adduct) => (/[+]$/.test(adduct) ? 1 : /-$/.test(adduct) ? -1 : 0);
    const db = base(ih.library).replace(/\.(msp|pkl)$/i, "");
    const lines = [];
    let k = 0;
    for (const g of rows) {
      let spec = R.defSpec, which = spec % 2 ? "single" : "consensus", kk = (which === "single" ? R.ms2At : R.ms2cAt).get(g);
      if (kk == null) {                         // a feature without that copy: the other one
        spec ^= 1; which = spec % 2 ? "single" : "consensus"; kk = (which === "single" ? R.ms2At : R.ms2cAt).get(g);
      }
      if (kk == null) continue;
      const m = (which === "single" ? R.m2 : R.m2c).a, prec = m[(which === "single" ? "ms2_" : "ms2c_") + "prec"][kk];
      const sp = R.spectrum(which, kk, false);
      const clean = cleanPeaks(sp.mz, sp.inten, prec, tol);
      let entropy = 0;
      for (const [, v] of clean) if (v > 0) entropy -= v * Math.log(v);
      const hits = R.hits(g, spec).slice(0, MW_MAX).map((j) => {
        const b = R.hit(j);
        return { db, id: String(b.lib), name: txt(b.name), adduct: txt(b.adduct), rt: "", precursor_mz: isFinite(b.prec) ? String(b.prec) : "",
          smiles: txt(b.smiles), user_name: "", spec_uid: "", predicted_rt_hilic: "", predicted_rt_rp: "", rt_confidence_level_hilic: "",
          rt_confidence_level_rp: "", anno_delta_rt: "", splash: "", index: b.lib, entropy_similarity: fl(+(+b.score).toFixed(4)),
          library_wiki_id: "", delta_predicted_rt: null, rt_confidence_level: null, confidence_in_domain: false, confidence_pct: null,
          ensemble_sd_pct: null };
      });
      k += hits.length;
      const x = R.annotation(g), named = !!(x && !x.unknown && x.name);
      const role = R.role(g);
      const sign = runSign || (named ? signOf(txt(x.adduct)) : 0) || (hits.length ? signOf(hits[0].adduct) : 0) || 1;
      const z = a.g_z[g] > 0 ? a.g_z[g] : 1;
      const ptype = (named && txt(x.adduct)) || (/^\[.+\]\d*[+-]$/.test(role) ? role : "") ||
        (z > 1 ? `[M${sign > 0 ? "+" : "-"}${z}H]${z}${sign > 0 ? "+" : "-"}` : sign > 0 ? "[M+H]+" : "[M-H]-");
      const gid = a.g_id[g], rt = +R.rtS[g].toFixed(2);
      const rec = {
        analysis: {
          reference_library: { identity_search: hits, open_search: [], neutral_loss_search: [], hybrid_search: [] },
          experimental_library: { similar_spectra: [], cluster_peaks: [] },
          annotation_library: { identity_search: [], open_search: [], neutral_loss_search: [], hybrid_search: [] },
          internal_library: { rt_similar: [], ms2_similar: [] },
        },
        spectrum: { id: String(gid), name: named ? txt(x.name) : "Unknown", searchmode: named ? "identity_search" : "None",
          formula: named ? none(x.formula) : "None", inchikey: named ? none(x.inchikey) : "None", smiles: named ? none(x.smiles) : "None",
          precursor_type: ptype, charge: sign * z, scan: gid, precursor_mz: fl(prec), peaks_clean: clean.map(([u, v]) => [fl(u), fl(v)]),
          peaks_source: "original", entropy, peaks: Array.from(sp.mz, (u, i) => [fl(u), fl(sp.inten[i])]), rt_normalized: fl(rt), rt: fl(rt),
          spec_uid: gid },
        wiki_id: `${R.h.title}/${gid}`,
        status: "success",
      };
      lines.push(JSON.stringify(rec).replace(FLOAT_RE, "$1.0"));
    }
    const parts = ["[\n"];
    for (let i = 0; i < lines.length; i += 2000) parts.push(lines.slice(i, i + 2000).join(",\n") + (i + 2000 < lines.length ? ",\n" : "\n"));
    parts.push("]\n");
    return { blob: new Blob(parts, { type: "application/json" }), n: lines.length, k };
  }

  // ---------------------------------------------------------------- viewer
  function mount(root, bundle, opts) {
    opts = opts || {};
    const R = new Results(bundle);
    const charts = [];
    const state = { tab: "overview", sel: null, sortKey: "max", sortDir: -1, query: "", mono: false, all: false, withEic: false, ppm: 10,
      mzLo: "", mzHi: "", rtLo: "", rtHi: "", withMs2: false, identified: false, halogen: false, hit: null, autoSearched: false, copy: null,
      starF: { red: false, blue: false, yellow: false }, cidF: null, skip: new Set(opts.skip || []) };
    let detailRefs = null, repaintRows = null;    // the Features page's detail parts and table, while it is shown
    let showSiblingsOf = null, clearCidF = null;  // the Features page's compound filter (list one compound's features)
    const track = (c) => { charts.push(c); return c; };
    const clearCharts = () => { while (charts.length) charts.pop().destroy(); };

    root.replaceChildren();
    root.classList.add("viewer");
    const tabsDef = [["overview", "insights", "Overview"], ["features", "table_rows", "Features"], ["alignment", "timeline", "Alignment"], ["details", "info", "Run details"]];
    const tabs = h("div.tabs", { role: "tablist" }, tabsDef.map(([id, ic, label]) =>
      h("button.tab.state", { type: "button", role: "tab", "data-tab": id, "aria-selected": "false", onclick: () => show(id) }, icon(ic), label)));
    const body = h("div.page");
    root.append(h("div.tabbar", tabs), body);

    function show(id) {
      state.tab = id;
      detailRefs = null;
      repaintRows = null;
      showSiblingsOf = clearCidF = null;
      body.className = id === "features" ? "page wide" : "page";
      headEl = annoEl = shown = null;
      chromCharts.length = 0;              // their charts are among those clearCharts destroys now
      annoCharts.length = 0;
      ms1Charts.length = 0;
      tabs.querySelectorAll(".tab").forEach((t) => t.setAttribute("aria-selected", String(t.dataset.tab === id)));
      clearCharts();
      body.replaceChildren();
      ({ overview, features, alignment, details })[id]();
      if (opts.onTab) opts.onTab(id);
    }

    // ---------------------------------------------------------------- overview
    function overview() {
      const st = R.params.stage_times_s || {};
      const mf = R.medianFeatures();
      const res = R.h.residuals;
      const total = STAGE_KEYS.reduce((s, [k]) => s + (st[k] || 0), 0);
      // a report the app built without the flagged files the user left out
      const ex = R.h.excluded;
      if (ex && ex.files && ex.files.length) {
        const kinds = FILE_WARN.filter(([k]) => ex.files.some((f) => hasFlag(f, k))).map(([, label]) => label);
        const names = ex.files.slice(0, 10).map((f) => f.stem).join(", ") + (ex.files.length > 10 ? ` and ${int(ex.files.length - 10)} more` : "");
        const gone = Math.max(0, ex.n_groups_before - R.G);
        body.append(banner(`${int(ex.files.length)} file${ex.files.length > 1 ? "s" : ""} left out of this report` +
          `${kinds.length ? " (" + kinds.join(", ") + ")" : ""}: ${names}. Their columns${gone ? ` and the ${int(gone)} feature${gone > 1 ? "s" : ""} only they held` : ""} ` +
          `are not here; detection counts are over the other ${int(R.N)} files.`));
      }
      body.append(h("div.tiles",
        tile("Files", int(R.N), R.params.input_files ? "centroided MS1 mzML" : null),
        tile("Aligned features", int(R.G), st.n_groups_all ? `${int(st.n_groups_all)} groups before the presence rule` : null),
        R.compounds ? tile("Feature groups", int(R.compounds.n_multi), `features of one compound (adducts, fragments, isotopes) grouped together, holding ${int(R.compounds.n_in_multi)} features; ${int(R.G - R.compounds.n_in_multi)} features on their own`) : null,
        tile("Features per file", mf ? int(mf.med) : "–", mf && R.N > 1 ? `median; range ${int(mf.lo)}–${int(mf.hi)}` : "picked, before alignment"),
        tile("RT alignment", res ? `${fixed(res.mad_after, 2)} s` : R.N > 1 ? "off" : "n/a", res ? `anchor MAD, ${fixed(res.mad_before, 2)} s before correction` : "single file or correction disabled"),
        tile("Processing time", total ? duration(total) : "–", `${R.kernelLabel} ${R.kernelVersion}`),
        R.m2 ? tile("With MS2", int(R.m2.h.n), R.id ? `${int(R.nAnnotated)} annotated${R.nByUser ? ` (${int(R.nByUser)} by you)` : ""} · ${R.id.h.method_label || R.id.h.method || "library search"}` : "no identity search yet") : null));

      const mapEl = h("div");
      const mapCard = card(".elevated", "Feature map", "Every aligned feature at its m/z and retention time; colour is its largest " + R.quant.toLowerCase() + " across files. Click a point to inspect it.", mapEl,
        P3.plot.legend([{ kind: "ramp", from: "--viz-ramp-from", to: "--viz-ramp-to", left: "lower", right: "higher " + R.quant.toLowerCase() }]));
      const detEl = h("div"), perEl = h("div");
      const side = h("div.stack",
        card(".elevated", "Detection across files", "How many files each feature was picked in (before gap filling).", detEl),
        card(".elevated", "Features per file", "Features the picker reported in each file, in file order.", perEl));
      body.append(h("div.grid.span-main", mapCard, side));

      let lo = Infinity, hi = -Infinity;
      for (let g = 0; g < R.G; g++) if (R.a.g_max[g] > 0) { lo = Math.min(lo, R.logMax[g]); hi = Math.max(hi, R.logMax[g]); }
      track(P3.plot.scatter(mapEl, {
        height: 460, x: R.rtS, y: R.a.g_mz, v: R.logMax, vDomain: [lo, hi], xLabel: "Retention time (s)", yLabel: "m/z",
        xFmt: (v, d) => fixed(v, Math.min(d, 1)), yFmt: (v, d) => fixed(v, Math.min(d, 1)), selected: state.sel,
        tip: (g) => `<b>m/z ${fixed(R.a.g_mz[g], 4)}</b><br>RT ${fixed(R.rtS[g], 1)} s · max ${sci(R.a.g_max[g])}<br>detected in ${R.a.g_ndet[g]} of ${R.N} files` +
          (R.annotated[g] ? `<br>${esc(R.annotation(g).name)}` : ""),
        onClick: (g) => { state.sel = g; show("features"); },
      }));
      const counts = new Array(Math.max(1, R.N)).fill(0);
      for (let g = 0; g < R.G; g++) { const k = R.a.g_ndet[g]; if (k >= 1 && k <= R.N) counts[k - 1]++; }
      track(P3.plot.bars(detEl, {
        height: 180, values: counts, color: S1, xLabel: "Files detected in", yLabel: "Features",
        xTick: (i) => String(i + 1), tip: (i) => `<b>${int(counts[i])}</b> features detected in ${i + 1} file${i ? "s" : ""}`,
      }));
      const nf = R.files.map((f) => f.n_features || 0);
      track(P3.plot.bars(perEl, {
        height: 180, values: nf, color: S1, xLabel: "File", yLabel: "Features",
        xTick: (i) => String(i + 1), tip: (i) => `<b>${esc(R.stems[i])}</b><br>${int(nf[i])} features`,
      }));
    }

    // ---------------------------------------------------------------- features
    function rows() {
      const a = R.a, out = [];
      const q = state.query.trim();
      let mzq = null, idq = null;
      if (/^#\d+$/.test(q)) idq = +q.slice(1);
      else if (q && isFinite(+q)) mzq = +q;
      const bound = (v, d) => (v !== "" && isFinite(+v) ? +v : d);
      const mzLo = bound(state.mzLo, -Infinity), mzHi = bound(state.mzHi, Infinity);
      const rtLo = bound(state.rtLo, -Infinity), rtHi = bound(state.rtHi, Infinity);
      const starOn = STARS.some((c) => state.starF[c]);
      const onlyCid = state.cidF != null && R.cid;
      for (let g = 0; g < R.G; g++) {
        if (onlyCid) { if (R.cid[g] === state.cidF) out.push(g); continue; }
        if (a.g_mz[g] < mzLo || a.g_mz[g] > mzHi || R.rtS[g] < rtLo || R.rtS[g] > rtHi) continue;
        if (state.mono && a.g_iso[g] > 0) continue;
        if (state.all && a.g_ndet[g] < R.N) continue;
        if (state.withEic && !R.eicAt.has(g)) continue;
        if (state.withMs2 && !R.nms2[g]) continue;
        if (state.identified && !R.annotated[g]) continue;
        if (state.halogen && R.hal(g) < 2) continue;
        if (starOn && !R.stars(g).some((c) => state.starF[c])) continue;
        if (idq != null && a.g_id[g] !== idq) continue;
        if (mzq != null && Math.abs(a.g_mz[g] - mzq) / mzq * 1e6 > state.ppm) continue;
        out.push(g);
      }
      const key = {
        id: a.g_id, mz: a.g_mz, rt: a.g_rt, ndet: a.g_ndet, nfill: a.g_nfill, max: a.g_max, cv: a.g_cv, iso: a.g_iso,
        ms2: R.nms2, score: R.annScore, cmp: R.cid ? a.g_nmem : null, hal: a.g_hal,
      }[state.sortKey];
      if (mzq != null) out.sort((p, r) => Math.abs(a.g_mz[p] - mzq) - Math.abs(a.g_mz[r] - mzq));
      else if (state.sortKey === "ident") {         // annotated first, by name; then the rest by best score
        const name = (g) => (R.annotated[g] ? String(R.annotation(g).name).toLowerCase() : null);
        out.sort((p, r) => { const x = name(p), y = name(r); return x == null ? (y == null ? (R.annScore[r] || 0) - (R.annScore[p] || 0) : 1) : y == null ? -1 : state.sortDir * x.localeCompare(y); });
      } else if (state.sortKey === "ion") {          // by compound, then m/z: a compound's ions stay together
        const c = (g) => (R.cid ? R.cid[g] : 0);
        out.sort((p, r) => state.sortDir * ((c(p) - c(r)) || (a.g_mz[p] - a.g_mz[r])));
      } else if (state.sortKey === "width") out.sort((p, r) => state.sortDir * ((a.g_rt_max[p] - a.g_rt_min[p]) - (a.g_rt_max[r] - a.g_rt_min[r])));
      else if (key) out.sort((p, r) => { const x = key[p], y = key[r]; return isNaN(x) ? 1 : isNaN(y) ? -1 : state.sortDir * (x - y); });
      return out;
    }

    function features() {
      const a = R.a;
      const search = h("input", { type: "search", placeholder: "Search m/z or #id", value: state.query, "aria-label": "Search m/z or #group id" });
      const searchField = h("div.field.dense", h("div.box", h("span.lead", icon("search", "sm")), search, h("label", "m/z or #id")));
      const ppmIn = h("input", { type: "number", placeholder: "ppm", min: "0.1", step: "1", value: String(state.ppm), title: "m/z search tolerance (ppm)" });
      const ppmField = h("div.field.dense", { style: { width: "84px" } }, h("div.box", ppmIn, h("span.trail.body-small", { style: { paddingRight: "8px" } }, "ppm"), h("label", "m/z tolerance (ppm)")));
      const chip = (key, label) => {
        const c = h("button.chip.state", { type: "button", "aria-pressed": String(state[key]) }, state[key] ? icon("check") : null, label);
        c.addEventListener("click", () => { state[key] = !state[key]; features_refresh(); c.setAttribute("aria-pressed", String(state[key])); c.replaceChildren(...(state[key] ? [icon("check")] : []), label); });
        return c;
      };
      const count = h("span.body-small.muted");
      // the compound filter: while it is on, the list holds one compound's features and nothing else applies
      const cidBox = h("span.cid-filter");
      function showSiblings(g) {
        state.cidF = R.cid[g];
        scroller.scrollTop = 0;
        features_refresh();
        if (state.sel !== g) select(g); else renderHead(g);
      }
      function clearCid() {
        state.cidF = null;
        features_refresh();
        if (state.sel != null) renderHead(state.sel);
      }
      // one chip per star colour: the features carrying any of the chosen stars
      const starChip = (c) => {
        const b = h("button.chip.state.star-chip", { type: "button", "aria-pressed": String(state.starF[c]), title: `Only features with the ${c} ${MARK[c][1]}` },
          h("span.star-ic." + c, markIcon(c, true)), c[0].toUpperCase() + c.slice(1));
        b.addEventListener("click", () => { state.starF[c] = !state.starF[c]; b.setAttribute("aria-pressed", String(state.starF[c])); scroller.scrollTop = 0; features_refresh(); });
        return b;
      };
      const dl = h("button.btn.outlined.state", { type: "button", title: opts.exportTable ?
        "The features listed here (search, filters and sort applied), with the values from aligned_feature_table.tsv; retention times in seconds; gap_filled_files names the files whose value was gap-filled" :
        "The features listed here (search, filters and sort applied); retention times in seconds; gap_filled_files names the files whose value was gap-filled. Values are rounded to 0.04 %; the exact table is aligned_feature_table.tsv in the output folder (minutes)." },
        icon("download"), "Download table");
      dl.addEventListener("click", async () => {
        const skip = opts.exportTable ? [...state.skip] : [];
        const name = `${R.h.title}_features${list.length < R.G ? "_" + list.length : ""}${skip.length ? "_without_flagged_files" : ""}.tsv`;
        dl.disabled = true;
        try {
          const blob = opts.exportTable ? await opts.exportTable(list.map((g) => R.a.g_id[g]), skip) : bundleTable(R, list);
          P3.saveBlob(blob, name);
          // with files left out, a feature only they held is not in the table
          const n = skip.length ? Math.max(0, (await blob.text()).split("\n").filter(Boolean).length - 1) : list.length;
          P3.snack(`Downloaded ${int(n)} features as ${name}` + (n < list.length ? ` (${int(list.length - n)} held only by the files left out)` : ""));
        } catch (e) {
          P3.snack(String(e.message || e));
        } finally {
          dl.disabled = false;
        }
      });
      // the library candidates of the listed features as JSON (candidatesJson), from the viewer's own data: the app and
      // the report alike
      const dlc = R.m2 ? h("button.btn.outlined.state", { type: "button", title: "The identity-search candidates of the features listed here " +
        "(search, filters and sort applied) that have MS2, as a JSON array of MassWiki search records: per feature its MS2 spectrum and RT (s), " +
        "and every library spectrum the search listed (score above 0.1, best first, at most 100), those below the annotation threshold included" },
        icon("download"), "Download candidates") : null;
      if (dlc) dlc.addEventListener("click", () => {
        if (!R.id) return P3.snack("No identity search yet: choose a spectral library in the Identity search panel.");
        try {
          const out = candidatesJson(R, list);
          if (!out.n) return P3.snack("None of the listed features has an MS2 spectrum.");
          const name = `${R.h.title}_candidates${out.n < R.nWithMs2() ? "_" + out.n : ""}.json`;
          P3.saveBlob(out.blob, name);
          P3.snack(`Downloaded ${int(out.n)} features with ${int(out.k)} candidates as ${name}`);
        } catch (e) {
          P3.snack(String(e.message || e));
        }
      });
      // the chromatogram of any m/z in chosen files (needs the app's server: a report has no button)
      const eicBtn = opts.eic ? h("button.btn.outlined.state", { type: "button", onclick: () => eicDialog(),
        title: "The whole-run chromatogram of an m/z (the selected feature's, or one you type) in the files you choose; zoom in and out" },
        icon("show_chart"), "Extract EIC") : null;
      // the app only: whether the files the RT correction flagged go into the downloaded tables and the report
      // (Download table here; the feature tables and the report in the Download menu, and Open report)
      const skipBoxes = opts.exportTable ? FILE_WARN.map(([k, label, why]) => {
        const n = R.files.filter((f) => hasFlag(f, k)).length;
        if (!n) return null;
        const box = h("input", { type: "checkbox", checked: !state.skip.has(k) });
        box.addEventListener("change", () => {
          if (box.checked) state.skip.delete(k); else state.skip.add(k);
          if (opts.onSkip) opts.onSkip([...state.skip]);
        });
        return h("label.skip-check", { title: `${int(n)} file${n > 1 ? "s" : ""} ${why} (Alignment tab). Untick to leave them out of ` +
          "the downloaded tables and the report: their columns go, and so do the features only they held; detection counts are over the other files. " +
          "This list keeps every file." }, box, `${int(n)} file${n > 1 ? "s" : ""} with ${label}`);
      }).filter(Boolean) : [];
      const skipRow = skipBoxes.length ? h("div.toolbar.skip-files", h("span.body-small.muted", "In downloaded tables and the report, include"), ...skipBoxes) : null;
      // m/z and retention-time windows (either end may stay open); the mini map's box fills them too
      const rangeIn = {};
      const range = (label, lo, hi, unit) => {
        const box = (key, ph) => {
          const i = rangeIn[key] = h("input", { type: "number", step: "any", placeholder: ph, value: state[key], "aria-label": `${label} ${ph}` });
          i.addEventListener("input", () => { state[key] = i.value.trim(); clearTimeout(tmr); tmr = setTimeout(() => { scroller.scrollTop = 0; features_refresh(); }, 150); });
          return i;
        };
        return h("div.range", { title: `${label} window${unit ? " (" + unit + ")" : ""}; leave an end empty to keep it open` },
          h("span.range-label", label), box(lo, "from"), h("span.range-sep", "–"), box(hi, "to"), unit ? h("span.range-unit", unit) : null);
      };
      const idPanel = identityPanel();
      // mini map: every feature at its RT and m/z, the listed ones in colour; drag a box to set both windows
      const mapEl = h("div.minimap");
      let lo = Infinity, hi = -Infinity;
      for (let g = 0; g < R.G; g++) if (a.g_max[g] > 0) { lo = Math.min(lo, R.logMax[g]); hi = Math.max(hi, R.logMax[g]); }
      const setWindow = (xr, yr) => {
        const v = { rtLo: xr ? fixed(xr[0], 1) : "", rtHi: xr ? fixed(xr[1], 1) : "", mzLo: yr ? fixed(yr[0], 4) : "", mzHi: yr ? fixed(yr[1], 4) : "" };
        for (const [k, x] of Object.entries(v)) { state[k] = x; if (rangeIn[k]) rangeIn[k].value = x; }
        scroller.scrollTop = 0;
        features_refresh();
      };
      // the search settings and the filters sit above the feature list, so the detail columns start at the top
      const filters = [idPanel, h("div.toolbar", searchField, ppmField, range("m/z", "mzLo", "mzHi", ""), range("RT", "rtLo", "rtHi", "s")),
        h("div.toolbar", chip("mono", "Monoisotopic"), chip("all", "In every file"), R.h.eic ? chip("withEic", "With chromatograms") : null,
          R.m2 ? chip("withMs2", "With MS2") : null, R.id ? chip("identified", "Annotated") : null,
          a.g_hal ? chip("halogen", "Possible halogen") : null, ...STARS.map(starChip), cidBox, h("span.spacer"), count, eicBtn, dl, dlc), skipRow];
      const mini = track(P3.plot.scatter(mapEl, {
        height: 210, x: R.rtS, y: a.g_mz, v: R.logMax, vDomain: [lo, hi], r: R.G > 20000 ? 1 : R.G > 3000 ? 1.5 : 2.2,
        xLabel: "RT (s)", yLabel: "m/z", xFmt: (v, d) => fixed(v, Math.min(d, 0)), yFmt: (v, d) => fixed(v, Math.min(d, 0)),
        selected: state.sel, brushHint: "Drag to set the m/z and RT windows · double-click to clear", onBrush: setWindow,
        tip: (g) => `<b>#${a.g_id[g]} · m/z ${fixed(a.g_mz[g], 4)}</b><br>RT ${fixed(R.rtS[g], 1)} s` + (R.annotated[g] ? `<br>${esc(R.annotation(g).name)}` : ""),
        onClick: (g) => { select(g); const i = list.indexOf(g); if (i >= 0) ensureVisible(i); },
      }));

      const COLS = [["id", "ID", true], ...(R.m2 ? [["score", "Score", true], ["ident", "Annotation", true]] : []),
        ["mz", "m/z", true], ["rt", "RT (s)", true], ["width", "Width (s)", true], ["ndet", "Detected", true],
        ["nfill", "Gap-filled", true], ["max", "Max " + R.quant.toLowerCase(), true], ["cv", "CV", true], ["ion", "Ion", false],
        ...(a.g_hal ? [["hal", "Halogen", false]] : []),
        ...(R.cid ? [["cmp", "Feature group", true]] : []),
        ...(R.m2 ? [["ms2", "MS2", true]] : [])];
      const DESC = new Set(["max", "ndet", "ms2", "score", "cmp", "hal"]);
      const thead = h("thead", h("tr", COLS.map(([k, label]) => {
        const th = h("th.sortable.r", { "data-k": k }, label);
        if (k === "ion" || k === "ident" || k === "hal") th.classList.remove("r");
        if (k === "ion") th.title = "What the feature is within its feature group (the ions of one compound): M+n an isotope (peak shape confirmed), M+n? a possible isotope (mass arithmetic only), a named ion, or Δ the m/z difference to the compound's base ion";
        if (k === "cmp") th.title = "Feature group (the ions of one compound) and its size; the button lists only that group's features";
        if (k === "hal") th.title = "The run's MS1 isotope check of each monoisotopic feature in its strongest file: the best hypothesis, with ? when only possible";
        th.addEventListener("click", () => {
          if (state.sortKey === k) state.sortDir *= -1; else { state.sortKey = k; state.sortDir = DESC.has(k) ? -1 : 1; }
          features_refresh();
        });
        return th;
      })));
      const tbody = h("tbody");
      const scroller = h("div.table-wrap.vtable", { tabindex: "0" }, h("table.data", thead, tbody));
      const foot = h("div.table-foot");
      // the detail: a header over two columns, the chromatograms with the feature map below them (the map stays,
      // only its ring moves), and the annotation column
      const headBox = h("div"), chromBox = h("div"), annoBox = h("div");
      const detail = h("div.detail", h("div.detail-grid",
        h("div.detail-head", card(".elevated.head-card", null, null, headBox)),
        h("div.detail-col", card(".elevated", null, null, chromBox)),
        h("div.detail-col", card(".elevated", null, null, annoBox))));
      const mapCard = card(".elevated.map-card", null, null, h("div.section-title.first", h("h3.title-small", "Feature map"),
        h("p.body-small.muted", "Listed features in colour, the selected one ringed. Drag a box to set the m/z and RT windows; double-click to clear.")), mapEl);
      body.append(h("div.feature-split", h("div.list-col", h("div.filters", ...filters), h("div", scroller, foot), mapCard), detail));

      let list = [], rowH = 37;
      function paintHead() {
        thead.querySelectorAll("th").forEach((th) => {
          const k = th.dataset.k;
          th.querySelector("svg") && th.querySelector("svg").remove();
          if (k === state.sortKey && !state.query) th.appendChild(icon(state.sortDir > 0 ? "arrow_upward" : "arrow_downward"));
        });
      }
      function ionCell(g) {
        const x = R.ion(g), z = a.g_z[g] > 1 ? h("span.muted.body-small", ` z=${a.g_z[g]}`) : null;
        if (!x) return h("td.ion", h("span.muted", R.cid ? "–" : "M"), z);
        return h("td.ion", h("span.ion-tag." + x.cls, { title: x.title }, x.text), z);
      }
      function cmpCell(g) {
        const n = R.nMembers(g);
        if (n <= 1) return h("td.r.cmp", h("span.muted", "–"));
        const on = state.cidF === R.cid[g];
        const b = h("button.cmp-btn.state", { type: "button", "aria-pressed": String(on), title: on ? "Show all features again" : `Show only the ${n} features of feature group #${int(R.cid[g])}` },
          icon("filter_list"), `#${int(R.cid[g])} · ${n}`);
        b.addEventListener("click", (e) => { e.stopPropagation(); if (on) clearCid(); else showSiblings(g); });
        return h("td.r.cmp", b);
      }
      function rowEl(g) {
        const tr = h("tr", { "aria-selected": String(g === state.sel), "data-g": g },
          h("td.r.id-cell", R.stars(g).map((c) => h("span.star-ic.mini." + c, markIcon(c, true))), int(a.g_id[g])), ...(R.m2 ? [scoreCell(g), identCell(g)] : []),
          h("td.r", fixed(a.g_mz[g], 4)), h("td.r", fixed(R.rtS[g], 1)),
          h("td.r", fixed((a.g_rt_max[g] - a.g_rt_min[g]) * 60, 1)), h("td.r", `${a.g_ndet[g]}/${R.N}`),
          h("td.r", int(a.g_nfill[g])), h("td.r", sci(a.g_max[g])), h("td.r", isFinite(a.g_cv[g]) ? fixed(a.g_cv[g], 0) + "%" : "–"),
          ionCell(g), ...(a.g_hal ? [halCell(g)] : []), ...(R.cid ? [cmpCell(g)] : []), ...(R.m2 ? [h("td.r", R.nms2[g] ? int(R.nms2[g]) : "–")] : []));
        tr.addEventListener("click", () => select(g));
        return tr;
      }
      function halCell(g) {
        const v = R.hal(g);
        if (v === 3) return h("td.hal.yes", { title: `halogen isotope pattern (${hypName(R.halHint(g))})` }, icon("science"), hypName(R.halHint(g)));
        if (v === 2) return h("td.hal.maybe", { title: `possibly halogenated (${hypName(R.halHint(g))})` }, hypName(R.halHint(g)) + "?");
        return h("td.muted", v === 1 ? "no" : "–");
      }
      // the annotation's score; for a feature not annotated, its best hit's score, greyed
      function scoreCell(g) {
        const v = R.annScore[g];
        if (!isFinite(v)) return h("td.r.score.weak", "–");
        return h("td.r.score.num" + (R.annotated[g] ? "" : ".weak"), { title: R.annotated[g] ? "score of the annotation" : `best hit, below ${fixed(R.minScore, 2)}` }, fixed(v, 2));
      }
      function identCell(g) {
        const x = R.annotation(g);
        if (!x) return h("td.ident.muted", R.nms2[g] && R.id ? (isFinite(R.annScore[g]) ? "below threshold" : "no match") : "–");
        if (x.unknown) return h("td.ident.unknown", { title: "marked unknown by you" }, h("span.id-name", "unknown"), h("span.id-user", "you"));
        return h("td.ident", { title: `${x.name}${x.adduct ? " · " + x.adduct : ""} · score ${fixed(x.score, 3)}${x.source === "user" ? " · your choice" : ""}` },
          h("span.id-name", x.name), x.source === "user" ? h("span.id-user", "you") : null);
      }
      function paintRows() {
        const top = scroller.scrollTop, hgt = scroller.clientHeight || 560;
        const head = 45;
        const first = Math.max(0, Math.floor((top - head) / rowH) - 10);
        const last = Math.min(list.length, Math.ceil((top + hgt) / rowH) + 10);
        const pre = h("tr", h("td", { colspan: COLS.length, style: { height: first * rowH + "px", padding: 0, border: 0 } }));
        const post = h("tr", h("td", { colspan: COLS.length, style: { height: (list.length - last) * rowH + "px", padding: 0, border: 0 } }));
        const kids = [pre];
        for (let i = first; i < last; i++) kids.push(rowEl(list[i]));
        kids.push(post);
        tbody.replaceChildren(...kids);
        if (last > first) {
          const r = tbody.children[1].getBoundingClientRect().height;
          if (r && Math.abs(r - rowH) > 0.5) { rowH = r; paintRows(); }
        }
      }
      scroller.addEventListener("scroll", () => requestAnimationFrame(paintRows));
      scroller.addEventListener("keydown", (e) => {
        if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
        e.preventDefault();
        const i = list.indexOf(state.sel);
        const j = Math.max(0, Math.min(list.length - 1, (i < 0 ? -1 : i) + (e.key === "ArrowDown" ? 1 : -1)));
        if (list.length) { select(list[j]); ensureVisible(j); }
      });
      function ensureVisible(i) {
        const y = 45 + i * rowH;
        if (y < scroller.scrollTop + 45) scroller.scrollTop = y - 45;
        else if (y + rowH > scroller.scrollTop + scroller.clientHeight) scroller.scrollTop = y + rowH - scroller.clientHeight;
      }
      function features_refresh() {
        list = rows();
        count.textContent = state.cidF != null ? `${int(list.length)} features of feature group #${int(state.cidF)}` : `${int(list.length)} of ${int(R.G)} features`;
        if (state.cidF != null) {
          const x = h("button.chip.state.cmp-chip", { type: "button", "aria-pressed": "true", title: "Show all features again" }, icon("filter_list"), `Feature group #${int(state.cidF)}`, icon("close"));
          x.addEventListener("click", clearCid);
          cidBox.replaceChildren(x);
        } else cidBox.replaceChildren();
        const emph = new Uint8Array(R.G);
        for (const g of list) emph[g] = 1;
        const num = (v, d) => (v !== "" && isFinite(+v) ? +v : d);
        const boxed = [state.rtLo, state.rtHi, state.mzLo, state.mzHi].some((v) => v !== "");
        mini.update({ emph: list.length < R.G ? emph : null, selected: state.sel,
          box: boxed ? [[num(state.rtLo, -Infinity), num(state.rtHi, Infinity)], [num(state.mzLo, -Infinity), num(state.mzHi, Infinity)]] : null });
        foot.textContent = R.h.eic && R.h.eic.n < R.G ? `Chromatograms are stored for the ${int(R.h.eic.n)} most intense features (${int(R.h.eic.files.length)} of ${R.N} files each); the rest have values only.` : "";
        paintHead();
        paintRows();
      }
      let tmr = null;
      search.addEventListener("input", () => { state.query = search.value; clearTimeout(tmr); tmr = setTimeout(() => { scroller.scrollTop = 0; features_refresh(); }, 120); });
      ppmIn.addEventListener("input", () => { const v = +ppmIn.value; if (v > 0) { state.ppm = v; features_refresh(); } });

      function select(g) {
        if (g !== state.sel) { state.hit = null; state.copy = null; }
        state.sel = g;
        tbody.querySelectorAll("tr[data-g]").forEach((tr) => tr.setAttribute("aria-selected", String(+tr.dataset.g === g)));
        mini.update({ selected: g });
        renderDetail(g);
      }
      detailRefs = { head: headBox, chrom: chromBox, anno: annoBox };
      repaintRows = () => { paintRows(); features_refresh(); };
      showSiblingsOf = R.cid ? showSiblings : null;
      clearCidF = clearCid;
      features_refresh();
      if (state.sel == null && list.length) state.sel = list[0];
      if (state.sel != null) {
        select(state.sel);
        const i = list.indexOf(state.sel);
        if (i > 0) requestAnimationFrame(() => { scroller.scrollTop = Math.max(0, 45 + i * rowH - 120); paintRows(); });
      } else renderDetail(null);
    }

    // ---------------------------------------------------------------- identity search (GUI only: the server searches)
    function identityPanel() {
      if (!opts.identify) return null;
      const methods = opts.methods || {};
      const saved = (k, d) => { const v = P3.storeGet("peak3d-id-" + k); return v == null || v === "" ? d : v; };
      const last = R.id ? R.id.h : {};
      const pick = (k, sk, d) => (last[k] != null ? last[k] : saved(sk, d));
      const st = { library: pick("library", "library", ""), method: pick("method", "method", Object.keys(methods)[0] || "weighted_entropy"),
        precursor_ppm: +pick("precursor_ppm", "ppm", 10), fragment_tol: +pick("fragment_tol", "tol", 0.02), min_score: +pick("min_score", "min", 0.7) };
      const off = !R.m2;
      const libIn = h("input", { type: "text", placeholder: "Choose a .msp or .pkl file", spellcheck: "false", value: st.library, disabled: off || null });
      const browseBtn = opts.browseLibrary ? h("button.icon-btn.state", { type: "button", title: "Browse", "aria-label": "Browse for a .msp or .pkl library", disabled: off || null,
        onclick: () => opts.browseLibrary(st.library, (p) => { libIn.value = p; st.library = p; run(); }) }, icon("folder_open")) : null;
      const sel = h("select", { disabled: off || null }, Object.entries(methods).map(([k, label]) => h("option", { value: k }, label)));
      sel.value = st.method;
      const num = (key, step, min, max) => {
        const i = h("input", { type: "number", step, min, max, value: String(st[key]), disabled: off || null });
        i.addEventListener("change", () => { const v = +i.value; if (i.value !== "" && isFinite(v) && v >= min && (max == null || v <= max)) { st[key] = v; soon(); } else i.value = String(st[key]); });
        return i;
      };
      const go = h("button.btn.filled.state", { type: "button", disabled: off || null }, icon("search"), "Search");
      const status = h("p.body-small.muted.id-status");
      const summary = () => {
        if (off) return "These results hold no MS2 spectra (MS1-only files, or a run made before MS2 mapping): nothing to search.";
        if (!R.id) return st.library ? "Not searched yet with this library." : "Choose a spectral library (.msp or MassCube .pkl); the search then runs by itself whenever a setting changes.";
        const m = R.id.h;
        return `${int(R.nAnnotated)} of ${int(R.m2.h.n)} features with MS2 annotated${R.nByUser ? ` (${int(R.nByUser)} by you)` : ""} · ` +
          `${m.method_label || m.method} ≥ ${fixed(m.min_score, 2)} on the ${m.default_label || "consensus spectrum"} · ${int(m.n_hits != null ? m.n_hits : m.n)} hits above ${fixed(m.list_min != null ? m.list_min : 0.1, 1)} listed · ` +
          `${base(m.library)}${m.library_n ? ` (${int(m.library_n)} spectra)` : ""}${m.seconds != null ? ` · ${fixed(m.seconds, 1)} s` : ""}`;
      };
      status.textContent = summary();
      let busy = false, timer = null;
      const soon = () => { clearTimeout(timer); timer = setTimeout(run, 400); };
      async function run() {
        clearTimeout(timer);
        st.library = libIn.value.trim();
        st.method = sel.value;
        if (off || busy || !st.library) { status.textContent = summary(); return; }
        busy = true;
        go.disabled = true;
        status.classList.remove("err");
        status.replaceChildren(P3.spinner(14), h("span", " Searching " + base(st.library) + "…"));
        try {
          const res = await opts.identify({ library: st.library, method: st.method, precursor_ppm: st.precursor_ppm, fragment_tol: st.fragment_tol, min_score: st.min_score });
          [["library", st.library], ["method", st.method], ["ppm", st.precursor_ppm], ["tol", st.fragment_tol], ["min", st.min_score]]
            .forEach(([k, v]) => P3.storeSet("peak3d-id-" + k, String(v)));
          R.setMs2(res);
          state.hit = null;
          P3.snack(`${int(R.nAnnotated)} of ${int(R.m2.h.n)} features with MS2 annotated`);
          show("features");            // the table, chips and detail with the new hits; filters, sort and selection stay
        } catch (e) {
          status.classList.add("err");
          status.textContent = String(e.message || e);
          busy = false;
          go.disabled = false;
        }
      }
      libIn.addEventListener("change", run);
      libIn.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); run(); } });
      sel.addEventListener("change", soon);
      go.addEventListener("click", run);
      if (R.m2 && !R.id && st.library && !state.autoSearched) { state.autoSearched = true; setTimeout(run, 0); }   // automatic on open
      return card(".elevated.id-card", "Identity search",
        "Both MS2 spectra of each feature (the consensus and the highest-TIC spectrum) against a spectral library: candidates within the precursor tolerance of the feature m/z, each scored on the spectrum as measured and on the spectrum denoised with that candidate's own formula and adduct (only ions whose neutral loss is a plausible sub-formula of its precursor ion), all on cleaned spectra (precursor region removed, peaks below 1 % dropped). " +
        "Every hit above 0.1 on either version is listed with both scores; the best hit on the denoised consensus at or above the threshold becomes the feature's annotation, which you can change per feature.",
        h("div.id-form",
          field("Library (.msp, .pkl)", libIn, browseBtn ? h("span.trail", browseBtn) : null),
          field("Similarity", sel, h("span.select-arrow", icon("expand_more"))),
          field("Precursor (ppm)", num("precursor_ppm", "1", 0.1)),
          field("Fragment (Da)", num("fragment_tol", "0.005", 0.0005)),
          field("Annotate at ≥", num("min_score", "0.05", 0, 1)),
          h("div.id-go", go)),
        status);
    }

    // ---------------------------------------------------------------- feature detail
    // a header (with the user's stars), the annotation column (candidates, head-to-tail plot, structure, the user's
    // annotation, comment) and the chromatogram column, which stays as it is while the annotation changes
    const chromCharts = [], annoCharts = [], ms1Charts = [];
    const drop = (list) => { for (const c of list) { c.destroy(); const k = charts.indexOf(c); if (k >= 0) charts.splice(k, 1); } list.length = 0; };
    let headEl = null, annoEl = null, shown = null;
    function renderDetail(g) {
      drop(chromCharts);
      drop(annoCharts);
      drop(ms1Charts);
      shown = g;
      const D = detailRefs;
      if (!D) return;
      headEl = D.head;
      annoEl = D.anno;
      if (g == null) {
        headEl.replaceChildren(h("div.empty", icon("scatter_plot", "lg"), h("h2.title-medium", "No feature selected"), h("p.body-medium", "Pick a row in the table or a point on the feature map.")));
        D.chrom.replaceChildren();
        annoEl.replaceChildren();
        return;
      }
      renderHead(g);
      renderAnno(g);
      D.chrom.replaceChildren(...chromColumn(g));
    }

    function renderHead(g) {
      const a = R.a;
      const ionX = R.ion(g), nmem = R.nMembers(g), inCmp = R.cid && nmem > 1;
      const ionChip = ionX && ionX.cls !== "base" ? h("span.chip.static.ion-chip." + ionX.cls, { title: ionX.title }, ionX.text) :
        h("span.chip.static", a.g_iso[g] > 0 ? `M+${a.g_iso[g]} isotope` : "Monoisotopic");
      const baseChip = inCmp && a.g_base[g] ? h("span.chip.static.ok", { title: "the feature group's base ion: its strongest monoisotopic member" }, "base ion") : null;
      const nmChip = inCmp && isFinite(a.g_nm[g]) ? h("span.chip.static", { title: "neutral mass of the feature group's compound, from the ions a mass relation explains" }, `M ${fixed(a.g_nm[g], 4)}`) : null;
      let cmpBtn = null;
      if (inCmp && showSiblingsOf) {
        const on = state.cidF === R.cid[g];
        cmpBtn = h("button.chip.state.cmp-chip", { type: "button", "aria-pressed": String(on), title: on ? "Show all features again" :
          `List only the ${nmem} features of this feature group: adducts, in-source fragments, multimers and isotopes of one compound, grouped on peak shape and intensity correlation` },
          icon("filter_list"), `Feature group #${int(R.cid[g])} · ${nmem} features`);
        cmpBtn.addEventListener("click", () => (on ? clearCidF() : showSiblingsOf(g)));
      } else if (inCmp) cmpBtn = h("span.chip.static", `Feature group #${int(R.cid[g])} · ${nmem} features`);
      const all = a.g_ndet[g] === R.N;
      const x = R.m2 ? R.annotation(g) : null;
      const stars = h("div.stars", STARS.map((c) => {
        const on = R.stars(g).includes(c);
        const mark = `${c} ${MARK[c][1]}`;
        const b = h("button.star-btn.state" + (on ? ".on" : ""), { type: "button", "aria-pressed": String(on), "aria-label": `${c} mark`,
          title: opts.note ? `${on ? "Remove the" : "Give this feature the"} ${mark} (a mark for a rule of your own; filter by it above the list)` : `${mark}${on ? "" : " (not set)"}`,
          disabled: opts.note ? null : true }, h("span.star-ic." + c, markIcon(c, on)));
        if (opts.note) b.addEventListener("click", () => toggleStar(g, c));
        return b;
      }));
      const t0 = 60 * a.g_rt_min[g], t1 = 60 * a.g_rt_max[g];
      headEl.replaceChildren(h("div.head-row",
        h("div.head-main",
          h("div.head-line", h("span.label-medium.muted", `Feature #${a.g_id[g]}`), h("span.title-large.num", `m/z ${fixed(a.g_mz[g], 4)}`),
            h("span.body-medium.muted", `RT ${fixed(R.rtS[g], 1)} s (${fixed(t0, 1)}–${fixed(t1, 1)} s)`),
            x && !x.unknown ? h("span.body-medium.id-best", h("b", x.name), `${x.adduct ? " · " + x.adduct : ""} · ${fixed(x.score, 2)}${x.source === "user" ? (x.confirmed ? " · confirmed" : " · your choice") : ""}`) :
              x && x.unknown ? h("span.body-medium.id-best", h("i", "unknown"), " · set by you") : null),
          h("div.head-line",
            h("span.chip.static" + (all ? ".ok" : ".info"), icon(all ? "check_circle" : "info"), `Detected in ${a.g_ndet[g]} of ${R.N} files`),
            ionChip, baseChip, cmpBtn, nmChip, a.g_z[g] > 0 ? h("span.chip.static", `z = ${a.g_z[g]}`) : null,
            a.g_nfill[g] ? h("span.chip.static", `${a.g_nfill[g]} gap-filled`) : null,
            R.hal(g) === 3 ? h("span.chip.static.halogen", { title: "the run's MS1 isotope check; Show MS1 shows the pattern" }, icon("science"), `${hypName(R.halHint(g))} isotope pattern`) :
              R.hal(g) === 2 ? h("span.chip.static.info.maybe-halogen", { title: "the run's MS1 isotope check; Show MS1 shows the pattern" }, icon("info"), `${hypName(R.halHint(g))} pattern?`) : null,
            R.reviewed(g) ? h("span.chip.static.ok", icon("check"), "Reviewed") : null)),
        stars));
    }

    function renderAnno(g) {
      drop(annoCharts);
      annoEl.replaceChildren(...annoColumn(g));
    }

    // a structure drawing of the candidate (smiles-drawer, offline), or why there is none
    function structure(b) {
      const box = h("div.structure");
      const none = (why) => box.replaceChildren(h("p.body-small.muted", why),
        b && b.inchikey ? h("a.body-small", { href: `https://pubchem.ncbi.nlm.nih.gov/#query=${encodeURIComponent(b.inchikey)}`, target: "_blank", rel: "noopener" }, "Look it up on PubChem") : null);
      if (!b) { none("No candidate selected."); return box; }
      if (!b.smiles) { none("No structure (SMILES) in the library record."); return box; }
      if (!window.SmilesDrawer) { none("Structure drawing is not available here."); return box; }
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("viewBox", "0 0 260 190");
      svg.setAttribute("role", "img");
      svg.setAttribute("aria-label", `structure of ${b.name}`);
      box.append(svg);
      try {
        const d = new window.SmilesDrawer.SvgDrawer({ width: 260, height: 190, padding: 12, bondThickness: 1.1 });
        window.SmilesDrawer.parse(b.smiles, (tree) => { try { d.draw(tree, svg, "light"); } catch (e) { none("This SMILES could not be drawn."); } },
          () => none("This SMILES could not be read."));
      } catch (e) {
        none("This SMILES could not be drawn.");
      }
      return box;
    }

    // where the spectra behind the shown copy were taken, against the chromatogram: the merged ones (consensus) or the
    // highest-TIC one, inside the peak bounds or not, how far from the apex
    function ms2Timing(g, which) {
      const mk = (R.marks.get(g) || []).filter((m) => (which === "consensus" ? m.kind & 1 : m.kind & 2));
      if (!mk.length) return null;
      const a = R.a, t0 = 60 * a.g_rt_min[g], t1 = 60 * a.g_rt_max[g], apex = R.rtS[g];
      const ts = mk.map((m) => m.t).sort((p, q) => p - q);
      const inside = ts.filter((t) => t >= t0 && t <= t1).length;
      const away = ts.map((t) => Math.abs(t - apex)).sort((p, q) => p - q);
      const med = away[away.length >> 1];
      const ok = inside === ts.length;
      const text = ts.length === 1 ?
        `Taken at ${fixed(ts[0], 1)} s: ${fixed(away[0], 1)} s ${ts[0] < apex ? "before" : "after"} the apex, ${inside ? "inside" : "outside"} the peak (${fixed(t0, 1)}–${fixed(t1, 1)} s).` :
        `${ts.length} spectra taken ${fixed(ts[0], 1)}–${fixed(ts[ts.length - 1], 1)} s: ${inside === ts.length ? "all" : `${inside} of ${ts.length}`} inside the peak (${fixed(t0, 1)}–${fixed(t1, 1)} s), median ${fixed(med, 1)} s from the apex.`;
      return h("p.body-small.ms2-timing" + (ok ? ".ok" : ".off"), { title: "On the aligned RT axis, as the red marks on the chromatogram" }, icon(ok ? "check_circle" : "info"), text);
    }

    // a hit's matching and score, for the line under the head-to-tail plot
    const scoreLine = (method, b, other, otherLabel) => `${method} ${fixed(b.score, 3)}` +
      (other ? ` (${otherLabel} ${fixed(other.score, 3)})` : "") + ` · ${int(b.nmatch)} peak${b.nmatch === 1 ? "" : "s"} matched`;

    function annoColumn(g) {
      if (!R.m2) return [h("p.body-medium.muted", "These results hold no MS2 spectra: nothing to search."), commentBlock(g)];
      const kc = R.ms2cAt.get(g), ks = R.ms2At.get(g), n = R.nms2[g];
      if (ks == null) return [h("p.body-medium.muted", "No MS2 spectrum was acquired for this feature."), commentBlock(g)];
      const one = R.oneSpectrum(g);
      if (one || state.copy == null || (state.copy === "consensus" && kc == null)) state.copy = kc != null ? "consensus" : "single";
      const copy = state.copy, k = copy === "consensus" ? kc : ks, M = copy === "consensus" ? R.m2c : R.m2;
      // raw or denoised: the denoised spectrum by default, when the results have one; the search's hits follow
      const canDn = R.dn[copy] && (!R.dkey || R.hits(g, specOf(copy, true)).length > 0);   // search: per candidate
      if (state.dn == null) state.dn = true;
      const dn = canDn && state.dn, both = canDn && R.dnHits;
      const spec = specOf(copy, both && dn), other = both ? specOf(copy, !dn) : null;
      const merged = kc != null ? R.m2c.a.ms2c_merged[kc] : 1, files = kc != null ? R.m2c.a.ms2c_files[kc] : 1;
      const where = `${M.h.file[k]}, scan ${M.h.scan[k]}, ${fixed(M.a[copy === "consensus" ? "ms2c_rt" : "ms2_rt"][k], 1)} s`;
      const desc = (one ? `${n <= 1 ? "The only MS2 spectrum" : `The one usable MS2 spectrum of the ${int(n)}`}: ${where}.` :
        copy === "consensus" ? `${merged} of ${int(n)} spectra${files > 1 ? ` from ${files} files` : ""} merged (peaks in ≥ 2 spectra, weighted by precursor abundance); most abundant: ${where}.` :
          `Highest TIC of the ${int(n)}: ${where}.`);
      const dnDesc = h("span");
      // 1. which spectrum, how it agrees with the other copy and with the chromatogram
      const top = [];
      if (!one) {
        const sel = h("select", { "aria-label": "Which MS2 spectrum" },
          h("option", { value: "consensus" }, `Consensus of ${merged} spectra`), h("option", { value: "single" }, "Highest-TIC spectrum"));
        sel.value = copy;
        sel.addEventListener("change", () => { state.copy = sel.value; state.hit = null; renderAnno(g); });
        top.push(field("Spectrum", sel, h("span.select-arrow", icon("expand_more")), "230px"));
        const vs = kc != null && R.m2c.a.ms2c_vs ? R.m2c.a.ms2c_vs[kc] : NaN;
        if (isFinite(vs)) top.push(h("p.body-small.vs-note" + (vs >= 0.8 ? ".ok" : ".off"), { title: "Weighted entropy similarity of the consensus and the highest-TIC spectrum (raw), cleaned as for the search" },
          icon(vs >= 0.8 ? "check_circle" : "info"), `Consensus vs highest TIC: ${fixed(vs, 2)}`));
      }
      const when = ms2Timing(g, one ? "single" : copy);
      if (when) top.push(when);
      const out = [h("div.copy-pick", ...top), h("p.body-small.muted.ms2-desc", desc, dnDesc)];
      const hits = R.hits(g, spec);
      if (state.hit != null && !hits.includes(state.hit)) state.hit = R.partner(g, state.hit, spec);   // the same record, other copy
      if (state.hit == null || !hits.includes(state.hit)) state.hit = hits.length ? hits[0] : null;
      const lib = R.id ? R.id.a.id_lib : null;
      const otherOf = new Map(other != null && lib ? R.hits(g, other).map((j) => [lib[j], j]) : []);
      const pair = (j) => (j != null && lib && lib[j] >= 0 && otherOf.has(lib[j]) ? R.hit(otherOf.get(lib[j])) : null);
      const method = R.id ? String(R.id.h.method_label || R.id.h.method || "score").replace(/ similarity$/i, "") : "";
      // the denoised hit whose mask applies: the selected one on the denoised tab, its partner on the raw tab
      const dnHit = () => (!R.dkey || state.hit == null ? null : dn ? state.hit : (pair(state.hit) || { j: null }).j);
      const keptNow = () => (canDn ? R.kept(copy, k, dnHit()) : null);
      // what the denoising did to the spectrum shown: per candidate in a denoising search, once per spectrum otherwise
      const paintDesc = () => {
        const kept = keptNow();
        if (!canDn) { dnDesc.textContent = ""; return; }
        if (R.dkey) {
          const b = dnHit() != null ? R.hit(dnHit()) : null;
          dnDesc.textContent = !b ? "" : kept ? ` Denoised against this candidate's precursor ion ${b.dion || ""} (neutral losses that are sub-formulas of it, tier T1): kept ${int(kept[0])} of ${int(kept[1])} peaks.` :
            " This candidate's record gives no usable formula: its denoised score is the raw one.";
        } else dnDesc.textContent = kept ? ` Denoising (neutral losses from the feature m/z, ${(R.id && R.id.h.denoise) || "T0 top-5"}) kept ${int(kept[0])} of ${int(kept[1])} peaks.` : "";
      };
      const spectrum = () => {
        const b = state.hit != null ? R.hit(state.hit) : null;
        return { top: R.spectrum(copy, k, dn, state.hit), bottom: b && b.spec ? b.spec : null, bottomLabel: b ? b.name : null, prec: R.a.g_mz[g],
          score: b ? scoreLine(method, b, pair(state.hit), dn ? "raw" : "denoised") : null,
          topLabel: (copy === "consensus" && !one ? "consensus" : "measured") + (dn ? (R.dkey && b && b.dion ? `, denoised as ${b.dion}` : ", denoised") : "") };
      };
      paintDesc();
      // 2. head-to-tail with the score and the precursor; drag to zoom. Its tabs: the denoised or the raw spectrum, each
      // with the selected candidate's score on it (the zoom is kept across the switch)
      let tabs = null;
      // each tab's note: the selected candidate's score on that spectrum, or its peak count when nothing is scored
      const paintTabs = () => {
        if (!tabs) return;
        const b = state.hit != null ? R.hit(state.hit) : null, p = pair(state.hit), kept = keptNow();
        tabs.querySelectorAll(".tab").forEach((t, i) => {
          const on = i === 0;
          t.querySelector(".tab-note").textContent = b && both ? fixed((on === dn ? b : p || { score: NaN }).score, 3) :
            kept ? `${int(on ? kept[0] : kept[1])} peaks` : "";
        });
      };
      if (canDn) {
        tabs = h("div.tabs.spec-tabs", { role: "tablist", "aria-label": "MS2 spectrum shown" }, [[true, "Denoised"], [false, "Raw"]].map(([v, label]) => {
          const t = h("button.tab.state", { type: "button", role: "tab", "aria-selected": String(v === dn),
            title: v ? (R.dkey ? "Only the ions whose neutral loss is a plausible sub-formula of the selected candidate's precursor ion (specdenoise, tier T1)" :
              "Only the ions whose neutral loss from the precursor is a plausible formula (specdenoise)") : "The spectrum as measured (or merged)" },
            label, h("span.tab-note"));
          t.addEventListener("click", () => {
            if (v === dn) return;
            state.dn = v;
            state.view = chart.view;
            renderAnno(g);
          });
          return t;
        }));
        paintTabs();
        out.push(h("div.spec-head", tabs, R.dnHits || !R.id ? null : h("span.body-small.muted", "Search again to score the denoised spectra")));
      }
      const specEl = h("div");
      const chart = track(P3.plot.mirror(specEl, Object.assign({ height: 250, tol: (R.id && R.id.h.fragment_tol) || 0.02 }, spectrum())));
      if (state.view) { chart.view = state.view; state.view = null; chart.draw(); }
      annoCharts.push(chart);
      out.push(specEl);
      if (R.id) out.push(P3.plot.legend([{ color: S1, label: "measured, matched" }, { color: S2, label: "library, matched" }, { color: "--viz-neutral", label: "unmatched" },
        { kind: "dash", color: "--viz-muted", label: "precursor" }]));
      // 3. the selected candidate: structure beside its name, facts and the user's annotation
      const picEl = h("div.cmp-pic"), infoEl = h("div"), annBox = h("div"), recEl = h("div");
      const drawCand = () => {
        const b = state.hit != null ? R.hit(state.hit) : null, p = pair(state.hit);
        picEl.replaceChildren(structure(b));
        infoEl.replaceChildren(...(b ? [h("div.cmp-name", b.name),
          h("div.cmp-score.num", `${fixed(b.score, 3)}`, h("span.body-small.muted", ` ${method}${both ? (dn ? " · denoised" : " · raw") : ""} · ${int(b.nmatch)} peaks · candidate ${b.rank}`)),
          p ? h("div.body-small.muted.num", `${dn ? "Raw" : "Denoised"}: ${fixed(p.score, 3)} · ${int(p.nmatch)} peaks · candidate ${p.rank}`) : null,
          R.dkey && both ? h("div.body-small.muted", (dn ? b : p || b).dion ? `Denoised as ${(dn ? b : p || b).dion}` : "No usable formula: not denoised") : null,
          h("div.cmp-facts.body-small", [b.formula, b.adduct, isFinite(b.ppm) ? `${fixed(b.ppm, 1)} ppm` : null].filter(Boolean).join(" · "))] : []));
        recEl.replaceChildren(b ? kv([["InChIKey", b.inchikey ? h("code.mono.wrap", b.inchikey) : null], ["SMILES", b.smiles ? h("code.mono.wrap", b.smiles) : null],
          ["Library precursor", isFinite(b.prec) ? `m/z ${fixed(b.prec, 4)} (${fixed(b.ppm, 1)} ppm)` : null]]) : h("span"));
      };
      const drawAnn = () => annBox.replaceChildren(...userAnnotation(g));
      const tableParts = [];
      if (!R.id) tableParts.push(h("p.body-medium.muted", "No identity search yet: choose a spectral library in the Identity search panel."));
      else if (!hits.length) tableParts.push(banner(`No library spectrum scoring above ${fixed(R.id.h.list_min != null ? R.id.h.list_min : 0.1, 1)} (${R.id.h.method_label || R.id.h.method}) within the precursor tolerance.`));
      else {
        const x = R.annotation(g);
        // both scores of every candidate when the search scored the denoised copy: the shown copy's first (it orders the list)
        const cols = both ? ["#", "Library match", dn ? "Denoised" : "Raw", dn ? "Raw" : "Denoised", "Adduct", "Formula", "Δ ppm", "Peaks"] :
          ["#", "Library match", "Score", "Adduct", "Formula", "Δ ppm", "Peaks"];
        const right = both ? [0, 2, 3, 6, 7] : [0, 2, 5, 6];
        const tb = h("tbody", hits.map((j) => {
          const b = R.hit(j), p = both ? pair(j) : null;
          const chosen = x && !x.unknown && x.name === b.name && (x.lib == null || x.lib < 0 || x.lib === b.lib);
          const tr = h("tr" + (b.score < R.minScore ? ".below" : ""), { "aria-selected": String(j === state.hit) }, h("td.r", b.rank),
            h("td.ident", { title: b.name }, h("span.id-name", b.name), chosen ? h("span.id-user", "annotation") : null),
            h("td.r", fixed(b.score, 3)), both ? h("td.r.muted", p ? fixed(p.score, 3) : "–") : null,
            h("td", b.adduct || "–"), h("td", b.formula || "–"), h("td.r", fixed(b.ppm, 1)), h("td.r", int(b.nmatch)));
          tr.addEventListener("click", () => {
            state.hit = j;
            tb.querySelectorAll("tr").forEach((r) => r.setAttribute("aria-selected", String(r === tr)));
            chart.update(spectrum());
            paintTabs();
            paintDesc();
            drawCand();
            drawAnn();
          });
          return tr;
        }));
        const nBelow = hits.filter((j) => R.id.a.id_score[j] < R.minScore).length;
        tableParts.push(h("div.table-wrap.hits.scroll", h("table.data", h("thead", h("tr", cols
          .map((t, i) => h("th" + (right.includes(i) ? ".r" : ""), { title: both && (i === 2 || i === 3) ? `${method} on the ${t.toLowerCase()} spectrum` : null }, t)))), tb)),
          h("p.body-small.muted", { style: { margin: "6px 0 0" } }, `${int(hits.length)} listed (above ${fixed(R.id.h.list_min != null ? R.id.h.list_min : 0.1, 1)}${both ? " on the raw or the denoised spectrum" : ""}), ordered by the ${both ? (dn ? "denoised" : "raw") + " " : ""}score` +
            (nBelow ? `; ${int(nBelow)} below the annotation threshold ${fixed(R.minScore, 2)} (grey)` : "") + ". Click one to compare."));
        drawCand();
      }
      drawAnn();
      if (R.id && hits.length) out.push(h("div.compound", picEl, h("div.cmp-info", infoEl, annBox)));
      else out.push(annBox);
      // 4. all candidates; 5. the selected one's identifiers and the comment
      out.push(h("div.section-title", h("h3.title-small", `Candidates${hits.length ? ` (${int(hits.length)})` : ""}`)), ...tableParts);
      if (R.id && hits.length) out.push(recEl);
      out.push(commentBlock(g));
      return out;
    }

    // the user's annotation, apart from the candidates: the library's best consensus hit by default, confirmed or
    // replaced by the user, or "unknown"; set in the app (opts.annotate), shown read-only in a report
    function userAnnotation(g) {
      const x = R.annotation(g), e = R.notes.get(g), user = !!(e && e.annotation);
      const out = [h("div.ann-label.label-medium.muted", { title: `Default: the best candidate on the ${(R.id && R.id.h.default_label) || "consensus spectrum"} scoring ≥ ${fixed(R.minScore, 2)}. Used in the report and the export.` }, "Annotation")];
      if (!x) {
        const bs = R.bestScore(g);
        out.push(h("p.body-small.muted", { style: { margin: "2px 0 0" } }, !R.id ? "Not annotated: no identity search yet." : isFinite(bs) ?
          `Not annotated: the best candidate scores ${fixed(bs, 2)}, below ${fixed(R.minScore, 2)}.` : "Not annotated: no candidate."));
      } else if (x.unknown) {
        out.push(h("div.ann-head", h("span.chip.static.warn", icon("cancel"), "Unknown · set by you")));
      } else {
        out.push(h("div.ann-head", h("span.chip.static" + (user ? ".info" : ".ok"), icon(user ? "check" : "check_circle"),
          user ? (x.confirmed ? "Confirmed by you" : "Your choice") : "Library result · not reviewed"),
          h("span.body-small", h("b", x.name), ` · ${fixed(x.score, 3)}${x.rank > 1 ? ` · candidate ${x.rank}` : ""}`)));
      }
      if (opts.annotate && R.id) {
        const pick = state.hit != null ? R.hit(state.hit) : null;
        const same = pick && x && !x.unknown && x.name === pick.name && (x.lib == null || x.lib < 0 ? x.spectrum === pick.spectrum && x.rank === pick.rank : x.lib === pick.lib);
        const short = (t) => (t.length > 28 ? t.slice(0, 26) + "…" : t);
        const btns = [];
        if (x && x.source === "library") {
          const ok = h("button.btn.filled.state", { type: "button" }, icon("check"), "Confirm");
          ok.addEventListener("click", () => setAnnotation(g, { spectrum: SPEC_NAME[R.defSpec], rank: 1 }));
          btns.push(ok);
        }
        if (pick && !same) {
          const use = h("button.btn.tonal.state", { type: "button", title: `${pick.name} · score ${fixed(pick.score, 3)}` }, icon("check"), `Use “${short(pick.name)}”`);
          use.addEventListener("click", () => setAnnotation(g, { spectrum: pick.spectrum, rank: pick.rank }));
          btns.push(use);
        }
        const unk = h("button.btn.outlined.state", { type: "button", disabled: x && x.unknown ? true : null }, icon("cancel"), "Unknown");
        unk.addEventListener("click", () => setAnnotation(g, "unknown"));
        btns.push(unk);
        if (user) {
          const reset = h("button.btn.text.state", { type: "button" }, icon("history"), "Back to library result");
          reset.addEventListener("click", () => setAnnotation(g, null));
          btns.push(reset);
        }
        out.push(h("div.row.wrap.ann-actions", ...btns));
      } else if (!opts.annotate && user) {
        out.push(h("p.body-small.muted", "Annotations are set in the Mosa app."));
      }
      return out;
    }

    // the user's comment on the feature: saved as they type (app), shown in a report; exported with the table
    function commentBlock(g) {
      const title = h("div.section-title", h("h3.title-small", "Comment"));
      if (!opts.note) {
        const c = R.comment(g);
        return h("div", title, c ? h("p.body-medium.comment-text", c) : h("p.body-small.muted", "No comment."));
      }
      const ta = h("textarea.comment", { rows: "3", maxlength: "4000", placeholder: "Notes on this feature: saved as you type, exported with the table", "aria-label": "Comment" });
      ta.value = R.comment(g);
      const st = h("span.body-small.muted.comment-status");
      let timer = null, saved = ta.value;
      const save = async () => {
        clearTimeout(timer);
        const v = ta.value;
        if (v.trim() === saved.trim()) return;
        st.textContent = "Saving…";
        try {
          const r = await opts.note(R.a.g_id[g], { comment: v });
          saved = v;
          R.setNotes(g, r.notes);
          st.textContent = "Saved";
        } catch (e) {
          st.textContent = String(e.message || e);
        }
      };
      ta.addEventListener("input", () => { clearTimeout(timer); st.textContent = ""; timer = setTimeout(save, 900); });
      ta.addEventListener("blur", save);
      return h("div", title, ta, st);
    }

    async function setAnnotation(g, hit) {
      try {
        const r = await opts.annotate(R.a.g_id[g], hit);
        R.setNotes(g, r.notes);
        const x = R.annotation(g);
        P3.snack(hit === null ? "Back to the library result" : hit === "unknown" ? "Marked unknown" : x && x.confirmed ? `Confirmed ${x.name}` : `Annotated as ${x ? x.name : "?"}`);
        if (repaintRows) repaintRows();
        if (shown === g && headEl) { renderHead(g); renderAnno(g); }
      } catch (e) {
        P3.snack(String(e.message || e));
      }
    }

    async function toggleStar(g, c) {
      const now = R.stars(g), next = now.includes(c) ? now.filter((x) => x !== c) : now.concat(c);
      try {
        const r = await opts.note(R.a.g_id[g], { stars: next });
        R.setNotes(g, r.notes);
        if (repaintRows) repaintRows();
        if (shown === g && headEl) renderHead(g);
      } catch (e) {
        P3.snack(String(e.message || e));
      }
    }

    // how many MS1 scans the drawn peak rests on: the scans inside the peak window per shown file, and those that
    // hold the ion within the EIC tolerance. The line between two scans is interpolated, and only where both hold
    // signal, so a peak that rises within one scan is drawn as a step
    function scanNote(e) {
      const ns = e.traces.map((t) => t.ns).filter(Boolean);
      if (!ns.length) return null;
      const lo = (k) => Math.min(...ns.map((n) => n[k])), hi = (k) => Math.max(...ns.map((n) => n[k]));
      const range = (k) => lo(k) === hi(k) ? int(lo(k)) : `${int(lo(k))}–${int(hi(k))}`;
      const text = ns.length === 1
        ? `${int(ns[0].all)} MS1 scan${ns[0].all === 1 ? "" : "s"} inside the peak window, ${int(ns[0].signal)} of them holding the ion`
        : `${range("all")} MS1 scans per file inside the peak window, ${range("signal")} of them holding the ion`;
      return h("p.body-small.scan-note", h("b", text), ". Between scans the line is interpolated, and only where both neighbouring scans hold the ion; a rise within one scan is a step.");
    }

    // the chromatograms with the peak bounds, apexes and where the MS2 spectra were taken; the values per file and
    // the numbers. Built once per feature: annotation changes leave it alone
    function chromColumn(g) {
      const a = R.a, parts = [];
      const e = R.eic(g);
      const statusOf = (f) => R.status(g, f);
      const ms1 = ms1Control(g);
      if (e) {
        const chartEl = h("div");
        const t0 = 60 * a.g_rt_min[g], t1 = 60 * a.g_rt_max[g];
        const marks = R.m2 ? R.marks.get(g) || [] : [];
        const one = R.m2 ? R.oneSpectrum(g) : true;
        const tic = marks.filter((m) => m.kind & 2), merged = one ? [] : marks.filter((m) => m.kind & 1);
        const hasStatus = !!R.a.m_status;
        const colorOf = (f) => { const st = statusOf(f); return hasStatus && st != null && st < 3 ? S2 : S1; };
        const at = (x, y, t) => {            // a trace's value at time t (linear between its points)
          const n = x.length;
          if (!n) return 0;
          if (t <= x[0]) return y[0];
          if (t >= x[n - 1]) return y[n - 1];
          let lo = 0, hi = n - 1;
          while (hi - lo > 1) { const m = (lo + hi) >> 1; if (x[m] <= t) lo = m; else hi = m; }
          const w = (t - x[lo]) / (x[hi] - x[lo] || 1);
          return (1 - w) * y[lo] + w * y[hi];
        };
        // the traces: the window the results store around the peak or, once Whole run has read them (the app only),
        // every MS1 scan of the run for the same files, m/z and tolerance
        // each file at its own stored m/z window (Whole run reads the same window over the whole run)
        const whole = { on: false, q: { mz: +a.g_mz[g].toFixed(5), ppm: R.h.eic.ppm, offset: true } };
        const qOf = (t) => (a.eic_lo ? Object.assign({}, whole.q, { lo: +t.lo.toFixed(2), hi: +t.hi.toFixed(2) }) : whole.q);
        const winOf = (t) => (a.eic_lo ? ` (−${fixed(t.lo, 1)} / +${fixed(t.hi, 1)} ppm)` : "");
        const lines = () => {
          const series = e.traces.map((t) => {
            const w = whole.on ? eicPeek(R.stems[t.file], qOf(t)) : null;
            const full = !!(w && w.x);
            return { id: t.file, label: R.stems[t.file] + winOf(t), x: full ? w.x : e.x, y: full ? w.y : t.y, color: colorOf(t.file), width: 1.5, alpha: 0.9, ns: t.ns };
          });
          const dots = [];
          for (const sr of series) {          // each trace's apex inside the peak bounds
            let k = -1, bv = 0;
            for (let p = 0; p < sr.x.length; p++) if (sr.x[p] >= t0 && sr.x[p] <= t1 && sr.y[p] > bv) { bv = sr.y[p]; k = p; }
            if (k >= 0) dots.push({ x: sr.x[k], y: bv, color: sr.color });
          }
          for (const m of merged) {            // the spectra in the consensus: stars on their file's trace
            const sr = series.find((r) => r.id === m.file);
            const y = sr ? at(sr.x, sr.y, m.t) : Math.max(...series.map((r) => at(r.x, r.y, m.t)));
            dots.push({ x: m.t, y, color: "--viz-ms2", shape: "star", r: 6 });
          }
          return { series, dots };
        };
        const vlines = [{ x: t0, color: "--viz-ink-2", dash: [5, 4], width: 1.25 }, { x: t1, color: "--viz-ink-2", dash: [5, 4], width: 1.25 },
          ...tic.map((m) => ({ x: m.t, color: "--viz-ms2", dash: [], width: 2.5 }))];
        parts.push(h("div.section-title.first.with-action", h("div", h("h3.title-small", "Chromatograms"),
          h("p.body-small.muted", `${a.eic_lo ? `Each file over its own m/z span of the feature, at least ±${R.h.eic.ppm} ppm (here up to −${fixed(Math.max(...e.traces.map((t) => t.lo)), 1)} / +${fixed(Math.max(...e.traces.map((t) => t.hi)), 1)} ppm; a file without the feature: the median span)` : `±${R.h.eic.ppm} ppm around the feature`}, on the aligned RT axis${R.h.eic.files.length < R.N ? `; ${R.h.eic.files.length} of ${R.N} files (evenly spaced)` : ""}. ` +
            `Peak ${fixed(t0, 1)}–${fixed(t1, 1)} s (dashed), apex of each trace (dot)` +
            (tic.length ? (one ? "; red line: where the MS2 spectrum was taken" : `; red line: the highest-TIC MS2 spectrum; red stars: the ${int(merged.length)} spectra merged into the consensus, on their file's trace`) : "") +
            ". Hover a trace to name its file."),
          scanNote(e)), ms1.button));
        // zoom: drag, scroll, buttons; Peak returns to the stored window, Whole run reads the full chromatograms
        const win = [e.x[0], e.x[e.x.length - 1]];
        const readout = h("span.readout");
        const zb = (ic, title, f) => h("button.icon-btn.state", { type: "button", title, "aria-label": title, onclick: f }, icon(ic));
        const peakBtn = h("button.btn.text.state", { type: "button", "aria-pressed": "true", title: "Back to the window around the peak" }, "Peak");
        const wholeBtn = opts.eic ? h("button.btn.text.state", { type: "button", "aria-pressed": "false",
          title: "Read every MS1 scan of the whole run for these files (from the point-cloud cache) and show it all; zoom in for detail" }, icon("show_chart"), "Whole run") : null;
        // the button of what is shown is marked; any other range is spelled out
        const onView = (xd) => {
          const atWin = xd ? Math.abs(xd[0] - win[0]) < 1e-9 && Math.abs(xd[1] - win[1]) < 1e-9 : !whole.on;
          peakBtn.setAttribute("aria-pressed", String(atWin));
          if (wholeBtn) wholeBtn.setAttribute("aria-pressed", String(whole.on && !xd));
          readout.textContent = xd && !atWin ? `RT ${fixed(xd[0], 1)}–${fixed(xd[1], 1)} s` : "";
        };
        const wholeNote = h("div.chrom-note");
        parts.push(h("div.chrom-tools", zb("zoom_in", "Zoom in", () => c.zoomBy(0.5)), zb("zoom_out", "Zoom out", () => c.zoomBy(2)), peakBtn, wholeBtn, readout), chartEl, wholeNote);
        const c = track(P3.plot.line(chartEl, Object.assign({
          height: 250, vlines, xLabel: "Aligned RT (s)", yLabel: "Intensity", zoom: true, onView,
          xFmt: (v, d) => fixed(v, Math.min(d, 1)),
          label: (si) => c.spec.series[si].label,
          tip: (si, k) => {
            const sr = c.spec.series[si], st = statusOf(sr.id), ns = sr.ns;
            return `<b>${esc(sr.label)}</b><br>RT ${fixed(sr.x[k], 1)} s · ${sci(sr.y[k])}${st != null ? "<br>" + STATUS[st] : ""}` +
              (ns ? `<br>${int(ns.signal)} of ${int(ns.all)} MS1 scans in the peak window hold the ion` : "");
          },
        }, lines())));
        onView(null);
        peakBtn.addEventListener("click", () => (whole.on ? c.setView({ x: win.slice(), y: null }) : c.resetZoom()));
        if (wholeBtn) wholeBtn.addEventListener("click", async () => {
          if (whole.on) return c.resetZoom();
          const stems = e.traces.map((t) => R.stems[t.file]);
          const peek = (t) => eicPeek(R.stems[t.file], qOf(t));
          let done = 0;
          wholeBtn.disabled = true;
          const progress = () => wholeBtn.replaceChildren(P3.spinner(14), `Reading ${int(done)} of ${int(stems.length)}`);
          progress();
          wholeNote.replaceChildren();
          await eicLimited(e.traces, async (t) => { await eicGet(R.stems[t.file], qOf(t)); done++; if (shown === g) progress(); }, () => shown === g);
          if (shown !== g) return;
          wholeBtn.disabled = false;
          wholeBtn.replaceChildren(icon("show_chart"), "Whole run");
          const badT = e.traces.filter((t) => { const v = peek(t); return !v || v.error; });
          const bad = badT.map((t) => R.stems[t.file]);
          if (bad.length === stems.length) {
            eicRetry();
            return wholeNote.replaceChildren(banner(`Whole run not available: ${(peek(e.traces[0]) || {}).error || "no answer"}`, "error"));
          }
          whole.on = true;
          c.update(lines());
          c.resetZoom();
          wholeNote.replaceChildren(h("p.body-small.muted", `Whole run: every MS1 scan of ${int(stems.length - bad.length)} file${stems.length - bad.length > 1 ? "s" : ""}` +
            (bad.length ? ` (${int(bad.length)} kept the stored window: ${badT.map((t) => `${R.stems[t.file]}: ${(peek(t) || {}).error || "no answer"}`).join("; ")})` : "") +
            ". Drag across a peak to zoom in; Peak returns to the window around this feature."));
          if (bad.length) eicRetry();
        });
        chromCharts.push(c);
        parts.push(P3.plot.legend([
          ...(hasStatus ? [{ kind: "line", color: S1, label: "detected by the picker" }, { kind: "line", color: S2, label: "gap-filled" }] : []),
          { kind: "dash", color: "--viz-ink-2", label: "peak start / end" }, { kind: "dot", color: S1, label: "apex" },
          ...(tic.length ? (one ? [{ kind: "line", bold: true, color: "--viz-ms2", label: "MS2 spectrum" }] :
            [{ kind: "line", bold: true, color: "--viz-ms2", label: "highest-TIC MS2" }, { kind: "star", color: "--viz-ms2", label: "MS2 in the consensus" }]) : [])]));
      } else {
        parts.push(h("div.section-title.first.with-action", h("h3.title-small", "Chromatograms"), ms1.button),
          banner(R.h.eic ? "Not stored for this feature: the report keeps chromatograms for the most intense features only (peak3d report --eic-budget raises the limit)." :
            `Not available: ${R.h.eic_note || "no point-cloud cache"}. Re-run without --no-keep-cache, or build the report while the cache exists.`));
      }
      if (ms1.box) parts.push(ms1.box);
      if (R.a.m_q) {
        const vals = R.stems.map((_, f) => R.value(g, f));
        const cols = R.stems.map((_, f) => (statusOf(f) === 3 ? S1 : S2));
        const barEl = h("div");
        parts.push(h("div.section-title", h("h3.title-small", `${R.quant} in each file`)), barEl,
          P3.plot.legend([{ color: S1, label: "detected" }, { color: S2, label: "gap-filled" }]));
        chromCharts.push(track(P3.plot.bars(barEl, {
          height: 170, values: vals, colors: cols, xLabel: "File", yLabel: R.quant,
          xTick: (i) => String(i + 1), tip: (f) => `<b>${esc(R.stems[f])}</b><br>${sci(vals[f])} · ${STATUS[statusOf(f) || 0]}`,
        })));
      } else if (R.h.matrix_note) {
        parts.push(h("div.section-title", h("h3.title-small", `${R.quant} in each file`)), banner(`Per-file values are not embedded (${R.h.matrix_note}); see aligned_feature_table.tsv.`));
      }
      parts.push(h("div.section-title", h("h3.title-small", "Details")), kv([
        ["Max " + R.quant.toLowerCase(), sci(a.g_max[g])],
        ["CV across files", isFinite(a.g_cv[g]) ? fixed(a.g_cv[g], 1) + " %" : "–"],
        ["m/z spread", isFinite(a.g_ppm[g]) ? fixed(a.g_ppm[g], 2) + " ppm" : "–"],
        ["RT spread (SD)", isFinite(a.g_rtsd[g]) ? fixed(a.g_rtsd[g] * 60, 2) + " s" : "–"],
        ["Peak width", fixed((a.g_rt_max[g] - a.g_rt_min[g]) * 60, 1) + " s"],
        ["Found by", `picker in ${a.g_ndet[g]}, confirmed by gap filling in ${a.g_nconf[g]}, filled in ${a.g_nfill[g]}`],
      ]));
      return parts;
    }

    // ---------------------------------------------------------------- extracted ion chromatograms
    // on request (Extract EIC beside Download table): the whole-run chromatogram of any m/z in the files the user
    // picks, read from the point-cloud cache by the app's server (bundle.extract_eic), so a report shows no button.
    // A file keeps its colour while it stays chosen (eight categorical slots, the rest share the neutral grey).
    const EIC_COLORS = ["--viz-cat-1", "--viz-cat-2", "--viz-cat-3", "--viz-cat-4", "--viz-cat-5", "--viz-cat-6", "--viz-cat-7", "--viz-cat-8"];
    // whole-run chromatograms from the app's server (opts.eic), fetched once per file and query and shared by the
    // feature's chromatograms (Whole run) and Extract EIC; the oldest are dropped past EIC_KEEP
    const EIC_KEEP = 240;
    const eicStore = new Map();
    const eicKey = (s, q) => `${s}|${q.mz}|${q.ppm}|${q.offset}|${q.lo ?? ""}|${q.hi ?? ""}`;
    function eicGet(s, q) {
      const k = eicKey(s, q);
      let it = eicStore.get(k);
      if (!it) {
        it = { v: null };
        it.p = opts.eic(s, q.mz, q.ppm, q.offset, q.lo, q.hi).then(
          (r) => (it.v = { r, x: Float64Array.from(r.rt_s), xn: Float64Array.from(r.rt_native_s), y: Float64Array.from(r.intensity) }),
          (err) => (it.v = { error: err.message || String(err) }));
        eicStore.set(k, it);
        if (eicStore.size > EIC_KEEP) eicStore.delete(eicStore.keys().next().value);
      }
      return it.p;
    }
    const eicPeek = (s, q) => { const it = eicStore.get(eicKey(s, q)); return it ? it.v : null; };
    const eicRetry = () => { for (const [k, it] of eicStore) if (it.v && it.v.error) eicStore.delete(k); };
    // run fn over items, n at a time, while alive() holds
    async function eicLimited(items, fn, alive, n = 2) {
      const todo = items.slice();
      const next = async () => { while (todo.length && alive()) await fn(todo.shift()); };
      await Promise.all(Array.from({ length: n }, next));
    }

    function eicDialog() {
      const a = R.a, g = state.sel;
      const featMz = g != null ? +a.g_mz[g].toFixed(5) : null;
      const st = { axis: "aligned", offset: true, query: null };      // query: the m/z, ppm and offset last extracted
      const chosen = new Set(), slot = new Map();
      let chart = null, gen = 0;
      // first choice: the selected feature's three most intense files, else the first file
      if (g != null && a.m_q) {
        R.stems.map((s, f) => [s, R.value(g, f)]).filter(([, v]) => v > 0).sort((x, y) => y[1] - x[1]).slice(0, 3).forEach(([s]) => chosen.add(s));
      }
      if (!chosen.size && R.N) chosen.add(R.stems[0]);
      const take = (s) => { const used = new Set(slot.values()); for (let k = 0; k < EIC_COLORS.length; k++) if (!used.has(k)) { slot.set(s, k); return; } };
      chosen.forEach(take);
      const colorOf = (s) => (slot.has(s) ? EIC_COLORS[slot.get(s)] : "--viz-neutral");

      const dlg = h("dialog.dialog.eic-dialog", { "aria-label": "Extracted ion chromatograms" });
      const close = () => dlg.close();
      dlg.addEventListener("close", () => { gen++; if (chart) chart.destroy(); dlg.remove(); });

      // the query: m/z, tolerance, offset correction, RT axis
      const mzIn = h("input", { type: "number", step: "any", min: "0", placeholder: "e.g. 166.0863", "aria-label": "m/z",
        value: featMz != null ? String(featMz) : "" });
      // default tolerance: the selected feature's widest stored window (its own m/z span), else the run's ppm
      const ev = g != null && R.a.eic_lo ? R.eic(g) : null;
      const featPpm = ev ? Math.ceil(2 * Math.max(...ev.traces.map((t) => Math.max(t.lo, t.hi)))) / 2 : null;
      const ppmIn = h("input.narrow", { type: "number", step: "any", min: "0", "aria-label": "Tolerance (ppm)",
        value: String(featPpm || (R.h.eic && R.h.eic.ppm) || 10) });
      const offIn = h("input", { type: "checkbox", checked: true });
      offIn.addEventListener("change", () => { st.offset = offIn.checked; extract(); });
      const axisSeg = h("div.segmented", { role: "group", "aria-label": "RT axis" }, [["aligned", "Aligned RT"], ["native", "Raw RT"]].map(([k, label]) =>
        h("button.state", { type: "button", "aria-pressed": String(st.axis === k), title: k === "aligned" ? "Each file's retention times after the run's RT correction (as in the feature table)" : "Retention times as written in each mzML file",
          onclick: (e) => { st.axis = k; axisSeg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b === e.currentTarget))); redraw(); } }, label)));
      const go = h("button.btn.filled.state", { type: "button" }, icon("show_chart"), "Extract");
      go.addEventListener("click", () => extract());
      for (const i of [mzIn, ppmIn]) {
        i.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); extract(); } });
        i.addEventListener("input", () => { go.disabled = !valid(); });
      }
      const valid = () => +mzIn.value > 0 && +ppmIn.value > 0 && +ppmIn.value <= 1000 && chosen.size > 0;

      // the file list: filter, all / none of those listed, one checkbox per file
      const filterIn = h("input.eic-filter", { type: "search", placeholder: "Filter files", "aria-label": "Filter files" });
      const count = h("span.count");
      const list = h("div.eic-file-list", { role: "group", "aria-label": "Files" });
      const rows = R.stems.map((s, f) => {
        const box = h("input", { type: "checkbox", checked: chosen.has(s) });
        const sw = h("span.eic-sw");
        const v = g != null && a.m_q ? R.value(g, f) : null;
        const row = h("label.eic-file", { title: s }, box, sw, h("span.name", s), v != null ? h("span.val", { title: `${R.quant} of the selected feature` }, v > 0 ? sci(v) : "–") : null);
        box.addEventListener("change", () => setFile(s, box.checked));
        return { s, row, box, sw };
      });
      list.append(...rows.map((r) => r.row));
      const paint = () => {
        for (const r of rows) {
          r.box.checked = chosen.has(r.s);
          r.sw.style.background = chosen.has(r.s) ? `var(${colorOf(r.s)})` : "transparent";
        }
        count.textContent = `${int(chosen.size)} of ${int(R.N)} chosen`;
        go.disabled = !valid();
      };
      const shownRows = () => rows.filter((r) => r.row.style.display !== "none");
      filterIn.addEventListener("input", () => {
        const q = filterIn.value.trim().toLowerCase();
        for (const r of rows) r.row.style.display = !q || r.s.toLowerCase().includes(q) ? "" : "none";
      });
      const allBtn = h("button.btn.text.state", { type: "button", title: "Choose every file listed (the filter applies)" }, "All");
      const noneBtn = h("button.btn.text.state", { type: "button", title: "Clear the files listed (the filter applies)" }, "None");
      allBtn.addEventListener("click", () => { for (const r of shownRows()) if (!chosen.has(r.s)) { chosen.add(r.s); take(r.s); } paint(); load(); });
      noneBtn.addEventListener("click", () => { for (const r of shownRows()) { chosen.delete(r.s); slot.delete(r.s); } paint(); load(); });
      function setFile(s, on) {
        if (on) { chosen.add(s); take(s); } else { chosen.delete(s); slot.delete(s); }
        paint();
        load();
      }

      // the chart, its zoom controls, legend and status
      const chartEl = h("div");
      const empty = h("div.eic-empty", "Enter an m/z, choose files and press Extract.");
      const readout = h("span.readout");
      const zbtn = (ic, title, f) => h("button.icon-btn.state", { type: "button", title, "aria-label": title, onclick: () => chart && f() }, icon(ic));
      const zoomBar = h("div.eic-zoom", zbtn("zoom_in", "Zoom in", () => chart.zoomBy(0.5)), zbtn("zoom_out", "Zoom out", () => chart.zoomBy(2)),
        zbtn("fit_screen", "Show everything", () => chart.resetZoom()), readout);
      const legendEl = h("div");
      const status = h("div.eic-status");
      const chartCol = h("div.eic-chart-col", zoomBar, empty, chartEl, legendEl, status);
      const onView = (xd) => { readout.textContent = xd ? `RT ${fixed(xd[0], 1)}–${fixed(xd[1], 1)} s` : "Whole run"; };

      dlg.append(
        h("div.dialog-head.eic-head", h("div", h("h2.headline-small", "Extracted ion chromatograms"),
          h("p.body-small.muted", "The most intense centroid within the tolerance in every MS1 scan of each chosen file, over the whole run.")),
          h("button.icon-btn.state", { type: "button", title: "Close", "aria-label": "Close", onclick: close }, icon("close"))),
        h("div.eic-controls",
          h("div.range", { title: "The m/z to extract" }, h("span.range-label", "m/z"), mzIn),
          h("div.range", { title: "Tolerance around the m/z, in ppm" }, h("span.range-sep", "±"), ppmIn, h("span.range-unit", "ppm")),
          h("label.eic-check", { title: "Move the m/z by each file's measured m/z offset (from the alignment), as the run did; off: the m/z exactly as typed in every file" },
            offIn, "Correct each file's m/z offset"),
          axisSeg, h("span", { style: { flex: "1" } }), go),
        h("div.eic-main",
          h("div.eic-files", h("div.eic-files-head", filterIn, h("div.eic-files-tools", allBtn, noneBtn, count)), list),
          chartCol),
        h("div.dialog-actions", h("div.body-small.muted", { style: { flex: "1" } }, featMz != null ? `m/z filled in from feature #${a.g_id[g]}` : ""),
          h("button.btn.text.state", { type: "button", onclick: close }, "Close")));
      document.body.append(dlg);
      paint();
      dlg.showModal();
      if (+mzIn.value > 0) extract(); else mzIn.focus();

      // ---- data
      function extract() {
        if (!(+mzIn.value > 0)) { status.replaceChildren(banner("Enter an m/z above 0.", "error")); return; }
        if (!(+ppmIn.value > 0 && +ppmIn.value <= 1000)) { status.replaceChildren(banner("The tolerance must lie between 0 and 1000 ppm.", "error")); return; }
        st.query = { mz: +mzIn.value, ppm: +ppmIn.value, offset: st.offset };
        eicRetry();                    // Extract asks again for what failed
        load();
      }
      // fetch what the chosen files miss for the current query (two at a time), drawing as they arrive
      async function load() {
        const q = st.query;
        if (!q) return redraw();
        const my = ++gen;
        const todo = R.stems.filter((s) => chosen.has(s) && !eicPeek(s, q));
        const total = todo.length;
        let done = 0;
        const note = () => (done < total ? `Extracting… ${int(done)} of ${int(total)} file${total > 1 ? "s" : ""}` : null);
        redraw(note());
        await eicLimited(todo, async (s) => { await eicGet(s, q); done++; if (my === gen) redraw(note()); }, () => my === gen);
      }
      function redraw(progress) {
        const q = st.query;
        const got = q ? R.stems.filter((s) => chosen.has(s)).map((s) => [s, eicPeek(s, q)]).filter(([, c]) => c) : [];
        const okd = got.filter(([, c]) => !c.error), bad = got.filter(([, c]) => c.error);
        const native = st.axis === "native";
        const series = okd.map(([s, c]) => ({ id: s, label: s, x: native ? c.xn : c.x, y: c.y, color: colorOf(s), width: slot.has(s) ? 1.6 : 1.2, alpha: slot.has(s) ? 1 : 0.8, c }));
        series.sort((p, r) => (slot.has(p.id) ? 1 : 0) - (slot.has(r.id) ? 1 : 0));        // the grey ones underneath
        empty.style.display = series.length ? "none" : "";
        chartEl.style.display = series.length ? "" : "none";
        if (!series.length) {
          empty.textContent = !q ? "Enter an m/z, choose files and press Extract." : !chosen.size ? "Choose one or more files." :
            progress ? "Extracting…" : bad.length ? "Nothing to show: see the messages below." : "Nothing extracted yet.";
        }
        const sameFeature = featMz != null && q && Math.abs(q.mz - featMz) < 1e-4 && !native;
        const vlines = sameFeature ? [{ x: 60 * a.g_rt_min[g], color: "--viz-ink-2", dash: [5, 4], width: 1.25 }, { x: 60 * a.g_rt_max[g], color: "--viz-ink-2", dash: [5, 4], width: 1.25 }] : [];
        const spec = {
          series, vlines, xLabel: native ? "Raw RT (s)" : "Aligned RT (s)", yLabel: "Intensity",
          label: (si) => series[si].label,
          tip: (si, k) => {
            const sr = series[si], c = sr.c;
            return `<b>${esc(sr.label)}</b><br>RT ${fixed(sr.x[k], 1)} s · ${sci(sr.y[k])}<br>` +
              (native ? `aligned RT ${fixed(c.x[k], 1)} s` : `raw RT ${fixed(c.xn[k], 1)} s`);
          },
        };
        if (series.length && !chart) {
          const height = Math.round(Math.max(300, Math.min(640, window.innerHeight - 440)));     // the dialog's free height
          chart = P3.plot.line(chartEl, Object.assign({ height, zoom: true, onView, xFmt: (v, d) => fixed(v, Math.min(d, 1)) }, spec));
        } else if (chart) chart.update(spec);
        if (chart && series.length) onView(chart.view ? chart.xd : null);
        const keyed = series.filter((r) => slot.has(r.id)).sort((p, r) => slot.get(p.id) - slot.get(r.id));
        const grey = series.length - keyed.length;
        legendEl.replaceChildren(series.length ? P3.plot.legend([
          ...keyed.map((r) => ({ kind: "line", color: colorOf(r.id), label: r.label })),
          ...(grey ? [{ kind: "line", color: "--viz-neutral", label: `${int(grey)} more file${grey > 1 ? "s" : ""} (hover to name)` }] : []),
          ...(vlines.length ? [{ kind: "dash", color: "--viz-ink-2", label: `peak of feature #${a.g_id[g]}` }] : [])]) : "");
        const parts = [];
        if (progress) parts.push(h("p", P3.spinner(14), " ", progress));
        if (q && okd.length) {
          const offs = okd.map(([, c]) => c.r.offset_ppm).filter((o) => o[0] !== 0 || o[1] !== 0);
          const lo = Math.min(...okd.map(([, c]) => c.r.offset_ppm[0])), hi = Math.max(...okd.map(([, c]) => c.r.offset_ppm[1]));
          const sgn = (v) => (v > 0 ? "+" : "") + fixed(v, 1);
          const fromMzml = okd.filter(([, c]) => c.r.source === "mzML").length;
          parts.push(h("p", `m/z ${fixed(q.mz, 4)} ± ${fixed(q.ppm, 1)} ppm in ${int(okd.length)} file${okd.length > 1 ? "s" : ""}` +
            (q.offset ? (offs.length ? `; each file's m/z offset applied (${sgn(lo)} to ${sgn(hi)} ppm)` : "; no m/z offsets measured for these files") : "; m/z as typed in every file") +
            (fromMzml ? `; ${int(fromMzml)} read from the mzML file${fromMzml > 1 ? "s" : ""} (the point-cloud cache is gone)` : "") + "."));
        }
        for (const [s, c] of bad) parts.push(banner(`${s}: ${c.error}`, "error"));
        status.replaceChildren(...parts);
      }
    }

    // ---------------------------------------------------------------- MS1 at the apex
    // on request (the Show MS1 button beside the chromatograms): the MS1 spectrum at the feature's apex in one file, its
    // isotope pattern against the best elemental hypothesis, and the halogen test (peak3d.halogen). Needs the app's
    // server and the point-cloud cache, so a report shows no button.
    function ms1Control(g) {
      if (!opts.ms1) return { button: null, box: null };
      const box = h("div.ms1-box");
      let open = false, file = null;
      const button = h("button.btn.outlined.state.ms1-btn", { type: "button", "aria-pressed": "false",
        title: "The MS1 spectrum at this feature's apex: its isotope pattern, tested for chlorine and bromine" }, icon("bar_chart"), "Show MS1");
      let bestF = 0, bv = -1;
      for (let f = 0; f < R.N; f++) { const v = R.a.m_q ? R.value(g, f) : 0; if (v > bv) { bv = v; bestF = f; } }
      const load = async (stem) => {
        file = stem;
        drop(ms1Charts);
        box.replaceChildren(h("p.body-small.muted.ms1-wait", P3.spinner(14), h("span", " Reading the MS1 spectrum…")));
        try {
          const r = await opts.ms1(R.a.g_id[g], stem);
          if (shown !== g || !open || file !== stem) return;
          box.replaceChildren(...ms1Panel(g, r, load));
        } catch (e) {
          if (shown === g && open) box.replaceChildren(banner(e.message || String(e), "error"));
        }
      };
      button.addEventListener("click", () => {
        open = !open;
        button.setAttribute("aria-pressed", String(open));
        button.replaceChildren(icon(open ? "close" : "bar_chart"), open ? "Hide MS1" : "Show MS1");
        if (open) load(R.stems[bestF]);
        else { drop(ms1Charts); box.replaceChildren(); }
      });
      return { button, box };
    }

    function ms1Panel(g, r, reload) {
      const out = [];
      const sel = h("select", { "aria-label": "File" }, r.files.map((s) => h("option", { value: s }, s)));
      sel.value = r.file;
      sel.addEventListener("change", () => reload(sel.value));
      const t = r.test;
      const verdictChip = !t ? null : t.verdict === "halogen" ?
        h("span.chip.static.halogen", icon("science"), `Halogen pattern: ${hypName(t.best)}`) :
        t.verdict === "possible" ? h("span.chip.static.info", icon("info"), `Possibly halogenated: ${hypName(t.best)}`) :
          h("span.chip.static", "No halogen pattern");
      out.push(h("div.section-title.ms1-title", h("div", h("h3.title-small", "MS1 isotope pattern"),
        h("p.body-small.muted", r.note ? "" : `Apex scan ${int(r.apex_scan)} at ${fixed(r.rt_s, 1)} s (aligned), m/z ${fixed(r.target_mz, 4)} in this file` +
          (Math.abs(r.offset_ppm) >= 0.05 ? ` (${r.offset_ppm > 0 ? "+" : ""}${fixed(r.offset_ppm, 1)} ppm offset)` : "") +
          `; ratios summed over the top half of the peak (scans ${int(r.top_scans[0])}–${int(r.top_scans[1])}).`)),
        field("File", sel, h("span.select-arrow", icon("expand_more")), "200px")));
      if (r.note) { out.push(banner(r.note)); return out; }
      const z = r.z, j0 = r.k.indexOf(0), best = t.hypotheses[0], none = t.hypotheses.find((d) => d.name === "none") || best;
      const mono = r.mz[j0] != null ? r.mz[j0] : r.target_mz;
      const tol = 0.004 + mono * r.ppm * 1e-6;
      const specEl = h("div");
      ms1Charts.push(track(P3.plot.mirror(specEl, {
        height: 240, tol, prec: mono,
        top: { mz: r.apex_mz, inten: r.apex_inten },
        bottom: { mz: best.shift.map((d) => mono + d / z), inten: best.rel },
        topLabel: `measured · ${r.file}`, bottomLabel: `predicted · ${hypName(best.name)}, pattern model [${best.formula}]${t.ion}`,
        score: t.verdict === "halogen" ? `${hypName(t.best)} pattern` : t.verdict === "possible" ? `${hypName(t.best)}?` : "",
      })));
      out.push(specEl, P3.plot.legend([{ color: S1, label: "measured, at a predicted peak" }, { color: S2, label: "predicted pattern" },
        { color: "--viz-neutral", label: "other ions in the scan" }, { kind: "dash", color: "--viz-muted", label: "the feature (M)" }]));
      // the verdict and the numbers behind it
      const head = h("div.ms1-verdict", verdictChip,
        t.delta_chi2 != null ? h("span.body-small.muted", `Δχ² ${fixed(t.delta_chi2, 1)} for the best halogen hypothesis against the best halogen-free one (9 = call, 4 = possible)`) : null);
      out.push(head);
      if (t.verdict !== "none") {
        const line = headEl && headEl.querySelector(".head-line:last-of-type");
        if (line && !line.querySelector(".chip.halogen, .chip.maybe-halogen")) {
          line.append(t.verdict === "halogen" ? h("span.chip.static.halogen", icon("science"), `${hypName(t.best)} isotope pattern`) :
            h("span.chip.static.info.maybe-halogen", icon("info"), `${hypName(t.best)} pattern?`));
        }
      }
      const rows = [];
      for (let k = 1; k <= 4 && k < t.observed.rel.length; k++) {
        const o = t.observed.rel[k], use = t.observed.usable[k], d = t.observed.shift[k];
        rows.push(h("tr" + (use ? "" : ".off"),
          h("td", `M+${k}`), h("td.num", o == null ? "–" : fixed(o, 3) + (use ? "" : " ✕")),
          h("td.num", fixed(best.rel[k], 3)), h("td.num", fixed(none.rel[k], 3)),
          h("td.num", k >= 2 && d != null ? fixed(d, 4) : "–"), h("td.num", k >= 2 ? fixed(best.shift[k], 4) : "–"), h("td.num", k >= 2 ? fixed(none.shift[k], 4) : "–")));
      }
      out.push(h("div.table-wrap.ms1-table", h("table.data",
        h("thead", h("tr", h("th", "Peak"), h("th.num", "Observed / M"), h("th.num", `${hypName(best.name)} (best)`), h("th.num", "No halogen"),
          h("th.num", "Spacing (Da·z)"), h("th.num", "Best"), h("th.num", "No halogen"))),
        h("tbody", rows))));
      const alt = t.hypotheses.slice(0, 4).map((d) => `${hypName(d.name)} [${d.formula}]${t.ion}${Math.abs(d.mass_error_mda) > 20 ? ` (${fixed(d.mass_error_mda, 0)} mDa off)` : ""} χ² ${fixed(d.chi2, 1)}`).join(" · ");
      out.push(h("p.body-small.muted", `The compositions are pattern models, not identifications: the carbon count comes from M+1 (${int(best.n_c)} C in the best hypothesis), ` +
        `N, O and H then fill the ion's mass (${fixed(best.mass_error_mda, 1)} mDa off). ` +
        `Spacing: 37Cl and 81Br sit ${fixed(0.0097, 4)} and ${fixed(0.0088, 4)} Da below 13C2 (σ ${fixed(t.sigma_da, 4)} Da at ${r.ppm} ppm). Ranking: ${alt}.`));
      for (const f of t.flags) out.push(h("p.body-small.ms2-timing.off", icon("info"), f));
      return out;
    }

    // ---------------------------------------------------------------- alignment
    function alignment() {
      const a = R.a, res = R.h.residuals;
      const medoid = R.files.find((f) => f.role === "medoid");
      const flagged = R.files.filter((f) => f.flag && !String(f.flag).startsWith("ok") && f.role !== "medoid");
      if (R.N < 2 || !a.drift_off || a.drift_x.length === 0) {
        const by = R.kernel !== "peak3d" ? `Retention times were aligned by ${R.kernelLabel}; its corrections are not exported, so there is nothing to show here.` : null;
        body.append(h("div", { style: { marginTop: "16px" } }, banner(R.N < 2 ? "A single file: nothing to align." : by || "RT correction was disabled for this run (--no-rt-correction).")));
        return;
      }
      body.append(h("div.tiles",
        tile("Reference (medoid) file", medoid ? medoid.stem : "–", "other files are warped onto its RT axis"),
        tile("Anchor residual MAD", res ? `${fixed(res.mad_after, 2)} s` : "–", res ? `${fixed(res.mad_before, 2)} s before correction, ${int(res.n_after)} anchors` : null),
        tile("Files flagged", int(flagged.length), flagged.length ? "see the table below" : "all warps passed the checks")));
      const driftEl = h("div"), resEl = h("div");
      body.append(h("div.grid.cols-2",
        card(".elevated", "RT correction per file", "Shift applied to each file to reach the consensus axis. Light = first file, dark = last; hover a curve to name it.", driftEl,
          P3.plot.legend([{ kind: "ramp", from: "--viz-ramp-from", to: "--viz-ramp-to", left: "first file", right: "last file" }])),
        card(".elevated", "Anchor residuals", "Distance of each anchor from the consensus RT, before and after correction.", resEl,
          res ? P3.plot.legend([{ kind: "line", color: S1, label: `before (MAD ${fixed(res.mad_before, 2)} s)` }, { kind: "line", color: S2, label: `after (MAD ${fixed(res.mad_after, 2)} s)` }]) : null)));
      const P = { from: P3.css("--viz-ramp-from"), to: P3.css("--viz-ramp-to") };
      const series = R.stems.map((s, f) => {
        const o0 = a.drift_off[f], o1 = a.drift_off[f + 1];
        return { id: f, label: s, x: Float32Array.from(a.drift_x.subarray(o0, o1), (v) => v * 60), y: a.drift_y.subarray(o0, o1), width: R.files[f] && R.files[f].role === "medoid" ? 2.5 : 1.5, color: null, t: R.N > 1 ? f / (R.N - 1) : 0 };
      });
      const recolor = () => { const from = P3.css("--viz-ramp-from"), to = P3.css("--viz-ramp-to"); series.forEach((r) => (r.color = P3.plot.mix(from, to, r.t))); };
      recolor();
      const dc = track(P3.plot.line(driftEl, {
        height: 300, series, yZero: false, xLabel: "Native RT (s)", yLabel: "Shift (s)", highlight: medoid ? R.stems.indexOf(medoid.stem) : null,
        xFmt: (v, d) => fixed(v, Math.min(d, 1)), yFmt: (v, d) => fixed(v, Math.min(d, 1)),
        label: (si) => (R.files[si] && R.files[si].role === "medoid" ? "medoid: " : "") + series[si].label,
        tip: (si, k) => `<b>${esc(series[si].label)}</b><br>RT ${fixed(series[si].x[k], 1)} s · shift ${fixed(series[si].y[k], 2)} s`,
      }));
      const off = P3.onTheme(() => { recolor(); dc.draw(); });
      charts.push({ destroy: off });
      if (res) {
        const nb = res.before.length, w = 2 * res.lim / nb;
        const x = Float32Array.from({ length: nb }, (_, i) => -res.lim + (i + 0.5) * w);
        track(P3.plot.line(resEl, {
          height: 300, step: true, xLabel: "Residual to consensus RT (s)", yLabel: "Anchors",
          xFmt: (v, d) => fixed(v, Math.min(d, 1)),
          series: [{ id: "b", label: "before", x, y: Float32Array.from(res.before), color: S1, fill: true },
            { id: "a", label: "after", x, y: Float32Array.from(res.after), color: S2, fill: true }],
          tip: (si, k) => `<b>${si ? "after" : "before"}</b><br>${fixed(x[k] - w / 2, 2)} to ${fixed(x[k] + w / 2, 2)} s: ${int((si ? res.after : res.before)[k])} anchors`,
        }));
      } else resEl.append(banner("No anchors were matched."));

      const tb = h("tbody", R.files.map((f) => h("tr",
        h("td", f.stem),
        h("td.r", f.n_matched != null ? `${int(f.n_inlier)} / ${int(f.n_matched)}` : "–"),
        h("td.r", fixed(f.mad_before_s, 2)), h("td.r", fixed(f.mad_after_s, 2)), h("td.r", fixed(f.max_abs_shift_s, 1)),
        h("td.r", fixed(f.mz_offset_ppm, 2)),
        h("td", f.flag ? h("span.chip.static" + (String(f.flag).startsWith("ok") || f.role === "medoid" ? ".ok" : ".warn"), String(f.flag)) : "–"))));
      body.append(h("h3.title-medium.section-title", "Files"), h("div.table-wrap", h("table.data.static",
        h("thead", h("tr", ["File", "Anchors (inlier / matched)", "MAD before (s)", "MAD after (s)", "Max shift (s)", "m/z offset (ppm)", "Check"].map((t, i) => h("th" + (i && i < 6 ? ".r" : ""), t)))), tb)));
    }

    // ---------------------------------------------------------------- run details
    function details() {
      const p = R.params, o = R.opts, st = p.stage_times_s || {};
      const files = p.input_files || [];
      body.append(h("div.grid.cols-2", { style: { marginTop: "16px" } },
        card(".elevated", "Run", null, kv([
          ["Kernel", `${R.kernelLabel} ${R.kernelVersion}`], ["Picker variant", p.variant], ["Mosa", R.kernel !== "peak3d" ? p.peak3d_version : null],
          ["Command", p.command ? h("code.mono", p.command) : null],
          ["Files", int(R.N)], ["Alignment", p.alignment], ["Reference file", p.medoid],
          ["m/z tolerance", p.ppm_tol != null ? fixed(p.ppm_tol, 2) + " ppm" : null],
          ["RT tolerance", p.rt_tol_min != null ? fixed(p.rt_tol_min * 60, 2) + " s" : null],
          ["Software", ["numpy " + (p.numpy || "?"), "scipy " + (p.scipy || "?"), "numba " + (p.numba || "?"), "pandas " + (p.pandas || "?")].join(" · ")],
          ["Report built", R.h.created],
        ])),
        card(".elevated", "Options", null, kv(Object.entries(o).filter(([k]) => k !== "cmd").map(([k, v]) => [k.replace(/_/g, " "), v == null ? "–" : Array.isArray(v) ? v.join(", ") || "–" : String(v)]))),
        card(".elevated", "Time per stage", null, kv(STAGE_KEYS.filter(([k]) => st[k] != null).map(([k, label]) => [label.replace("{k}", R.kernelLabel), duration(st[k])]))),
        card(".elevated", "Notes", null, (p.notes && p.notes.length) ? h("ul.body-medium", { style: { margin: 0, paddingLeft: "20px" } }, p.notes.map((n) => h("li", n))) : h("p.body-medium.muted", "None."))));
      body.append(h("h3.title-medium.section-title", "Input files"), h("div.table-wrap", h("table.data.static",
        h("thead", h("tr", ["#", "File", "Features", "Scans", "Centroids", "Read (s)", "Pick (s)"].map((t, i) => h("th" + (i !== 1 ? ".r" : ""), t)))),
        h("tbody", R.files.map((f, i) => h("tr", h("td.r", i + 1), h("td", { title: files[i] || "" }, f.stem), h("td.r", int(f.n_features)), h("td.r", int(f.n_scans)),
          h("td.r", compact(f.n_centroids)), h("td.r", fixed(f.parse_s, 1)), h("td.r", fixed(f.pick_s, 1))))))));
      body.append(h("p.footer-note", "Values shown here are rounded for display; the exact numbers are in aligned_feature_table.tsv and the other tables in the output folder."));
    }

    show(opts.tab || "overview");
    return { results: R, show, destroy: clearCharts };
  }

  P3.viewer = { mount, Results, bundleTable, candidatesJson };
})(window.P3 = window.P3 || {});
