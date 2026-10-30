// Results viewer: Overview, Features (table + chromatograms), Alignment and Run details for one peak3d
// output. Mounted by report.html (standalone) and by the GUI. Reads the decoded bundle only.
(function (P3) {
  "use strict";
  const { h, icon, fixed, int, compact, sci, duration } = P3;
  const S1 = "--viz-series-1", S2 = "--viz-series-2";
  const STATUS = ["not found", "gap-filled", "gap-filled (peak confirmed)", "detected"];
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
      this.kernel = header.kernel || "peak3d";
      this.kernelLabel = header.kernel_label || this.kernel;
      this.kernelVersion = header.kernel_version || this.params.peak3d_version || "";
      this.setMs2({ header, a });
    }
    // both MS2 copies of each group (the consensus of all its spectra, m2c; the highest-TIC spectrum, m2), where their
    // spectra were taken (marks), the hits of the last identity search on both copies (every hit above 0.1) and the
    // user's annotations, all by table row; a new search hands in a bundle with only these parts
    setMs2({ header, a }) {
      this.m2 = header.ms2 ? { h: header.ms2, a } : null;
      this.m2c = header.ms2 && header.ms2c ? { h: header.ms2c, a } : null;
      this.id = header.ms2 && header.ids ? { h: header.ids, a } : null;
      this.ms2At = new Map();
      this.ms2cAt = new Map();
      this.idsAt = [new Map(), new Map()];      // [consensus, single]: row -> hit indices, best first
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
      this.defSpec = this.m2c ? 0 : 1;                    // the copy whose best hit is the default annotation
      this.minScore = this.id && this.id.h.min_score != null ? this.id.h.min_score : 0.7;
      this.setAnnotations(header.ann || {});
    }
    // the user's choices ({row: entry}) and what follows for every row: annotated or not, the score shown
    setAnnotations(ann) {
      this.ann = new Map(Object.entries(ann || {}).map(([r, v]) => [+r, v]));
      this.annotated = new Uint8Array(this.G);
      this.annScore = new Float32Array(this.G).fill(NaN);
      this.nAnnotated = 0;
      this.nByUser = 0;
      if (!this.m2) return;
      for (let g = 0; g < this.G; g++) {
        if (!this.nms2[g] && !this.ann.has(g)) continue;
        const x = this.annotation(g);
        if (x && !x.unknown) { this.annotated[g] = 1; this.annScore[g] = x.score; this.nAnnotated++; }
        else this.annScore[g] = this.bestScore(g);
        if (x && x.source === "user") this.nByUser++;
      }
    }
    hits(g, spec) { return (this.id && this.idsAt[spec].get(g)) || []; }
    bestScore(g) { const l = this.hits(g, this.defSpec); return l.length ? this.id.a.id_score[l[0]] : NaN; }
    // the annotation in force: the user's choice, else the best hit on the default copy at or above min_score
    annotation(g) {
      const u = this.ann.get(g);
      if (u) return u.status === "unknown" ? { source: "user", unknown: true } : { source: "user", name: u.name, adduct: u.precursor_type,
        formula: u.formula, inchikey: u.inchikey, smiles: u.smiles, score: u.score, spectrum: u.spectrum, rank: u.rank };
      const l = this.hits(g, this.defSpec);
      if (!l.length) return null;
      const b = this.hit(l[0]);
      return b.score >= this.minScore ? Object.assign({ source: "library" }, b) : null;
    }
    // one copy's spectrum: which = "consensus" (k into m2c) or "single" (k into m2)
    spectrum(which, k) {
      const c = which === "consensus", a = (c ? this.m2c : this.m2).a, p = c ? "ms2c_" : "ms2_";
      const o0 = a[p + "off"][k], o1 = a[p + "off"][k + 1];
      return { mz: a[p + "mz"].subarray(o0, o1), inten: a[p + "int"].subarray(o0, o1) };
    }
    hit(j) {
      const a = this.id.a, h = this.id.h;
      const spec = h.spectra ? { mz: a.id_mz.subarray(a.id_off[j], a.id_off[j + 1]), inten: a.id_int.subarray(a.id_off[j], a.id_off[j + 1]) } : null;
      return { j, rank: a.id_rank[j], score: a.id_score[j], nmatch: a.id_nmatch[j], ppm: a.id_ppm[j], prec: a.id_prec[j],
        spectrum: a.id_spec && !a.id_spec[j] ? "consensus" : "single",
        name: h.name[j], adduct: h.precursor_type[j], formula: h.formula[j], inchikey: h.inchikey[j], smiles: h.smiles[j], spec };
    }
    value(g, f) {
      const q = this.a.m_q[g * this.N + f];
      if (!q) return 0;
      const m = this.h.matrix;
      return Math.pow(10, m.log10_lo + (q - 1) / 65534 * (m.log10_hi - m.log10_lo));
    }
    status(g, f) { return this.a.m_status ? this.a.m_status[g * this.N + f] : null; }
    eic(g) {
      const k = this.eicAt.get(g);
      if (k == null) return null;
      const e = this.h.eic, P = e.points, F = e.files.length, a = this.a;
      const x = new Float32Array(P);
      for (let j = 0; j < P; j++) x[j] = 60 * (a.eic_x0[k] + j * a.eic_dx[k]);
      const traces = e.files.map((f, j) => {
        const y = new Float32Array(P), off = (k * F + j) * P, sc = a.eic_scale[k] / 65535;
        for (let p = 0; p < P; p++) y[p] = a.eic_q[off + p] * sc;
        return { file: f, y };
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
  // times in seconds, plus the MS2 count and best library hit when the results have them
  const TABLE_COLS = ["group_id", "mz", "rt_s", "rt_min_s", "rt_max_s", "n_detected", "n_confirmed", "n_filled", "mz_ppm_spread",
    "rt_sd_s", "iso_offset", "charge"];
  const MS2_COLS = ["n_ms2", "n_consensus", "annotation", "annotation_source", "annotation_score", "annotation_adduct",
    "annotation_formula", "annotation_inchikey", "best_hit", "best_score"];
  function bundleTable(R, rows) {
    const a = R.a, num = (v, d) => (isFinite(v) ? v.toFixed(d) : ""), clean = (t) => String(t || "").replace(/[\t\r\n]/g, " ");
    const parts = [TABLE_COLS.concat(R.m2 ? MS2_COLS : [], a.m_q ? R.stems : []).join("\t") + "\n"];
    let buf = [];
    for (const g of rows) {
      const row = [a.g_id[g], a.g_mz[g].toFixed(6), num(60 * a.g_rt[g], 2), num(60 * a.g_rt_min[g], 2), num(60 * a.g_rt_max[g], 2),
        a.g_ndet[g], a.g_nconf[g], a.g_nfill[g], num(a.g_ppm[g], 3), num(60 * a.g_rtsd[g], 3), a.g_iso[g], a.g_z[g]];
      if (R.m2) {
        const x = R.annotation(g), kc = R.ms2cAt.get(g), l = R.hits(g, R.defSpec), b = l.length ? R.hit(l[0]) : null;
        const named = x && !x.unknown;
        row.push(R.nms2[g], kc != null ? R.m2c.a.ms2c_merged[kc] : 0, x ? (x.unknown ? "unknown" : clean(x.name)) : "", x ? x.source : "",
          named ? num(x.score, 4) : "", named ? clean(x.adduct) : "", named ? clean(x.formula) : "", named ? clean(x.inchikey) : "",
          b ? clean(b.name) : "", b ? b.score.toFixed(4) : "");
      }
      if (a.m_q) for (let f = 0; f < R.N; f++) { const v = R.value(g, f); row.push(v ? String(+v.toPrecision(5)) : "0"); }
      buf.push(row.join("\t"));
      if (buf.length >= 2000) { parts.push(buf.join("\n") + "\n"); buf = []; }
    }
    if (buf.length) parts.push(buf.join("\n") + "\n");
    return new Blob(parts, { type: "text/tab-separated-values" });
  }

  // ---------------------------------------------------------------- viewer
  function mount(root, bundle, opts) {
    opts = opts || {};
    const R = new Results(bundle);
    const charts = [];
    const state = { tab: "overview", sel: null, sortKey: "max", sortDir: -1, query: "", mono: false, all: false, withEic: false, ppm: 10,
      mzLo: "", mzHi: "", rtLo: "", rtHi: "", withMs2: false, identified: false, hit: null, autoSearched: false, copy: null };
    let detailEl = null, repaintRows = null;      // the Features page's detail panel and table, while it is shown
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
      detailEl = null;
      repaintRows = null;
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
      body.append(h("div.tiles",
        tile("Files", int(R.N), R.params.input_files ? "centroided MS1 mzML" : null),
        tile("Aligned features", int(R.G), st.n_groups_all ? `${int(st.n_groups_all)} groups before the presence rule` : null),
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
      for (let g = 0; g < R.G; g++) {
        if (a.g_mz[g] < mzLo || a.g_mz[g] > mzHi || R.rtS[g] < rtLo || R.rtS[g] > rtHi) continue;
        if (state.mono && a.g_iso[g] > 0) continue;
        if (state.all && a.g_ndet[g] < R.N) continue;
        if (state.withEic && !R.eicAt.has(g)) continue;
        if (state.withMs2 && !R.nms2[g]) continue;
        if (state.identified && !R.annotated[g]) continue;
        if (idq != null && a.g_id[g] !== idq) continue;
        if (mzq != null && Math.abs(a.g_mz[g] - mzq) / mzq * 1e6 > state.ppm) continue;
        out.push(g);
      }
      const key = {
        id: a.g_id, mz: a.g_mz, rt: a.g_rt, ndet: a.g_ndet, nfill: a.g_nfill, max: a.g_max, cv: a.g_cv, iso: a.g_iso,
        ms2: R.nms2, score: R.annScore,
      }[state.sortKey];
      if (mzq != null) out.sort((p, r) => Math.abs(a.g_mz[p] - mzq) - Math.abs(a.g_mz[r] - mzq));
      else if (state.sortKey === "ident") {         // annotated first, by name; then the rest by best score
        const name = (g) => (R.annotated[g] ? String(R.annotation(g).name).toLowerCase() : null);
        out.sort((p, r) => { const x = name(p), y = name(r); return x == null ? (y == null ? (R.annScore[r] || 0) - (R.annScore[p] || 0) : 1) : y == null ? -1 : state.sortDir * x.localeCompare(y); });
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
      const dl = h("button.btn.outlined.state", { type: "button", title: opts.exportTable ?
        "The features listed here (search, filters and sort applied), with the values from aligned_feature_table.tsv; retention times in seconds" :
        "The features listed here (search, filters and sort applied); retention times in seconds. Values are rounded to 0.04 %; the exact table is aligned_feature_table.tsv in the output folder (minutes)." },
        icon("download"), "Download table");
      dl.addEventListener("click", async () => {
        const name = `${R.h.title}_features${list.length < R.G ? "_" + list.length : ""}.tsv`;
        dl.disabled = true;
        try {
          const blob = opts.exportTable ? await opts.exportTable(list.map((g) => R.a.g_id[g])) : bundleTable(R, list);
          P3.saveBlob(blob, name);
          P3.snack(`Downloaded ${int(list.length)} features as ${name}`);
        } catch (e) {
          P3.snack(String(e.message || e));
        } finally {
          dl.disabled = false;
        }
      });
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
      if (idPanel) body.append(idPanel);
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
      body.append(h("div.filter-row", mapEl, h("div.filter-tools",
        h("div.toolbar", searchField, ppmField, range("m/z", "mzLo", "mzHi", ""), range("RT", "rtLo", "rtHi", "s")),
        h("div.toolbar", chip("mono", "Monoisotopic"), chip("all", "In every file"), R.h.eic ? chip("withEic", "With chromatograms") : null,
          R.m2 ? chip("withMs2", "With MS2") : null, R.id ? chip("identified", "Annotated") : null, h("span.spacer"), count, dl))));
      const mini = track(P3.plot.scatter(mapEl, {
        height: 200, x: R.rtS, y: a.g_mz, v: R.logMax, vDomain: [lo, hi], r: R.G > 20000 ? 1 : R.G > 3000 ? 1.5 : 2.2,
        xLabel: "RT (s)", yLabel: "m/z", xFmt: (v, d) => fixed(v, Math.min(d, 0)), yFmt: (v, d) => fixed(v, Math.min(d, 0)),
        selected: state.sel, brushHint: "Drag to set the m/z and RT windows · double-click to clear", onBrush: setWindow,
        tip: (g) => `<b>#${a.g_id[g]} · m/z ${fixed(a.g_mz[g], 4)}</b><br>RT ${fixed(R.rtS[g], 1)} s` + (R.annotated[g] ? `<br>${esc(R.annotation(g).name)}` : ""),
        onClick: (g) => { select(g); const i = list.indexOf(g); if (i >= 0) ensureVisible(i); },
      }));

      const COLS = [["id", "ID", true], ...(R.m2 ? [["score", "Score", true], ["ident", "Annotation", true]] : []),
        ["mz", "m/z", true], ["rt", "RT (s)", true], ["width", "Width (s)", true], ["ndet", "Detected", true],
        ["nfill", "Gap-filled", true], ["max", "Max " + R.quant.toLowerCase(), true], ["cv", "CV", true], ["iso", "Isotope", false],
        ...(R.m2 ? [["ms2", "MS2", true]] : [])];
      const DESC = new Set(["max", "ndet", "ms2", "score"]);
      const thead = h("thead", h("tr", COLS.map(([k, label]) => {
        const th = h("th.sortable.r", { "data-k": k }, label);
        if (k === "iso" || k === "ident") th.classList.remove("r");
        th.addEventListener("click", () => {
          if (state.sortKey === k) state.sortDir *= -1; else { state.sortKey = k; state.sortDir = DESC.has(k) ? -1 : 1; }
          features_refresh();
        });
        return th;
      })));
      const tbody = h("tbody");
      const scroller = h("div.table-wrap.vtable", { tabindex: "0" }, h("table.data", thead, tbody));
      const foot = h("div.table-foot");
      const detail = h("div.detail");
      body.append(h("div.feature-split", h("div", scroller, foot), detail));

      let list = [], rowH = 37;
      function paintHead() {
        thead.querySelectorAll("th").forEach((th) => {
          const k = th.dataset.k;
          th.querySelector("svg") && th.querySelector("svg").remove();
          if (k === state.sortKey && !state.query) th.appendChild(icon(state.sortDir > 0 ? "arrow_upward" : "arrow_downward"));
        });
      }
      function rowEl(g) {
        const iso = a.g_iso[g] > 0 ? `M+${a.g_iso[g]}` : "M";
        const z = a.g_z[g] > 1 ? ` · z=${a.g_z[g]}` : "";
        const tr = h("tr", { "aria-selected": String(g === state.sel), "data-g": g },
          h("td.r", int(a.g_id[g])), ...(R.m2 ? [scoreCell(g), identCell(g)] : []),
          h("td.r", fixed(a.g_mz[g], 4)), h("td.r", fixed(R.rtS[g], 1)),
          h("td.r", fixed((a.g_rt_max[g] - a.g_rt_min[g]) * 60, 1)), h("td.r", `${a.g_ndet[g]}/${R.N}`),
          h("td.r", int(a.g_nfill[g])), h("td.r", sci(a.g_max[g])), h("td.r", isFinite(a.g_cv[g]) ? fixed(a.g_cv[g], 0) + "%" : "–"),
          h("td", iso + z), ...(R.m2 ? [h("td.r", R.nms2[g] ? int(R.nms2[g]) : "–")] : []));
        tr.addEventListener("click", () => select(g));
        return tr;
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
        count.textContent = `${int(list.length)} of ${int(R.G)} features`;
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
        renderDetail(detail, g);
      }
      detailEl = detail;
      repaintRows = () => { paintRows(); features_refresh(); };
      features_refresh();
      if (state.sel == null && list.length) state.sel = list[0];
      if (state.sel != null) {
        select(state.sel);
        const i = list.indexOf(state.sel);
        if (i > 0) requestAnimationFrame(() => { scroller.scrollTop = Math.max(0, 45 + i * rowH - 120); paintRows(); });
      } else renderDetail(detail, null);
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
          `${m.method_label || m.method} ≥ ${fixed(m.min_score, 2)} on the consensus spectrum · ${int(m.n_hits != null ? m.n_hits : m.n)} hits above ${fixed(m.list_min != null ? m.list_min : 0.1, 1)} listed · ` +
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
        "Both MS2 spectra of each feature (the consensus and the highest-TIC spectrum) against a spectral library: candidates within the precursor tolerance of the feature m/z, scored on cleaned spectra (precursor region removed, peaks below 1 % dropped). " +
        "Every hit above 0.1 is listed; the best consensus hit at or above the threshold becomes the feature's annotation, which you can change per feature.",
        h("div.id-form",
          field("Library (.msp, .pkl)", libIn, browseBtn ? h("span.trail", browseBtn) : null),
          field("Similarity", sel, h("span.select-arrow", icon("expand_more"))),
          field("Precursor (ppm)", num("precursor_ppm", "1", 0.1)),
          field("Fragment (Da)", num("fragment_tol", "0.005", 0.0005)),
          field("Annotate at ≥", num("min_score", "0.05", 0, 1)),
          h("div.id-go", go)),
        status);
    }

    let detailCharts = [];
    function renderDetail(el, g) {
      detailCharts.forEach((c) => { c.destroy(); const k = charts.indexOf(c); if (k >= 0) charts.splice(k, 1); });
      detailCharts = [];
      if (g == null) {
        el.replaceChildren(card(".elevated", null, null, h("div.empty", icon("scatter_plot", "lg"), h("h2.title-medium", "No feature selected"), h("p.body-medium", "Pick a row in the table or a point on the feature map."))));
        return;
      }
      const a = R.a;
      const iso = a.g_iso[g] > 0 ? `M+${a.g_iso[g]} isotope` : "Monoisotopic";
      const all = a.g_ndet[g] === R.N;
      const head = h("div",
        h("div.label-medium.muted", `Feature #${a.g_id[g]}`),
        h("h2.headline-small.num", `m/z ${fixed(a.g_mz[g], 4)}`),
        h("p.body-medium.muted", `RT ${fixed(R.rtS[g], 1)} s · window ${fixed(60 * a.g_rt_min[g], 1)}–${fixed(60 * a.g_rt_max[g], 1)} s`),
        R.annotated[g] ? (() => { const x = R.annotation(g); return h("p.body-medium.id-best", h("b", x.name), `${x.adduct ? " · " + x.adduct : ""} · score ${fixed(x.score, 2)}${x.source === "user" ? " · your annotation" : ""}`); })() : null,
        h("div.row.wrap", { style: { marginTop: "12px" } },
          h("span.chip.static" + (all ? ".ok" : ".info"), icon(all ? "check_circle" : "info"), `Detected in ${a.g_ndet[g]} of ${R.N} files`),
          h("span.chip.static", iso), a.g_z[g] > 0 ? h("span.chip.static", `z = ${a.g_z[g]}`) : null,
          a.g_nfill[g] ? h("span.chip.static", `${a.g_nfill[g]} gap-filled`) : null));
      const parts = [head];
      const ann = R.m2 ? annotationSection(g) : null;
      if (ann) parts.push(ann.el);

      const e = R.eic(g);
      const statusOf = (f) => R.status(g, f);
      if (e) {
        const chartEl = h("div");
        const t0 = 60 * a.g_rt_min[g], t1 = 60 * a.g_rt_max[g];
        const marks = R.marks.get(g) || [];
        const nIn = marks.filter((m) => m.kind & 1).length;
        parts.push(h("div.section-title", h("h3.title-small", "Chromatograms"),
          h("p.body-small.muted", `±${R.h.eic.ppm} ppm around the feature, on the aligned RT axis${R.h.eic.files.length < R.N ? `; ${R.h.eic.files.length} of ${R.N} files (evenly spaced)` : ""}. ` +
            `Peak ${fixed(t0, 1)}–${fixed(t1, 1)} s (dashed), apex of each trace (dot)` +
            (marks.length ? `; orange: where the MS2 spectra were taken (${int(nIn)} in the consensus, bold: the highest-TIC spectrum)` : "") + ". Hover a trace to name its file.")), chartEl);
        const hasStatus = !!R.a.m_status;
        const series = e.traces.map((t) => {
          const st = statusOf(t.file);
          return { id: t.file, label: R.stems[t.file], x: e.x, y: t.y, color: hasStatus && st != null && st < 3 ? S2 : S1, width: 1.5, alpha: 0.9 };
        });
        const dots = [];
        for (const sr of series) {          // each trace's apex inside the peak bounds
          let k = -1, bv = 0;
          for (let p = 0; p < sr.x.length; p++) if (sr.x[p] >= t0 && sr.x[p] <= t1 && sr.y[p] > bv) { bv = sr.y[p]; k = p; }
          if (k >= 0) dots.push({ x: sr.x[k], y: bv, color: sr.color });
        }
        const vlines = [{ x: t0, color: "--viz-ink-2", dash: [5, 4], width: 1.25 }, { x: t1, color: "--viz-ink-2", dash: [5, 4], width: 1.25 }];
        for (const m of marks) if (!(m.kind & 2)) vlines.push({ x: m.t, color: "--viz-ms2", dash: [3, 3], width: 1.25 });
        for (const m of marks) if (m.kind & 2) vlines.push({ x: m.t, color: "--viz-ms2", dash: [7, 3], width: 2.5 });
        const c = track(P3.plot.line(chartEl, {
          height: 240, series, vlines, dots, xLabel: "Aligned RT (s)", yLabel: "Intensity",
          xFmt: (v, d) => fixed(v, Math.min(d, 1)),
          label: (si) => series[si].label,
          tip: (si, k) => { const st = statusOf(series[si].id); return `<b>${esc(series[si].label)}</b><br>RT ${fixed(e.x[k], 1)} s · ${sci(series[si].y[k])}${st != null ? "<br>" + STATUS[st] : ""}`; },
        }));
        detailCharts.push(c);
        parts.push(P3.plot.legend([
          ...(hasStatus ? [{ kind: "line", color: S1, label: "detected by the picker" }, { kind: "line", color: S2, label: "gap-filled" }] : []),
          { kind: "dash", color: "--viz-ink-2", label: "peak start / end" }, { kind: "dot", color: S1, label: "apex" },
          ...(marks.length ? [{ kind: "dash", color: "--viz-ms2", label: "MS2 in the consensus" }, { kind: "dash", bold: true, color: "--viz-ms2", label: "highest-TIC MS2" }] : [])]));
      } else {
        parts.push(h("div.section-title", h("h3.title-small", "Chromatograms")),
          banner(R.h.eic ? "Not stored for this feature: the report keeps chromatograms for the most intense features only (peak3d report --eic-budget raises the limit)." :
            `Not available: ${R.h.eic_note || "no point-cloud cache"}. Re-run with --keep-cache, or build the report while the cache exists.`));
      }

      if (R.m2) parts.push(...ms2Section(g, ann));

      if (R.a.m_q) {
        const vals = R.stems.map((_, f) => R.value(g, f));
        const cols = R.stems.map((_, f) => (statusOf(f) === 3 ? S1 : S2));
        const barEl = h("div");
        parts.push(h("div.section-title", h("h3.title-small", `${R.quant} in each file`)), barEl,
          P3.plot.legend([{ color: S1, label: "detected" }, { color: S2, label: "gap-filled" }]));
        const c = track(P3.plot.bars(barEl, {
          height: 180, values: vals, colors: cols, xLabel: "File", yLabel: R.quant,
          xTick: (i) => String(i + 1), tip: (f) => `<b>${esc(R.stems[f])}</b><br>${sci(vals[f])} · ${STATUS[statusOf(f) || 0]}`,
        }));
        detailCharts.push(c);
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
      el.replaceChildren(card(".elevated", null, null, ...parts));
    }

    // the user's annotation of the feature, apart from the search results: the library's best consensus hit by default,
    // or the hit the user picked, or "unknown"; set in the app (opts.annotate), shown read-only in a report
    function annotationSection(g) {
      const el = h("div");
      const ctl = { refresh: () => {} };
      const draw = () => {
        const x = R.annotation(g), user = R.ann.has(g);
        const out = [h("div.section-title", h("h3.title-small", "User annotation"), h("p.body-small.muted",
          `What this feature is taken to be. By default the best library hit on the consensus spectrum, when it scores at least ${fixed(R.minScore, 2)}; ` +
          "pick another hit from the search results below, or mark the feature unknown. Used in the report and the exported tables."))];
        if (!x) {
          const bs = R.bestScore(g);
          out.push(h("p.body-medium.muted", !R.id ? "Not annotated: no identity search yet." : isFinite(bs) ?
            `Not annotated: the best hit scores ${fixed(bs, 2)}, below ${fixed(R.minScore, 2)}.` : "Not annotated: no library hit."));
        } else if (x.unknown) {
          out.push(h("div.ann-head", h("span.chip.static.warn", icon("cancel"), "Unknown · set by you")));
        } else {
          out.push(h("div.ann-head", h("span.chip.static" + (user ? ".info" : ".ok"), icon(user ? "tune" : "check_circle"), user ? "Your choice" : "Library search result"),
            h("b.ann-name", x.name)), kv([["Adduct", x.adduct], ["Formula", x.formula],
            ["Score", `${fixed(x.score, 3)} on the ${x.spectrum === "consensus" ? "consensus" : "highest-TIC"} spectrum${x.rank > 1 ? `, hit ${x.rank}` : ""}`],
            ["InChIKey", x.inchikey ? h("code.mono", x.inchikey) : null]]));
        }
        if (opts.annotate && R.id) {
          const pick = state.hit != null ? R.hit(state.hit) : null;
          const same = pick && x && !x.unknown && x.name === pick.name && x.spectrum === pick.spectrum && x.rank === pick.rank;
          const short = (t) => (t.length > 36 ? t.slice(0, 34) + "…" : t);
          const use = h("button.btn.tonal.state", { type: "button", disabled: pick && !same ? null : true,
            title: pick ? `${pick.name} · score ${fixed(pick.score, 3)} on the ${pick.spectrum} spectrum` : "select a hit in the search results below" },
            icon("check"), pick && !same ? `Use “${short(pick.name)}”` : "Use the selected hit");
          use.addEventListener("click", () => setAnnotation(g, { spectrum: pick.spectrum, rank: pick.rank }));
          const unk = h("button.btn.outlined.state", { type: "button", disabled: x && x.unknown ? true : null }, icon("cancel"), "Mark unknown");
          unk.addEventListener("click", () => setAnnotation(g, "unknown"));
          const reset = h("button.btn.text.state", { type: "button", disabled: user ? null : true }, icon("history"), "Back to the library result");
          reset.addEventListener("click", () => setAnnotation(g, null));
          out.push(h("div.row.wrap.ann-actions", use, unk, reset));
        } else if (R.ann.size && !opts.annotate) {
          out.push(h("p.body-small.muted", "Annotations are set in the peak3d app."));
        }
        el.replaceChildren(...out);
      };
      ctl.refresh = draw;
      draw();
      return { el, refresh: () => ctl.refresh() };
    }

    async function setAnnotation(g, hit) {
      try {
        const r = await opts.annotate(R.a.g_id[g], hit);
        const all = {};
        for (const [k, v] of R.ann) all[k] = v;
        if (r && r.annotation) all[g] = r.annotation; else delete all[g];
        R.setAnnotations(all);
        P3.snack(hit === null ? "Back to the library result" : hit === "unknown" ? "Marked unknown" : `Annotated as ${r.annotation.name}`);
        if (repaintRows) repaintRows();
        if (detailEl && state.sel === g) renderDetail(detailEl, g);
      } catch (e) {
        P3.snack(String(e.message || e));
      }
    }

    // one of the feature's two MS2 copies (the consensus by default; the highest-TIC spectrum on request), mirrored
    // against the selected library hit, and that copy's search results: every hit above 0.1, weaker than the
    // annotation threshold greyed
    function ms2Section(g, ann) {
      const kc = R.ms2cAt.get(g), ks = R.ms2At.get(g);
      const n = R.nms2[g];
      if (ks == null) return [h("div.section-title", h("h3.title-small", "MS2 spectrum")), h("p.body-medium.muted", "No MS2 spectrum was acquired for this feature.")];
      if (state.copy == null || (state.copy === "consensus" && kc == null)) state.copy = kc != null ? "consensus" : "single";
      const copy = state.copy, k = copy === "consensus" ? kc : ks, M = copy === "consensus" ? R.m2c : R.m2;
      const merged = kc != null ? R.m2c.a.ms2c_merged[kc] : 0, files = kc != null ? R.m2c.a.ms2c_files[kc] : 0;
      const sel = h("select", { "aria-label": "Which MS2 spectrum" },
        kc != null ? h("option", { value: "consensus" }, `Consensus of ${merged} spectr${merged === 1 ? "um" : "a"}`) : null,
        h("option", { value: "single" }, "Highest-TIC spectrum (MassCube's choice)"));
      sel.value = copy;
      sel.addEventListener("change", () => { state.copy = sel.value; state.hit = null; renderDetail(detailEl || sel.closest(".detail"), g); });
      const where = `${M.h.file[k]}, scan ${M.h.scan[k]}, ${fixed(M.a[copy === "consensus" ? "ms2c_rt" : "ms2_rt"][k], 1)} s`;
      const desc = copy === "consensus" ?
        `${merged} of the ${int(n)} MS2 spectra mapped to this feature${files > 1 ? `, from ${files} files` : ""}, merged: a peak stays when at least two spectra hold it, intensities weighted by precursor abundance${merged === 1 ? " (only one usable spectrum: it is shown as it is)" : ""}. Most abundant precursor: ${where}.` :
        `The spectrum with the highest total ion current of the ${int(n)} mapped to this feature: ${where}, precursor m/z ${fixed(R.m2.a.ms2_prec[ks], 4)}.`;
      const out = [h("div.section-title", h("h3.title-small", "MS2 spectrum and library search"), h("p.body-small.muted", desc)),
        h("div.copy-pick", field("Spectrum", sel, h("span.select-arrow", icon("expand_more")), "300px"))];
      const hits = R.hits(g, copy === "consensus" ? 0 : 1);
      if (state.hit == null || !hits.includes(state.hit)) state.hit = hits.length ? hits[0] : null;
      const spec = () => {
        const b = state.hit != null ? R.hit(state.hit) : null;
        return { top: R.spectrum(copy, k), bottom: b && b.spec ? b.spec : null, bottomLabel: b ? b.name : null };
      };
      const specEl = h("div");
      const chart = track(P3.plot.mirror(specEl, Object.assign({ height: 250, topLabel: copy === "consensus" ? "consensus" : "measured", tol: (R.id && R.id.h.fragment_tol) || 0.02 }, spec())));
      detailCharts.push(chart);
      out.push(specEl);
      if (!R.id) return out.concat(h("p.body-small.muted", "No identity search yet."));
      out.push(P3.plot.legend([{ color: S1, label: "measured, matched" }, { color: S2, label: "library, matched" }, { color: "--viz-neutral", label: "unmatched" }]));
      if (!hits.length) return out.concat(banner(`No library spectrum scoring above ${fixed(R.id.h.list_min != null ? R.id.h.list_min : 0.1, 1)} (${R.id.h.method_label || R.id.h.method}) within the precursor tolerance.`));
      const more = h("div");
      const showMore = () => {
        const b = R.hit(state.hit);
        more.replaceChildren(kv([["InChIKey", b.inchikey ? h("code.mono", b.inchikey) : null], ["SMILES", b.smiles ? h("code.mono", b.smiles) : null],
          ["Library precursor", isFinite(b.prec) ? `m/z ${fixed(b.prec, 4)} (${fixed(b.ppm, 1)} ppm)` : null]]));
      };
      const x = R.annotation(g);
      const tb = h("tbody", hits.map((j) => {
        const b = R.hit(j);
        const chosen = x && !x.unknown && x.name === b.name && x.spectrum === b.spectrum && x.rank === b.rank;
        const tr = h("tr" + (b.score < R.minScore ? ".below" : ""), { "aria-selected": String(j === state.hit) }, h("td.r", b.rank),
          h("td.ident", { title: b.name }, h("span.id-name", b.name), chosen ? h("span.id-user", "annotation") : null),
          h("td.r", fixed(b.score, 3)), h("td", b.adduct || "–"), h("td", b.formula || "–"), h("td.r", fixed(b.ppm, 1)), h("td.r", int(b.nmatch)));
        tr.addEventListener("click", () => {
          state.hit = j;
          tb.querySelectorAll("tr").forEach((r) => r.setAttribute("aria-selected", String(r === tr)));
          chart.update(spec());
          showMore();
          if (ann) ann.refresh();
        });
        return tr;
      }));
      showMore();
      if (ann) ann.refresh();
      const nBelow = hits.filter((j) => R.id.a.id_score[j] < R.minScore).length;
      return out.concat(h("p.body-small.muted", { style: { margin: "12px 0 0" } },
        `${int(hits.length)} hit${hits.length === 1 ? "" : "s"} above ${fixed(R.id.h.list_min != null ? R.id.h.list_min : 0.1, 1)}` +
        (nBelow ? `; ${int(nBelow)} below the annotation threshold ${fixed(R.minScore, 2)} (grey)` : "") + ". Click one to compare; use it as the annotation above."),
        h("div.table-wrap.hits.scroll", h("table.data", h("thead", h("tr", ["#", "Library match", "Score", "Adduct", "Formula", "Δ ppm", "Peaks"]
          .map((t, i) => h("th" + ([0, 2, 5, 6].includes(i) ? ".r" : ""), t)))), tb)), more);
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
          ["Kernel", `${R.kernelLabel} ${R.kernelVersion}`], ["Picker variant", p.variant], ["peak3d", R.kernel !== "peak3d" ? p.peak3d_version : null],
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

  P3.viewer = { mount, Results, bundleTable };
})(window.P3 = window.P3 || {});
