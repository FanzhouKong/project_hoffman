// GC-MS compounds (the viewer's Compounds tab): the deconvolved compounds of a GC-MS output (peak3d gc, or peak3d
// process --mode gc), each listed at its quantifier ion's m/z and its retention time, followed by its spectrum, its
// library hits, its ions' chromatograms (the app only: read from the point-cloud cache) and its abundance in every
// file. Reads the block peak3d/web/gcview.py packs: the app's /api/results/gc, a report's peak3d-gc block.
(function (P3) {
  "use strict";
  const { h, icon, fixed, int, sci } = P3;
  const S1 = "--viz-series-1", S2 = "--viz-series-2";
  const CAT = ["--viz-cat-1", "--viz-cat-2", "--viz-cat-3", "--viz-cat-4", "--viz-cat-5", "--viz-cat-6", "--viz-cat-7", "--viz-cat-8"];
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const card = (cls, ...body) => h("section.card" + cls, ...body);
  const kv = (pairs) => h("dl.kv.body-medium", pairs.filter((p) => p && p[1] != null && p[1] !== "").flatMap(([k, v]) => [h("dt", k), h("dd", v)]));
  const banner = (text, kind) => h("div.banner" + (kind === "error" ? ".error" : ""), icon(kind === "error" ? "error" : "info"), h("div.body-medium", text));
  const clean = (t) => String(t == null ? "" : t).replace(/[\t\r\n]+/g, " ");
  const unitMass = (mz) => { for (let i = 0; i < mz.length; i++) if (Math.abs(mz[i] - Math.round(mz[i])) > 0.02) return false; return mz.length > 0; };

  // ---------------------------------------------------------------- data access
  class Compounds {
    constructor({ header, a }) {
      this.h = header;
      this.a = a;
      this.n = header.n;
      this.stems = header.stems || [];
      this.N = this.stems.length;
      this.digits = header.digits === 1 ? 1 : 4;          // decimals worth showing in an m/z (1: unit-mass data)
      this.tol = this.digits === 1 ? 0.3 : 0.005;         // Da: a typed m/z matches an ion within this
      this.summary = header.summary || {};
      this.name = header.name || [];
      this.formula = header.formula || [];
      this.inchikey = header.inchikey || [];
      this.rtS = Float64Array.from(a.c_rt, (v) => v * 60);       // retention times are shown in seconds
      this.logH = Float32Array.from(a.c_height, (v) => (v > 0 ? Math.log10(v) : 0));
      this.hasRi = a.c_ri.some((v) => isFinite(v));
      this.hasHits = !!(a.h_off && a.h_rank && a.h_rank.length);
      this.nAnnotated = this.name.filter(Boolean).length;
    }
    mz(v) { return fixed(v, this.digits); }
    spectrum(k) {
      const o = this.a.s_off;
      return { mz: this.a.s_mz.subarray(o[k], o[k + 1]), inten: this.a.s_ab.subarray(o[k], o[k + 1]), shared: this.a.s_sh.subarray(o[k], o[k + 1]) };
    }
    value(k, f) { return this.a.q[k * this.N + f]; }
    hasIon(k, mz) {
      const o = this.a.s_off, m = this.a.s_mz;
      for (let i = o[k]; i < o[k + 1]; i++) { if (Math.abs(m[i] - mz) <= this.tol) return true; if (m[i] > mz + this.tol) break; }
      return false;
    }
    hits(k) {
      const out = [];
      if (this.hasHits) for (let j = this.a.h_off[k]; j < this.a.h_off[k + 1]; j++) out.push(j);
      return out;
    }
    hit(j) {
      const a = this.a, t = this.h.hits || {}, s = a.h_spec ? a.h_spec[j] : -1;
      return { rank: a.h_rank[j], match: a.h_match[j], reverse: a.h_reverse[j], nmatch: a.h_nmatch[j], lri: a.h_lri[j], dri: a.h_dri[j],
        name: (t.name || [])[j] || "", formula: (t.formula || [])[j] || "", inchikey: (t.inchikey || [])[j] || "", library: (t.library || [])[j] || "",
        spectrum: s >= 0 ? { mz: a.hs_mz.subarray(a.hs_off[s], a.hs_off[s + 1]), inten: a.hs_in.subarray(a.hs_off[s], a.hs_off[s + 1]) } : null };
    }
  }

  // the listed compounds as a tab-separated table: retention times in seconds, then the quantifier's abundance in
  // every file (empty: compound not found there)
  function table(G, rows) {
    const a = G.a, num = (v, d) => (isFinite(v) ? (+v).toFixed(d) : "");
    const head = ["compound_id", "rt_s", ...(G.hasRi ? ["ri"] : []), "quant_mz", "base_mz", "n_ions", "n_shared_ions", "n_files", "height",
      ...(G.hasHits ? ["name", "formula", "inchikey", "match", "reverse", "library_ri", "delta_ri"] : []), "reference_file", ...G.stems];
    const lines = [head.join("\t")];
    for (const k of rows) {
      const v = [a.c_id[k], num(G.rtS[k], 2), ...(G.hasRi ? [num(a.c_ri[k], 1)] : []), num(a.c_qmz[k], 5), num(a.c_bmz[k], 5), a.c_nions[k], a.c_nshared[k],
        a.c_nfiles[k], isFinite(a.c_height[k]) ? (+a.c_height[k]).toPrecision(6) : "",
        ...(G.hasHits ? [clean(G.name[k]), clean(G.formula[k]), clean(G.inchikey[k]), num(a.c_match[k], 4), num(a.c_reverse[k], 4), num(a.c_lri[k], 1), num(a.c_dri[k], 1)] : []),
        clean((G.h.ref_file || [])[k])];
      for (let f = 0; f < G.N; f++) { const x = G.value(k, f); v.push(isFinite(x) ? (+x).toPrecision(6) : ""); }
      lines.push(v.join("\t"));
    }
    return new Blob([lines.join("\n") + "\n"], { type: "text/tab-separated-values" });
  }
  // ... and their spectra as an EI .msp (intensities scaled to a base peak of 999, retention time in minutes)
  function msp(G, rows) {
    const a = G.a, out = [];
    for (const k of rows) {
      const s = G.spectrum(k);
      let top = 0;
      for (let i = 0; i < s.inten.length; i++) if (s.inten[i] > top) top = s.inten[i];
      out.push(`Name: compound ${a.c_id[k]}${G.name[k] ? " | " + clean(G.name[k]) : ""}`);
      if (isFinite(a.c_ri[k])) out.push(`Retention_index: ${(+a.c_ri[k]).toFixed(1)}`);
      out.push(`RetentionTime: ${(+a.c_rt[k]).toFixed(4)}`, `Quant_mz: ${(+a.c_qmz[k]).toFixed(5)}`, `Num Peaks: ${s.mz.length}`);
      for (let i = 0; i < s.mz.length; i++) out.push(`${s.mz[i].toFixed(5)} ${(top > 0 ? 999 * s.inten[i] / top : 0).toFixed(2)}`);
      out.push("");
    }
    return new Blob([out.join("\n") + "\n"], { type: "text/plain" });
  }

  // ---------------------------------------------------------------- the Compounds tab
  // ctx: { state (kept by the viewer between tab changes), track(chart), drop(list of charts), alive(), title,
  //        trace(compound_id, file) -> promise (the app only) }
  function render(body, G, ctx) {
    const a = G.a, st = ctx.state;
    for (const [k, v] of Object.entries({ sel: null, sortKey: "rt", sortDir: 1, query: "", ion: "", rtLo: "", rtHi: "", all: false, annotated: false,
      hit: null, file: null, norm: false })) if (!(k in st)) st[k] = v;
    const detailCharts = [];
    let shown = null;

    // ---- the list
    function rows() {
      const q = st.query.trim(), out = [];
      let idq = null, mzq = null, text = null;
      if (/^#\d+$/.test(q)) idq = +q.slice(1);
      else if (q && isFinite(+q)) mzq = +q;
      else if (q) text = q.toLowerCase();
      const ionq = st.ion !== "" && isFinite(+st.ion) ? +st.ion : null;
      const bound = (v, d) => (v !== "" && isFinite(+v) ? +v : d);
      const rtLo = bound(st.rtLo, -Infinity), rtHi = bound(st.rtHi, Infinity);
      for (let k = 0; k < G.n; k++) {
        if (G.rtS[k] < rtLo || G.rtS[k] > rtHi) continue;
        if (st.all && a.c_nfiles[k] < G.N) continue;
        if (st.annotated && !G.name[k]) continue;
        if (idq != null && a.c_id[k] !== idq) continue;
        if (mzq != null && Math.abs(a.c_qmz[k] - mzq) > G.tol) continue;
        if (text != null && !String(G.name[k]).toLowerCase().includes(text)) continue;
        if (ionq != null && !G.hasIon(k, ionq)) continue;
        out.push(k);
      }
      const key = { id: a.c_id, qmz: a.c_qmz, rt: a.c_rt, ri: a.c_ri, bmz: a.c_bmz, nions: a.c_nions, nfiles: a.c_nfiles, height: a.c_height, match: a.c_match }[st.sortKey];
      if (st.sortKey === "name") {       // annotated first, by name
        const nm = (k) => (G.name[k] ? String(G.name[k]).toLowerCase() : null);
        out.sort((p, r) => { const x = nm(p), y = nm(r); return x == null ? (y == null ? 0 : 1) : y == null ? -1 : st.sortDir * x.localeCompare(y); });
      } else if (key) out.sort((p, r) => { const x = key[p], y = key[r]; return isNaN(x) ? (isNaN(y) ? 0 : 1) : isNaN(y) ? -1 : st.sortDir * (x - y); });
      return out;
    }

    const search = h("input", { type: "search", placeholder: "Quantifier m/z, #id or name", value: st.query, "aria-label": "Search quantifier m/z, #compound id or name" });
    const searchField = h("div.field.dense", { title: `A number lists the compounds quantified on that m/z (within ${G.tol} Da), #12 the compound with that id, text the annotated compounds whose name holds it` },
      h("div.box", h("span.lead", icon("search", "sm")), search, h("label", "Quantifier m/z, #id or name")));
    const ionIn = h("input", { type: "number", step: "any", placeholder: "Spectrum has m/z", value: st.ion, "aria-label": "Spectrum has this m/z" });
    const ionField = h("div.field.dense", { style: { width: "150px" }, title: `Only the compounds whose spectrum holds an ion at this m/z (within ${G.tol} Da)` },
      h("div.box", ionIn, h("label", "Spectrum has m/z")));
    const rangeIn = {};
    const rtBox = (key, ph) => {
      const i = rangeIn[key] = h("input", { type: "number", step: "any", placeholder: ph, value: st[key], "aria-label": `RT ${ph}` });
      i.addEventListener("input", () => { st[key] = i.value.trim(); later(); });
      return i;
    };
    const rtRange = h("div.range", { title: "Retention-time window (s); leave an end empty to keep it open" },
      h("span.range-label", "RT"), rtBox("rtLo", "from"), h("span.range-sep", "–"), rtBox("rtHi", "to"), h("span.range-unit", "s"));
    const chip = (key, label, title) => {
      const c = h("button.chip.state", { type: "button", "aria-pressed": String(st[key]), title }, st[key] ? icon("check") : null, label);
      c.addEventListener("click", () => { st[key] = !st[key]; c.setAttribute("aria-pressed", String(st[key])); c.replaceChildren(...(st[key] ? [icon("check")] : []), label); scroller.scrollTop = 0; refresh(); });
      return c;
    };
    const FIELDS = ["query", "ion", "rtLo", "rtHi"], CHIPS = ["all", "annotated"];
    const filtering = () => FIELDS.some((k) => String(st[k]).trim() !== "") || CHIPS.some((k) => st[k]);
    const clearBtn = h("button.btn.text.state.clear-filters", { type: "button", title: "Clear the search, the spectrum m/z, the RT window and the chips" }, icon("close"), "Clear filters");
    clearBtn.addEventListener("click", () => { for (const k of FIELDS) st[k] = ""; for (const k of CHIPS) st[k] = false; body.replaceChildren(); ctx.redraw(); });
    const count = h("span.body-small.muted");
    const dl = h("button.btn.outlined.state", { type: "button", title: "The compounds listed here (search, filters and sort applied): quantifier m/z, retention time in seconds, the annotation, and the quantifier's abundance in every file. The exact tables are gc/compounds.tsv and gc/compound_table.tsv in the output folder (minutes)." }, icon("download"), "Download table");
    dl.addEventListener("click", () => {
      const name = `${ctx.title || G.h.title}_compounds${list.length < G.n ? "_" + list.length : ""}.tsv`;
      P3.saveBlob(table(G, list), name);
      P3.snack(`Downloaded ${int(list.length)} compounds as ${name}`);
    });
    const dls = h("button.btn.outlined.state", { type: "button", title: "The spectra of the compounds listed here as an EI .msp file (intensities scaled to a base peak of 999), to search in other software" }, icon("download"), "Download spectra");
    dls.addEventListener("click", () => {
      const name = `${ctx.title || G.h.title}_compounds${list.length < G.n ? "_" + list.length : ""}.msp`;
      P3.saveBlob(msp(G, list), name);
      P3.snack(`Downloaded ${int(list.length)} spectra as ${name}`);
    });

    // each compound at its quantifier m/z and retention time first, then what else is known of it
    const COLS = [["id", "ID"], ["qmz", "Quantifier m/z"], ["rt", "RT (s)"], ...(G.hasRi ? [["ri", "RI"]] : []),
      ...(G.hasHits ? [["name", "Annotation"], ["match", "Match"]] : []), ["nions", "Ions"], ["nfiles", "Files"], ["height", "Height"], ["bmz", "Base peak m/z"]];
    const DESC = new Set(["height", "nions", "nfiles", "match"]);
    const TIPS = { qmz: "The quantifier: the strongest ion of the compound's spectrum that no other compound shares; its abundance is the compound's value in each file",
      bmz: "The strongest ion of the spectrum", nions: "Ions in the compound's spectrum", nfiles: "Files the compound was found in",
      height: "Abundance of the quantifier ion in the file where the compound is strongest", match: "Library match of the annotation",
      ri: "Retention index from the run's ladder" };
    const thead = h("thead", h("tr", COLS.map(([k, label]) => {
      const th = h("th.sortable" + (k === "name" ? "" : ".r"), { "data-k": k, title: TIPS[k] || null }, label);
      th.addEventListener("click", () => {
        if (st.sortKey === k) st.sortDir *= -1; else { st.sortKey = k; st.sortDir = DESC.has(k) ? -1 : 1; }
        refresh();
      });
      return th;
    })));
    const tbody = h("tbody");
    const scroller = h("div.table-wrap.vtable", { tabindex: "0" }, h("table.data.gc-table", thead, tbody));
    const mapEl = h("div.minimap");
    const headBox = h("div"), specBox = h("div"), chromBox = h("div");
    const detail = h("div.detail", h("div.detail-grid",
      h("div.detail-head", card(".elevated.head-card", headBox)),
      h("div.detail-col", card(".elevated", specBox)),
      h("div.detail-col", card(".elevated", chromBox))));
    const s = G.summary;
    const facts = [`${int(G.n)} compounds from ${int(s.n_components)} components in ${int(G.N)} file${G.N > 1 ? "s" : ""}` +
      (s.min_presence > 1 ? ` (each found in at least ${int(s.min_presence)})` : ""),
      s.n_ladder ? `retention index from ${int(s.n_ladder)} ladder compounds${s.ladder ? ` (${s.ladder === "fame" ? "FAMEs" : "n-alkanes"})` : ""}` : null,
      s.libraries && s.libraries.length ? `${int(G.nAnnotated)} annotated at match ≥ ${fixed(s.min_similarity, 2)} (${s.libraries.join(", ")})` : "no EI library searched"].filter(Boolean).join(" · ");
    const note = s.picked_in && s.picked_in !== "gc" ? banner(`These files were picked in ${String(s.picked_in).toUpperCase()} mode; the GC-MS mode also keeps fragment peaks that ride on a continuous background of the same ion.`) : null;
    const filters = [note, h("p.body-small.muted", { style: { margin: "0 0 8px" } }, facts), h("div.toolbar", searchField, ionField, rtRange),
      h("div.toolbar", G.N > 1 ? chip("all", "In every file", "Only the compounds found in every file") : null,
        G.hasHits ? chip("annotated", "Annotated", "Only the compounds with a library annotation") : null, clearBtn, h("span.spacer"), count, dl, dls)];
    const mapCard = card(".elevated.map-card", h("div.section-title.first", h("h3.title-small", "Compound map"),
      h("p.body-small.muted", "Every compound at its retention time and quantifier m/z; the listed ones in colour, the selected one ringed. Drag a box to set the RT window; double-click to clear.")), mapEl);
    body.append(h("div.feature-split.gc-split", h("div.list-col", h("div.filters", ...filters), h("div", scroller), mapCard), detail));

    let lo = Infinity, hi = -Infinity;
    for (let k = 0; k < G.n; k++) if (a.c_height[k] > 0) { lo = Math.min(lo, G.logH[k]); hi = Math.max(hi, G.logH[k]); }
    const mini = ctx.track(P3.plot.scatter(mapEl, {
      height: 210, x: G.rtS, y: a.c_qmz, v: G.logH, vDomain: [lo, hi], r: G.n > 20000 ? 1 : G.n > 3000 ? 1.5 : 2.4,
      xLabel: "RT (s)", yLabel: "Quantifier m/z", xFmt: (v, d) => fixed(v, Math.min(d, 0)), yFmt: (v, d) => fixed(v, Math.min(d, 0)),
      selected: st.sel, brushHint: "Drag to set the RT window · double-click to clear",
      onBrush: (xr) => {
        st.rtLo = xr ? fixed(xr[0], 1) : ""; st.rtHi = xr ? fixed(xr[1], 1) : "";
        rangeIn.rtLo.value = st.rtLo; rangeIn.rtHi.value = st.rtHi;
        scroller.scrollTop = 0;
        refresh();
      },
      tip: (k) => `<b>#${a.c_id[k]} · m/z ${G.mz(a.c_qmz[k])}</b><br>RT ${fixed(G.rtS[k], 1)} s · ${int(a.c_nions[k])} ions` + (G.name[k] ? `<br>${esc(G.name[k])}` : ""),
      onClick: (k) => { select(k); const i = list.indexOf(k); if (i >= 0) ensureVisible(i); },
    }));

    let list = [], rowH = 33, tmr = null;
    const later = () => { clearTimeout(tmr); tmr = setTimeout(() => { scroller.scrollTop = 0; refresh(); }, 150); };
    const CELL = {
      id: (k) => h("td.r", int(a.c_id[k])),
      qmz: (k) => h("td.r.num.gc-q", G.mz(a.c_qmz[k])),
      rt: (k) => h("td.r", fixed(G.rtS[k], 1)),
      ri: (k) => h("td.r", { title: a.c_riex[k] ? "beyond the ladder: extrapolated" : null }, isFinite(a.c_ri[k]) ? fixed(a.c_ri[k], 0) + (a.c_riex[k] ? "*" : "") : "–"),
      name: (k) => (G.name[k] ? h("td.ident", { title: G.name[k] }, h("span.id-name", G.name[k])) : h("td.ident.muted", G.hits(k).length ? "below threshold" : "no match")),
      match: (k) => h("td.r.score.num", isFinite(a.c_match[k]) ? fixed(a.c_match[k], 2) : "–"),
      nions: (k) => h("td.r", int(a.c_nions[k])),
      nfiles: (k) => h("td.r", `${a.c_nfiles[k]}/${G.N}`),
      height: (k) => h("td.r", sci(a.c_height[k])),
      bmz: (k) => h("td.r", G.mz(a.c_bmz[k])),
    };
    function rowEl(k) {
      const tr = h("tr", { "aria-selected": String(k === st.sel), "data-k": k }, COLS.map(([c]) => CELL[c](k)));
      tr.addEventListener("click", () => select(k));
      return tr;
    }
    function paintRows() {
      const top = scroller.scrollTop, hgt = scroller.clientHeight || 560, head = 45;
      const first = Math.max(0, Math.floor((top - head) / rowH) - 10);
      const last = Math.min(list.length, Math.ceil((top + hgt) / rowH) + 10);
      const pad = (n) => h("tr", h("td", { colspan: COLS.length, style: { height: n * rowH + "px", padding: 0, border: 0 } }));
      const kids = [pad(first)];
      for (let i = first; i < last; i++) kids.push(rowEl(list[i]));
      kids.push(pad(list.length - last));
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
      const i = list.indexOf(st.sel);
      const j = Math.max(0, Math.min(list.length - 1, (i < 0 ? -1 : i) + (e.key === "ArrowDown" ? 1 : -1)));
      if (list.length) { select(list[j]); ensureVisible(j); }
    });
    function ensureVisible(i) {
      const y = 45 + i * rowH;
      if (y < scroller.scrollTop + 45) scroller.scrollTop = y - 45;
      else if (y + rowH > scroller.scrollTop + scroller.clientHeight) scroller.scrollTop = y + rowH - scroller.clientHeight;
    }
    function refresh() {
      list = rows();
      clearBtn.disabled = !filtering();
      count.textContent = `${int(list.length)} of ${int(G.n)} compounds`;
      const emph = new Uint8Array(G.n);
      for (const k of list) emph[k] = 1;
      const num = (v, d) => (v !== "" && isFinite(+v) ? +v : d);
      const boxed = st.rtLo !== "" || st.rtHi !== "";
      mini.update({ emph: list.length < G.n ? emph : null, selected: st.sel, box: boxed ? [[num(st.rtLo, -Infinity), num(st.rtHi, Infinity)], [-Infinity, Infinity]] : null });
      thead.querySelectorAll("th").forEach((th) => {
        if (th.querySelector("svg")) th.querySelector("svg").remove();
        if (th.dataset.k === st.sortKey) th.appendChild(icon(st.sortDir > 0 ? "arrow_upward" : "arrow_downward"));
      });
      paintRows();
    }
    search.addEventListener("input", () => { st.query = search.value; later(); });
    ionIn.addEventListener("input", () => { st.ion = ionIn.value.trim(); later(); });

    function select(k) {
      if (k !== st.sel) { st.hit = null; st.file = null; }
      st.sel = k;
      tbody.querySelectorAll("tr[data-k]").forEach((tr) => tr.setAttribute("aria-selected", String(+tr.dataset.k === k)));
      mini.update({ selected: k });
      renderDetail(k);
    }

    // ---- the detail: a header, the spectrum with its library hits, and the chromatograms with the per-file values
    function renderDetail(k) {
      ctx.drop(detailCharts);
      shown = k;
      if (k == null) {
        headBox.replaceChildren(h("div.empty", icon("science", "lg"), h("h2.title-medium", "No compound selected"), h("p.body-medium", "Pick a row in the table or a point on the compound map.")));
        specBox.replaceChildren();
        chromBox.replaceChildren();
        return;
      }
      const all = a.c_nfiles[k] === G.N;
      headBox.replaceChildren(h("div.head-row", h("div.head-main",
        h("div.head-line", h("span.label-medium.muted", `Compound #${a.c_id[k]}`),
          h("span.title-large.num", { title: "The quantifier ion: the strongest ion of the spectrum that no other compound shares" }, `m/z ${G.mz(a.c_qmz[k])}`),
          h("span.body-medium.muted", `RT ${fixed(G.rtS[k], 1)} s` + (isFinite(a.c_ri[k]) ? ` · RI ${fixed(a.c_ri[k], 0)}${a.c_riex[k] ? " (beyond the ladder)" : ""}` : "")),
          G.name[k] ? h("span.body-medium.id-best", h("b", G.name[k]), ` · ${fixed(a.c_match[k], 2)}`) : null),
        h("div.head-line",
          h("span.chip.static" + (all ? ".ok" : ".info"), icon(all ? "check_circle" : "info"), `Found in ${a.c_nfiles[k]} of ${G.N} files`),
          h("span.chip.static", `${int(a.c_nions[k])} ions`),
          a.c_nshared[k] ? h("span.chip.static", { title: "Ions whose chromatogram also needed another compound's elution profile" }, `${int(a.c_nshared[k])} shared with a co-eluting compound`) : null,
          h("span.chip.static", { title: "The strongest ion of the spectrum" }, `base peak m/z ${G.mz(a.c_bmz[k])}`)))));
      specBox.replaceChildren(...spectrumColumn(k));
      chromBox.replaceChildren(...chromColumn(k));
    }

    function spectrumColumn(k) {
      const out = [], sp = G.spectrum(k), hits = G.hits(k);
      if (st.hit == null || !hits.includes(st.hit)) st.hit = hits.length ? hits[0] : null;
      const refFile = (G.h.ref_file || [])[k] || "";
      const spec = () => {
        const b = st.hit != null ? G.hit(st.hit) : null, lib = b && b.spectrum;
        return { top: { mz: sp.mz, inten: sp.inten }, bottom: lib || null, prec: a.c_qmz[k], precText: false, mzDigits: G.digits,
          tol: G.digits === 1 || (lib && unitMass(lib.mz)) ? 0.5 : 0.003, ppm: G.digits === 1 || (lib && unitMass(lib.mz)) ? 0 : 10,
          topLabel: `compound #${a.c_id[k]}`, bottomLabel: b ? b.name : "", score: b && lib ? fixed(b.match, 3) : "",
          peakNote: (si, i) => (si === 0 && sp.shared[i] ? "<br>shared with a co-eluting compound" : "") };
      };
      const desc = h("p.body-small.muted");
      const paintDesc = () => {
        const b = st.hit != null ? G.hit(st.hit) : null;
        desc.textContent = `The compound's deconvolved spectrum (${int(sp.mz.length)} ions, from ${refFile || "its strongest file"}), each ion at the abundance its chromatogram holds of the compound's elution profile` +
          (b ? (b.spectrum ? `, against the library spectrum of ${b.name} below.` : `. The library spectrum of candidate ${b.rank} is not stored (the best three are).`) : ".") +
          " Drag across to zoom.";
      };
      paintDesc();
      out.push(h("div.section-title.first", h("h3.title-small", "Spectrum"), desc));
      const specEl = h("div");
      const chart = ctx.track(P3.plot.mirror(specEl, Object.assign({ height: 280 }, spec())));
      detailCharts.push(chart);
      out.push(specEl, P3.plot.legend([...(hits.length ? [{ color: S1, label: "measured, matched" }, { color: S2, label: "library, matched" }, { color: "--viz-neutral", label: "unmatched" }] :
        [{ color: S1, label: "measured" }]), { kind: "dash", color: "--viz-muted", label: "quantifier ion" }]));

      // library hits
      out.push(h("div.section-title", h("h3.title-small", `Library hits${hits.length ? ` (${int(hits.length)})` : ""}`)));
      if (!G.hasHits && !(G.summary.libraries || []).length) {
        out.push(h("p.body-medium.muted", "No EI library was searched. Give one in the run form's GC-MS card (or `peak3d gc OUT --library LIB.msp`) to annotate the compounds."));
      } else if (!hits.length) {
        out.push(banner("No library spectrum matched this compound's spectrum" + (isFinite(G.summary.ri_tol) && G.summary.ri_tol != null ? ` within the retention-index window (±${fixed(G.summary.ri_tol, 0)}).` : ".")));
      } else {
        const manyLibs = new Set((G.h.hits.library || [])).size > 1;
        const cols = ["#", "Library match", "Match", "Reverse", "Peaks", ...(G.hasRi ? ["Library RI", "Δ RI"] : []), "Formula", ...(manyLibs ? ["Library"] : [])];
        const right = new Set(["#", "Match", "Reverse", "Peaks", "Library RI", "Δ RI"]);
        const min = G.summary.min_similarity != null ? G.summary.min_similarity : 0.7;
        const tb = h("tbody", hits.map((j) => {
          const b = G.hit(j);
          const tr = h("tr" + (b.match < min ? ".below" : ""), { "aria-selected": String(j === st.hit) }, h("td.r", b.rank),
            h("td.ident", { title: b.name }, h("span.id-name", b.name), b.rank === 1 && G.name[k] ? h("span.id-user", "annotation") : null),
            h("td.r", fixed(b.match, 3)), h("td.r", fixed(b.reverse, 3)), h("td.r", int(b.nmatch)),
            ...(G.hasRi ? [h("td.r", isFinite(b.lri) ? fixed(b.lri, 0) : "–"), h("td.r", isFinite(b.dri) ? fixed(b.dri, 0) : "–")] : []),
            h("td", b.formula || "–"), ...(manyLibs ? [h("td.id-db", b.library)] : []));
          tr.addEventListener("click", () => {
            st.hit = j;
            tb.querySelectorAll("tr").forEach((r) => r.setAttribute("aria-selected", String(r === tr)));
            chart.update(spec());
            paintDesc();
          });
          return tr;
        }));
        out.push(h("div.table-wrap.hits.scroll", h("table.data", h("thead", h("tr", cols.map((t) => h("th" + (right.has(t) ? ".r" : ""), t)))), tb)),
          h("p.body-small.muted", { style: { margin: "6px 0 0" } }, `Best first; the best hit is the annotation when its match reaches ${fixed(min, 2)} (grey: below). ` +
            "Match compares the whole spectrum, Reverse only the library spectrum's ions. Click one to compare."));
      }

      // the ions, strongest first
      const order = Array.from(sp.mz.keys()).sort((i, j) => sp.inten[j] - sp.inten[i]);
      const top = order.length ? sp.inten[order[0]] : 1;
      const qi = order.length ? order.reduce((b, i) => (Math.abs(sp.mz[i] - a.c_qmz[k]) < Math.abs(sp.mz[b] - a.c_qmz[k]) ? i : b), order[0]) : -1;
      out.push(h("details.expand.gc-ions", h("summary.state", icon("table_rows"), `Ions (${int(sp.mz.length)})`, h("span.spacer"), icon("expand_more", "chev")),
        h("div.table-wrap.scroll", { style: { maxHeight: "260px" } }, h("table.data.static", h("thead", h("tr", h("th.r", "m/z"), h("th.r", "Relative (%)"), h("th.r", "Abundance"), h("th", ""))),
          h("tbody", order.map((i) => h("tr", h("td.r", G.mz(sp.mz[i])), h("td.r", fixed(top > 0 ? 100 * sp.inten[i] / top : 0, 1)), h("td.r", sci(sp.inten[i])),
            h("td.muted", [i === qi ? "quantifier" : null, sp.shared[i] ? "shared" : null].filter(Boolean).join(" · ")))))))));
      return out;
    }

    function chromColumn(k) {
      const out = [];
      // the chromatograms of its ions: read by the app's server from the point-cloud cache
      if (ctx.trace) {
        const box = h("div");
        out.push(h("div.section-title.first", h("h3.title-small", "Chromatograms of its ions"),
          h("p.body-small.muted", "The quantifier and the compound's other strongest ions in one file, on that file's own time axis: ions of one compound rise and fall together.")), box);
        loadTrace(k, box);
      }
      // the quantifier's abundance in each file
      const vals = G.stems.map((_, f) => { const v = G.value(k, f); return isFinite(v) ? v : 0; });
      const barEl = h("div");
      out.push(h("div.section-title" + (ctx.trace ? "" : ".first"), h("h3.title-small", "Quantifier abundance in each file"),
        h("p.body-small.muted", `Abundance of m/z ${G.mz(a.c_qmz[k])} in each file's own deconvolved spectrum of this compound; no bar: not found there.`)), barEl);
      detailCharts.push(ctx.track(P3.plot.bars(barEl, {
        height: 170, values: vals, color: S1, xLabel: "File", yLabel: "Abundance",
        xTick: (i) => String(i + 1), tip: (f) => `<b>${esc(G.stems[f])}</b><br>${isFinite(G.value(k, f)) ? sci(vals[f]) : "not found"}`,
      })));
      out.push(h("div.section-title", h("h3.title-small", "Details")), kv([
        ["Quantifier m/z", G.mz(a.c_qmz[k])], ["Base peak m/z", G.mz(a.c_bmz[k])],
        ["Retention time", `${fixed(G.rtS[k], 2)} s (${fixed(a.c_rt[k], 3)} min, aligned)`],
        ["Retention index", isFinite(a.c_ri[k]) ? fixed(a.c_ri[k], 1) + (a.c_riex[k] ? " (extrapolated beyond the ladder)" : "") : null],
        ["Ions", `${int(a.c_nions[k])}${a.c_nshared[k] ? `, ${int(a.c_nshared[k])} shared with a co-eluting compound` : ""}`],
        ["Spectrum from", `${(G.h.ref_file || [])[k] || "–"} (component ${int(a.c_refc[k])})`],
        ["Annotation", G.name[k] ? `${G.name[k]}${G.formula[k] ? " · " + G.formula[k] : ""}` : null], ["InChIKey", G.inchikey[k] || null],
        ["Match / reverse", isFinite(a.c_match[k]) ? `${fixed(a.c_match[k], 3)} / ${fixed(a.c_reverse[k], 3)}` : null],
        ["Library RI", isFinite(a.c_lri[k]) ? `${fixed(a.c_lri[k], 0)} (Δ ${fixed(a.c_dri[k], 0)})` : null]]));
      return out;
    }

    const traces = ctx.traces || (ctx.traces = new Map());       // compound|file -> answer, while the view is open
    async function loadTrace(k, box) {
      const key = `${a.c_id[k]}|${st.file || ""}`;
      let t = traces.get(key);
      if (!t) {
        box.replaceChildren(h("div.loading", { style: { padding: "28px" } }, P3.spinner(18), h("span.body-small.muted", "Reading the chromatograms…")));
        try { t = await ctx.trace(a.c_id[k], st.file); } catch (e) { t = { error: String(e.message || e), files: [] }; }
        if (!t.error) { traces.set(key, t); if (traces.size > 120) traces.delete(traces.keys().next().value); }
      }
      if (!ctx.alive() || shown !== k || !box.isConnected) return;
      // which file: the compound's reference file first, then any other file that holds it
      const files = t.files && t.files.length ? t.files : G.stems.filter((_, f) => isFinite(G.value(k, f)));
      const sel = h("select", { "aria-label": "File" }, files.map((f) => h("option", { value: f }, f)));
      sel.value = t.file || st.file || (G.h.ref_file || [])[k] || "";
      sel.addEventListener("change", () => { st.file = sel.value; ctx.drop(traceCharts); loadTrace(k, box); });
      const fileField = h("div.field.dense", { style: { minWidth: "220px", flex: "1" } }, h("div.box", sel, h("label", "File"), h("span.select-arrow", icon("expand_more"))));
      if (t.error) return box.replaceChildren(h("div.chrom-tools", fileField), banner(`Not available: ${t.error}`, "error"));
      const normBtn = (v, label, title) => {
        const b = h("button.btn.text.state", { type: "button", "aria-pressed": String(st.norm === v), title }, label);
        b.addEventListener("click", () => { if (st.norm !== v) { st.norm = v; ctx.drop(traceCharts); loadTrace(k, box); } });
        return b;
      };
      const chartEl = h("div");
      box.replaceChildren(h("div.chrom-tools", fileField, normBtn(false, "Intensity", "Each ion at its measured intensity"),
        normBtn(true, "Same height", "Each ion scaled to its own maximum inside the peak, to compare the shapes")), chartEl);
      const x = t.rt_s, inPeak = (v) => v >= t.rt_min_s && v <= t.rt_max_s;
      const series = t.ions.map((ion, i) => {
        let m = 0;
        for (let p = 0; p < x.length; p++) if (inPeak(x[p]) && ion.y[p] > m) m = ion.y[p];
        const y = st.norm ? ion.y.map((v) => (m > 0 ? 100 * v / m : 0)) : ion.y;
        return { id: i, label: `m/z ${G.mz(ion.mz)}${ion.quant ? " (quantifier)" : ""}${ion.shared ? " · shared" : ""}`, x, y, color: CAT[i % CAT.length], width: ion.quant ? 2.5 : 1.5, alpha: ion.quant ? 1 : 0.85, ion };
      });
      const q = t.ions.find((ion) => ion.quant) || t.ions[0];
      if (t.model && q) {             // the compound's elution profile at the quantifier's abundance
        const scale = st.norm ? 100 : q.abundance;
        series.unshift({ id: -1, label: "elution profile of the compound", x: t.model.rt_s, y: t.model.y.map((v) => v * scale), color: "--viz-neutral", width: 1, fill: true, alpha: 0.9 });
      }
      const c = ctx.track(P3.plot.line(chartEl, {
        height: 260, series, zoom: true, xLabel: `RT in ${t.file} (s)`, yLabel: st.norm ? "% of the ion's maximum" : "Intensity",
        xFmt: (v, d) => fixed(v, Math.min(d, 1)), yDomain: st.norm ? [0, 115] : null,
        vlines: [{ x: t.rt_min_s, color: "--viz-ink-2", dash: [5, 4], width: 1.25 }, { x: t.rt_max_s, color: "--viz-ink-2", dash: [5, 4], width: 1.25 }, { x: t.apex_s, color: "--viz-muted", dash: [2, 3], width: 1 }],
        label: (si) => c.spec.series[si].label,
        tip: (si, p) => { const r = c.spec.series[si]; return `<b>${esc(r.label)}</b><br>RT ${fixed(r.x[p], 1)} s · ${st.norm ? fixed(r.y[p], 0) + " %" : sci(r.y[p])}` + (r.ion ? `<br>${fixed(r.ion.rel, 1)} % of the base peak in the spectrum` : ""); },
      }));
      traceCharts.push(c);
      detailCharts.push(c);
      box.append(P3.plot.legend([...t.ions.map((ion, i) => ({ kind: "line", bold: ion.quant, color: CAT[i % CAT.length], label: `${G.mz(ion.mz)}${ion.quant ? " (quantifier)" : ""}` })),
        ...(t.model ? [{ color: "--viz-neutral", label: "elution profile" }] : []), { kind: "dash", color: "--viz-ink-2", label: "component start / end" }]),
        h("p.body-small.muted", { style: { margin: "6px 0 0" } }, `${int(t.ions.length)} of the ${int(t.n_ions)} ions of this file's component ${int(t.component)}; the grey area is the profile every ion was fitted with, drawn at the quantifier's abundance.`));
    }
    const traceCharts = [];

    refresh();
    if ((st.sel == null || st.sel >= G.n) && list.length) st.sel = list[0];
    if (st.sel != null && st.sel < G.n) {
      select(st.sel);
      const i = list.indexOf(st.sel);
      if (i > 0) requestAnimationFrame(() => { scroller.scrollTop = Math.max(0, 45 + i * rowH - 120); paintRows(); });
    } else renderDetail(null);
  }

  P3.gc = { Compounds, render, table, msp };
})(window.P3 = window.P3 || {});
