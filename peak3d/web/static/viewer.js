// Results viewer: Overview, Features (table + chromatograms), Alignment and Run details for one peak3d
// output. Mounted by report.html (standalone) and by the GUI. Reads the decoded bundle only.
(function (P3) {
  "use strict";
  const { h, icon, fixed, int, compact, sci, duration } = P3;
  const S1 = "--viz-series-1", S2 = "--viz-series-2";
  const STATUS = ["not found", "gap-filled", "gap-filled (peak confirmed)", "detected"];

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
      for (let j = 0; j < P; j++) x[j] = a.eic_x0[k] + j * a.eic_dx[k];
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

  // the listed features as a tab-separated table, from the values stored in the bundle (rounded to 0.04 %)
  const TABLE_COLS = ["group_id", "mz", "rt", "rt_min", "rt_max", "n_detected", "n_confirmed", "n_filled", "mz_ppm_spread",
    "rt_sd", "iso_offset", "charge"];
  function bundleTable(R, rows) {
    const a = R.a, num = (v, d) => (isFinite(v) ? v.toFixed(d) : "");
    const parts = [TABLE_COLS.concat(a.m_q ? R.stems : []).join("\t") + "\n"];
    let buf = [];
    for (const g of rows) {
      const row = [a.g_id[g], a.g_mz[g].toFixed(6), num(a.g_rt[g], 4), num(a.g_rt_min[g], 4), num(a.g_rt_max[g], 4),
        a.g_ndet[g], a.g_nconf[g], a.g_nfill[g], num(a.g_ppm[g], 3), num(a.g_rtsd[g], 5), a.g_iso[g], a.g_z[g]];
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
    const state = { tab: "overview", sel: null, sortKey: "max", sortDir: -1, query: "", mono: false, all: false, withEic: false, ppm: 10 };
    const track = (c) => { charts.push(c); return c; };
    const clearCharts = () => { while (charts.length) charts.pop().destroy(); };

    root.replaceChildren();
    root.classList.add("viewer");
    const tabsDef = [["overview", "insights", "Overview"], ["features", "table_rows", "Features"], ["alignment", "timeline", "Alignment"], ["details", "info", "Run details"]];
    const tabs = h("div.tabs", { role: "tablist" }, tabsDef.map(([id, ic, label]) =>
      h("button.tab.state", { type: "button", role: "tab", "data-tab": id, "aria-selected": "false", onclick: () => show(id) }, icon(ic), label)));
    const body = h("div.page");
    root.append(tabs, body);

    function show(id) {
      state.tab = id;
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
      const total = ["pick_s", "align_s", "group_s", "gapfill_s", "qc_s"].reduce((s, k) => s + (st[k] || 0), 0);
      body.append(h("div.tiles",
        tile("Files", int(R.N), R.params.input_files ? "centroided MS1 mzML" : null),
        tile("Aligned features", int(R.G), st.n_groups_all ? `${int(st.n_groups_all)} groups before the presence rule` : null),
        tile("Features per file", mf ? int(mf.med) : "–", mf && R.N > 1 ? `median; range ${int(mf.lo)}–${int(mf.hi)}` : "picked, before alignment"),
        tile("RT alignment", res ? `${fixed(res.mad_after, 2)} s` : R.N > 1 ? "off" : "n/a", res ? `anchor MAD, ${fixed(res.mad_before, 2)} s before correction` : "single file or correction disabled"),
        tile("Processing time", total ? duration(total) : "–", `peak3d ${R.params.peak3d_version || ""}`)));

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
        height: 460, x: R.a.g_rt, y: R.a.g_mz, v: R.logMax, vDomain: [lo, hi], xLabel: "Retention time (min)", yLabel: "m/z",
        xFmt: (v, d) => fixed(v, Math.min(d, 2)), yFmt: (v, d) => fixed(v, Math.min(d, 1)), selected: state.sel,
        tip: (g) => `<b>m/z ${fixed(R.a.g_mz[g], 4)}</b><br>RT ${fixed(R.a.g_rt[g], 3)} min · max ${sci(R.a.g_max[g])}<br>detected in ${R.a.g_ndet[g]} of ${R.N} files`,
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
      for (let g = 0; g < R.G; g++) {
        if (state.mono && a.g_iso[g] > 0) continue;
        if (state.all && a.g_ndet[g] < R.N) continue;
        if (state.withEic && !R.eicAt.has(g)) continue;
        if (idq != null && a.g_id[g] !== idq) continue;
        if (mzq != null && Math.abs(a.g_mz[g] - mzq) / mzq * 1e6 > state.ppm) continue;
        out.push(g);
      }
      const key = {
        id: a.g_id, mz: a.g_mz, rt: a.g_rt, ndet: a.g_ndet, nfill: a.g_nfill, max: a.g_max, cv: a.g_cv, iso: a.g_iso,
      }[state.sortKey];
      if (mzq != null) out.sort((p, r) => Math.abs(a.g_mz[p] - mzq) - Math.abs(a.g_mz[r] - mzq));
      else if (state.sortKey === "width") out.sort((p, r) => state.sortDir * ((a.g_rt_max[p] - a.g_rt_min[p]) - (a.g_rt_max[r] - a.g_rt_min[r])));
      else if (key) out.sort((p, r) => { const x = key[p], y = key[r]; return isNaN(x) ? 1 : isNaN(y) ? -1 : state.sortDir * (x - y); });
      return out;
    }

    function features() {
      const a = R.a;
      const search = h("input", { type: "search", placeholder: " ", value: state.query, "aria-label": "Search m/z or #group id" });
      const searchField = h("div.field.dense", h("div.box", h("span.lead", icon("search", "sm")), search, h("label", "m/z or #id")));
      const ppmIn = h("input", { type: "number", placeholder: " ", min: "0.1", step: "1", value: String(state.ppm) });
      const ppmField = h("div.field.dense", { style: { width: "96px" } }, h("div.box", ppmIn, h("label", "± ppm")));
      const chip = (key, label) => {
        const c = h("button.chip.state", { type: "button", "aria-pressed": String(state[key]) }, state[key] ? icon("check") : null, label);
        c.addEventListener("click", () => { state[key] = !state[key]; features_refresh(); c.setAttribute("aria-pressed", String(state[key])); c.replaceChildren(...(state[key] ? [icon("check")] : []), label); });
        return c;
      };
      const count = h("span.body-small.muted");
      const dl = h("button.btn.outlined.state", { type: "button", title: opts.exportTable ?
        "The features listed here (search, filters and sort applied), with the exact values from aligned_feature_table.tsv" :
        "The features listed here (search, filters and sort applied). Values are rounded to 0.04 %; the exact table is aligned_feature_table.tsv in the output folder." },
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
      body.append(h("div.toolbar", searchField, ppmField, chip("mono", "Monoisotopic"), chip("all", "In every file"),
        R.h.eic ? chip("withEic", "With chromatograms") : null, h("span.spacer"), count, dl));

      const COLS = [["id", "ID", true], ["mz", "m/z", true], ["rt", "RT (min)", true], ["width", "Width (s)", true], ["ndet", "Detected", true],
        ["nfill", "Gap-filled", true], ["max", "Max " + R.quant.toLowerCase(), true], ["cv", "CV", true], ["iso", "Isotope", false]];
      const thead = h("thead", h("tr", COLS.map(([k, label]) => {
        const th = h("th.sortable.r", { "data-k": k }, label);
        if (k === "iso") th.classList.remove("r");
        th.addEventListener("click", () => {
          if (state.sortKey === k) state.sortDir *= -1; else { state.sortKey = k; state.sortDir = k === "max" || k === "ndet" ? -1 : 1; }
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
          h("td.r", int(a.g_id[g])), h("td.r", fixed(a.g_mz[g], 4)), h("td.r", fixed(a.g_rt[g], 3)),
          h("td.r", fixed((a.g_rt_max[g] - a.g_rt_min[g]) * 60, 1)), h("td.r", `${a.g_ndet[g]}/${R.N}`),
          h("td.r", int(a.g_nfill[g])), h("td.r", sci(a.g_max[g])), h("td.r", isFinite(a.g_cv[g]) ? fixed(a.g_cv[g], 0) + "%" : "–"),
          h("td", iso + z));
        tr.addEventListener("click", () => select(g));
        return tr;
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
        foot.textContent = R.h.eic && R.h.eic.n < R.G ? `Chromatograms are stored for the ${int(R.h.eic.n)} most intense features (${int(R.h.eic.files.length)} of ${R.N} files each); the rest have values only.` : "";
        paintHead();
        paintRows();
      }
      let tmr = null;
      search.addEventListener("input", () => { clearTimeout(tmr); tmr = setTimeout(() => { state.query = search.value; scroller.scrollTop = 0; features_refresh(); }, 120); });
      ppmIn.addEventListener("input", () => { const v = +ppmIn.value; if (v > 0) { state.ppm = v; features_refresh(); } });

      function select(g) {
        state.sel = g;
        tbody.querySelectorAll("tr[data-g]").forEach((tr) => tr.setAttribute("aria-selected", String(+tr.dataset.g === g)));
        renderDetail(detail, g);
      }
      features_refresh();
      if (state.sel == null && list.length) state.sel = list[0];
      if (state.sel != null) {
        select(state.sel);
        const i = list.indexOf(state.sel);
        if (i > 0) requestAnimationFrame(() => { scroller.scrollTop = Math.max(0, 45 + i * rowH - 120); paintRows(); });
      } else renderDetail(detail, null);
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
        h("p.body-medium.muted", `RT ${fixed(a.g_rt[g], 3)} min · window ${fixed(a.g_rt_min[g], 2)}–${fixed(a.g_rt_max[g], 2)} min`),
        h("div.row.wrap", { style: { marginTop: "12px" } },
          h("span.chip.static" + (all ? ".ok" : ".info"), icon(all ? "check_circle" : "info"), `Detected in ${a.g_ndet[g]} of ${R.N} files`),
          h("span.chip.static", iso), a.g_z[g] > 0 ? h("span.chip.static", `z = ${a.g_z[g]}`) : null,
          a.g_nfill[g] ? h("span.chip.static", `${a.g_nfill[g]} gap-filled`) : null));
      const parts = [head];

      const e = R.eic(g);
      const statusOf = (f) => R.status(g, f);
      if (e) {
        const chartEl = h("div");
        parts.push(h("div.section-title", h("h3.title-small", "Chromatograms"),
          h("p.body-small.muted", `±${R.h.eic.ppm} ppm around the feature, on the aligned RT axis${R.h.eic.files.length < R.N ? `; ${R.h.eic.files.length} of ${R.N} files (evenly spaced)` : ""}. Hover a trace to name its file.`)), chartEl);
        const hasStatus = !!R.a.m_status;
        const series = e.traces.map((t) => {
          const st = statusOf(t.file);
          return { id: t.file, label: R.stems[t.file], x: e.x, y: t.y, color: hasStatus && st != null && st < 3 ? S2 : S1, width: 1.5, alpha: 0.9 };
        });
        const c = track(P3.plot.line(chartEl, {
          height: 240, series, xLabel: "Aligned RT (min)", yLabel: R.quant === "Area" ? "Intensity" : "Intensity",
          xFmt: (v, d) => fixed(v, Math.min(Math.max(d, 1), 3)), vline: a.g_rt[g],
          label: (si) => series[si].label,
          tip: (si, k) => { const st = statusOf(series[si].id); return `<b>${esc(series[si].label)}</b><br>RT ${fixed(e.x[k], 3)} min · ${sci(series[si].y[k])}${st != null ? "<br>" + STATUS[st] : ""}`; },
        }));
        detailCharts.push(c);
        if (hasStatus) parts.push(P3.plot.legend([{ kind: "line", color: S1, label: "detected by the picker" }, { kind: "line", color: S2, label: "gap-filled from the chromatogram" }]));
      } else {
        parts.push(h("div.section-title", h("h3.title-small", "Chromatograms")),
          banner(R.h.eic ? "Not stored for this feature: the report keeps chromatograms for the most intense features only (peak3d report --eic-budget raises the limit)." :
            `Not available: ${R.h.eic_note || "no point-cloud cache"}. Re-run with --keep-cache, or build the report while the cache exists.`));
      }

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

    // ---------------------------------------------------------------- alignment
    function alignment() {
      const a = R.a, res = R.h.residuals;
      const medoid = R.files.find((f) => f.role === "medoid");
      const flagged = R.files.filter((f) => f.flag && !String(f.flag).startsWith("ok") && f.role !== "medoid");
      if (R.N < 2 || !a.drift_off || a.drift_x.length === 0) {
        body.append(h("div", { style: { marginTop: "16px" } }, banner(R.N < 2 ? "A single file: nothing to align." : "RT correction was disabled for this run (--no-rt-correction).")));
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
        return { id: f, label: s, x: a.drift_x.subarray(o0, o1), y: a.drift_y.subarray(o0, o1), width: R.files[f] && R.files[f].role === "medoid" ? 2.5 : 1.5, color: null, t: R.N > 1 ? f / (R.N - 1) : 0 };
      });
      const recolor = () => { const from = P3.css("--viz-ramp-from"), to = P3.css("--viz-ramp-to"); series.forEach((r) => (r.color = P3.plot.mix(from, to, r.t))); };
      recolor();
      const dc = track(P3.plot.line(driftEl, {
        height: 300, series, yZero: false, xLabel: "Native RT (min)", yLabel: "Shift (s)", highlight: medoid ? R.stems.indexOf(medoid.stem) : null,
        xFmt: (v, d) => fixed(v, Math.min(d, 2)), yFmt: (v, d) => fixed(v, Math.min(d, 1)),
        label: (si) => (R.files[si] && R.files[si].role === "medoid" ? "medoid: " : "") + series[si].label,
        tip: (si, k) => `<b>${esc(series[si].label)}</b><br>RT ${fixed(series[si].x[k], 2)} min · shift ${fixed(series[si].y[k], 2)} s`,
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
          ["peak3d", p.peak3d_version], ["Command", p.command ? h("code.mono", p.command) : null],
          ["Files", int(R.N)], ["Alignment", p.alignment], ["Reference file", p.medoid],
          ["m/z tolerance", p.ppm_tol != null ? fixed(p.ppm_tol, 2) + " ppm" : null],
          ["RT tolerance", p.rt_tol_min != null ? fixed(p.rt_tol_min * 60, 2) + " s" : null],
          ["Software", ["numpy " + (p.numpy || "?"), "scipy " + (p.scipy || "?"), "numba " + (p.numba || "?"), "pandas " + (p.pandas || "?")].join(" · ")],
          ["Report built", R.h.created],
        ])),
        card(".elevated", "Options", null, kv(Object.entries(o).filter(([k]) => k !== "cmd").map(([k, v]) => [k.replace(/_/g, " "), v == null ? "–" : Array.isArray(v) ? v.join(", ") || "–" : String(v)]))),
        card(".elevated", "Time per stage", null, kv([["Picking", duration(st.pick_s)], ["Alignment", duration(st.align_s)], ["Grouping", duration(st.group_s)], ["Gap filling", duration(st.gapfill_s)], ["QC figures", duration(st.qc_s)]])),
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
