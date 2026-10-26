// peak3d GUI: the Run page (folder pickers, options generated from the CLI parser, live progress) and the
// Results page (the same viewer as report.html, fed by the local server).
(function (P3) {
  "use strict";
  const { h, icon, int, duration, snack } = P3;
  const TOKEN = (document.querySelector('meta[name="peak3d-token"]') || {}).content || "";

  // ---------------------------------------------------------------- server calls
  async function api(path, opts) {
    const r = await fetch(path, Object.assign({ credentials: "same-origin", headers: { "X-Peak3d-Token": TOKEN, "Content-Type": "application/json" } }, opts));
    const ct = r.headers.get("Content-Type") || "";
    if (ct.includes("json")) {
      const j = await r.json();
      if (!r.ok) { const e = new Error(j.error || r.statusText); e.status = r.status; e.body = j; throw e; }
      return j;
    }
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return new Uint8Array(await r.arrayBuffer());
  }
  const post = (path, body) => api(path, { method: "POST", body: JSON.stringify(body || {}) });
  const enc = encodeURIComponent;

  // ---------------------------------------------------------------- state
  const S = { info: null, schema: [], form: {}, input: "", output: "", inputInfo: null, runs: new Map(), page: "run", resultsPath: null, viewer: null, viewerTab: "overview" };
  const pageEl = document.getElementById("page");
  const scroll = document.getElementById("scroll");
  const bar = document.getElementById("bar");
  scroll.addEventListener("scroll", () => bar.classList.toggle("scrolled", scroll.scrollTop > 0));

  const LABELS = {
    cores: "CPU cores", min_score: "Minimum feature score", polarity: "Polarity", model: "Re-scoring model (optional)",
    keep_rejected: "Also write rejected candidates", set: "Picker overrides", ppm: "m/z tolerance (ppm)",
    rt_tol_max: "Maximum RT tolerance (min)", no_rt_correction: "RT correction", no_gap_fill: "Gap filling",
    min_presence: "Minimum files per feature", holdout: "Held-out anchor check", quant: "Main table value",
    keep_cache: "Keep the point-cloud cache", no_report: "Interactive report (report.html)",
  };
  const HELP = {
    cores: "Files are picked in parallel; each worker holds one file in memory (about 0.5–1.5 GB).",
    no_rt_correction: "Align the runs on their own most credible features before grouping.",
    no_gap_fill: "Fill missing values from each file's chromatogram, and confirm real peaks there.",
    keep_cache: "Keeps .cache/ (large) so `peak3d align` can re-run alignment without re-picking.",
    set: "One NAME=VALUE per line; advanced picker parameters.",
    no_report: "A single HTML file with these views and chromatograms; opens in any browser, easy to share.",
  };
  const INVERTED = new Set(["no_rt_correction", "no_gap_fill", "no_report"]);
  const BASIC = ["cores", "polarity", "quant", "min_score", "ppm", "no_rt_correction", "no_gap_fill"];
  const SEGMENTS = { polarity: [[null, "Auto"], ["pos", "Positive"], ["neg", "Negative"]], quant: [["height", "Height"], ["area", "Area"]] };

  function stamp() {
    const d = new Date(), p = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`;
  }
  const join = (a, b) => (a.endsWith(S.info.sep) ? a + b : a + S.info.sep + b);
  const basename = (p) => p.split(/[\\/]/).filter(Boolean).pop() || p;
  const dirname = (p) => { const parts = p.split(/[\\/]/); parts.pop(); return parts.join(S.info.sep) || S.info.sep; };

  // ---------------------------------------------------------------- navigation
  const rail = document.querySelectorAll(".rail-item");
  rail.forEach((b) => {
    b.querySelector(".indicator").appendChild(icon(b.dataset.page === "run" ? "play_arrow" : "insights"));
    b.addEventListener("click", () => go(b.dataset.page));
  });
  document.getElementById("brand").appendChild(icon("science"));
  document.getElementById("rail-bottom").appendChild(P3.themeButton());

  function go(page) {
    S.page = page;
    rail.forEach((b) => { if (b.dataset.page === page) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current"); });
    if (S.viewer) { S.viewer.destroy(); S.viewer = null; }
    document.getElementById("bar-actions").replaceChildren();
    pageEl.replaceChildren();
    scroll.scrollTop = 0;
    (page === "run" ? runPage : resultsPage)();
  }
  function setTitle(t, sub) {
    document.getElementById("page-title").textContent = t;
    document.getElementById("page-sub").textContent = sub || "";
  }

  // ---------------------------------------------------------------- form fields
  function textField(label, value, opts) {
    opts = opts || {};
    const input = h("input", { type: opts.type || "text", placeholder: " ", spellcheck: "false", autocomplete: "off" });
    input.value = value == null ? "" : String(value);
    if (opts.min != null) input.min = opts.min;
    if (opts.max != null) input.max = opts.max;
    if (opts.step != null) input.step = opts.step;
    const support = h("div.support", opts.help || "");
    const box = h("div.box", opts.lead ? h("span.lead", icon(opts.lead)) : null, input, h("label", label), opts.trail ? h("span.trail", opts.trail) : null);
    const el = h("div.field", box, support);
    if (opts.oninput) input.addEventListener("input", () => opts.oninput(input.value));
    if (opts.onchange) input.addEventListener("change", () => opts.onchange(input.value));
    return { el, input, support, setError(msg) { el.classList.toggle("error", !!msg); support.textContent = msg || opts.help || ""; }, setHelp(t) { opts.help = t; if (!el.classList.contains("error")) support.textContent = t; } };
  }
  function switchRow(label, help, checked, onchange) {
    const input = h("input", { type: "checkbox", role: "switch", "aria-label": label });
    input.checked = !!checked;
    input.addEventListener("change", () => onchange(input.checked));
    return h("label.switch-row", h("div.text", h("div.body-large", label), help ? h("div.supporting", help) : null),
      h("span.switch", input, h("span.track"), h("span.handle")));
  }
  function segmented(options, value, onchange) {
    const el = h("div.segmented", { role: "group" });
    const paint = (v) => el.querySelectorAll("button").forEach((b, i) => {
      const on = options[i][0] === v;
      b.setAttribute("aria-pressed", String(on));
      b.replaceChildren(...(on ? [icon("check")] : []), options[i][1]);
    });
    options.forEach(([v, label]) => el.appendChild(h("button.state", { type: "button", onclick: () => { paint(v); onchange(v); } }, label)));
    paint(value);
    return el;
  }

  function fieldFor(f, onAny) {
    const label = LABELS[f.dest] || f.flag.replace(/^--/, "").replace(/-/g, " ");
    const raw = HELP[f.dest] || f.help || "";
    const help = raw.startsWith("m/z") ? raw : raw.charAt(0).toUpperCase() + raw.slice(1);
    const set = (v) => { S.form[f.dest] = v; onAny(); };
    if (f.kind === "flag") {
      const inv = INVERTED.has(f.dest);
      return switchRow(label, help, inv ? !S.form[f.dest] : S.form[f.dest], (on) => set(inv ? !on : on));
    }
    if (SEGMENTS[f.dest]) {
      return h("div", { style: { padding: "4px 0 12px" } }, h("div.label-large", { style: { marginBottom: "8px" } }, label),
        segmented(SEGMENTS[f.dest], S.form[f.dest], set), help && f.dest !== "quant" ? h("div.body-small.muted", { style: { marginTop: "6px" } }, help) : null);
    }
    if (f.kind === "choice") {
      const sel = h("select", f.choices.map((c) => h("option", { value: c }, c)));
      sel.value = S.form[f.dest];
      sel.addEventListener("change", () => set(sel.value));
      return h("div.field", h("div.box", sel, h("label", label), h("span.select-arrow", icon("expand_more"))), h("div.support", help));
    }
    if (f.kind === "list") {
      const ta = h("textarea", { placeholder: " ", rows: "3", spellcheck: "false" });
      ta.value = (S.form[f.dest] || []).join("\n");
      ta.addEventListener("input", () => set(ta.value.split("\n").map((s) => s.trim()).filter(Boolean)));
      return h("div.field.textarea", h("div.box", ta, h("label", label)), h("div.support", help));
    }
    const num = f.kind === "int" || f.kind === "float";
    const t = textField(label, S.form[f.dest], {
      type: num ? "number" : "text", help, step: f.kind === "int" ? "1" : "any", min: f.min_exclusive != null ? (f.kind === "int" ? 1 : 0) : null, max: f.max,
      oninput: (v) => {
        let err = null;
        if (num && v !== "") {
          const x = +v;
          if (!isFinite(x)) err = "Enter a number";
          else if (f.kind === "int" && !Number.isInteger(x)) err = "Enter a whole number";
          else if (f.min_exclusive != null && x <= f.min_exclusive) err = "Must be greater than 0";
          else if (f.max != null && x > f.max) err = `Must be at most ${f.max}`;
        }
        t.setError(err);
        S.form[f.dest] = v === "" ? f.default : num ? +v : v;
        S.formErrors[f.dest] = !!err;
        onAny();
      },
    });
    return t.el;
  }

  function argvPreview() {
    const parts = ["peak3d", "process", "--input", q(S.input || "<folder>"), "--output", q(S.output || "<folder>")];
    for (const f of S.schema) {
      const v = S.form[f.dest];
      if (f.kind === "flag") { if (v) parts.push(f.flag); }
      else if (f.kind === "list") (v || []).forEach((x) => parts.push(f.flag, q(x)));
      else if (v != null && String(v) !== "" && String(v) !== String(f.default)) parts.push(f.flag, q(v));
    }
    return parts.join(" ");
  }
  const q = (s) => (/^[\w./:=+@%,\\-]+$/.test(String(s)) ? String(s) : `"${String(s).replace(/"/g, '\\"')}"`);

  // ---------------------------------------------------------------- run page
  function runPage() {
    setTitle("New run");
    const page = h("div.page.narrow");
    pageEl.appendChild(page);
    page.appendChild(h("div", { style: { margin: "8px 0 24px" } },
      h("p.body-large.muted", "Choose a folder of centroided MS1 .mzML files. peak3d picks every file, aligns the runs on their own features, and writes one feature table plus an interactive report.")));

    const runBtn = h("button.btn.filled.lg.state", { type: "button" }, icon("play_arrow"), "Run peak3d");
    const cmd = h("code");
    const refresh = () => {
      cmd.textContent = argvPreview();
      const bad = !S.inputInfo || !S.inputInfo.n_mzml || !S.output || Object.values(S.formErrors).some(Boolean);
      runBtn.disabled = bad;
    };

    const inField = textField("mzML folder", S.input, {
      lead: "folder", help: "The folder that holds your .mzML files",
      trail: h("button.icon-btn.state", { type: "button", title: "Browse", "aria-label": "Browse for the input folder", onclick: () => browse("input", S.input, (p) => { inField.input.value = p; setInput(p); }) }, icon("folder_open")),
      onchange: (v) => setInput(v.trim()),
    });
    const outField = textField("Output folder", S.output, {
      lead: "drive_folder_upload", help: "Created if missing; tables, QC figures and report.html go here",
      trail: h("button.icon-btn.state", { type: "button", title: "Browse", "aria-label": "Browse for the output location", onclick: () => browse("output", S.output ? dirname(S.output) : S.input, (p) => { S.output = join(p, basename(S.output || "peak3d_" + stamp())); outField.input.value = S.output; refresh(); }) }, icon("folder_open")),
      oninput: (v) => { S.output = v.trim(); refresh(); },
    });

    let seq = 0;
    async function setInput(p) {
      S.input = p;
      P3.storeSet("peak3d-last-input", p);
      const my = ++seq;
      if (!p) { S.inputInfo = null; inField.setError(null); refresh(); return; }
      try {
        const info = await api(`/api/ls?path=${enc(p)}`);
        if (my !== seq) return;
        S.inputInfo = info;
        if (!info.n_mzml) inField.setError("No .mzML files in this folder");
        else {
          inField.setError(null);
          inField.setHelp(`${int(info.n_mzml)} mzML file${info.n_mzml > 1 ? "s" : ""}: ${info.mzml.slice(0, 3).join(", ")}${info.n_mzml > 3 ? ", …" : ""}`);
          if (!S.output || S.autoOutput === S.output) { S.output = S.autoOutput = join(info.path, "peak3d_" + stamp()); outField.input.value = S.output; }
        }
      } catch (e) {
        if (my !== seq) return;
        S.inputInfo = null;
        inField.setError(e.message);
      }
      refresh();
    }

    const exBtn = h("button.btn.tonal.state", { type: "button" }, icon("science"), "Create example data");
    exBtn.addEventListener("click", async () => {
      exBtn.disabled = true;
      exBtn.replaceChildren(P3.spinner(18), "Writing example data…");
      try {
        const r = await post("/api/example");
        inField.input.value = r.input;
        await setInput(r.input);
        S.output = S.autoOutput = r.output;
        outField.input.value = r.output;
        refresh();
        snack(`Example data ready: ${r.n_mzml} synthetic runs. Press Run peak3d.`);
      } catch (e) {
        snack(e.message);
      } finally {
        exBtn.disabled = false;
        exBtn.replaceChildren(icon("science"), "Create example data");
      }
    });
    page.appendChild(h("section.card.elevated", h("div.card-head", h("h3.title-medium", "Data")), h("div.stack", inField.el, outField.el),
      h("div.row.wrap", { style: { marginTop: "4px", gap: "12px" } }, exBtn,
        h("span.body-small.muted", { style: { flex: "1", minWidth: "220px" } }, "No mzML files at hand? This writes three small synthetic LC-MS runs into a peak3d_example folder so you can try everything."))));

    const basic = h("div.form"), adv = h("div.form");
    for (const f of S.schema) (BASIC.includes(f.dest) ? basic : adv).appendChild(fieldFor(f, refresh));
    page.appendChild(h("section.card.elevated", { style: { marginTop: "16px" } }, h("div.card-head", h("h3.title-medium", "Settings")), basic,
      h("details.expand", { style: { marginTop: "8px" } }, h("summary.state", icon("tune"), "Advanced options", h("span.spacer"), icon("expand_more", "chev")), h("div", { style: { padding: "8px 0 0" } }, adv))));

    const copyBtn = h("button.icon-btn.state", { type: "button", title: "Copy command", "aria-label": "Copy the equivalent command", onclick: () => {
      (navigator.clipboard ? navigator.clipboard.writeText(cmd.textContent) : Promise.reject()).then(() => snack("Command copied"), () => snack("Copy failed; select the text instead"));
    } }, icon("content_copy"));
    page.appendChild(h("div", { style: { margin: "20px 0 8px" } }, h("div.label-medium.muted", { style: { margin: "0 0 6px 4px" } }, "Equivalent command"), h("div.cmd", cmd, copyBtn)));
    page.appendChild(h("div.row", { style: { margin: "16px 0 8px" } }, runBtn));
    runBtn.addEventListener("click", () => start(false));

    const runsBox = h("div.stack#runs");
    page.appendChild(h("div", h("h2.title-large.section-title", { style: { marginTop: "32px" } }, "Runs"), runsBox));
    renderRuns();

    async function start(overwrite) {
      runBtn.disabled = true;
      try {
        const r = await post("/api/runs", { input: S.input, output: S.output, options: S.form, overwrite });
        S.runs.set(r.id, { status: r, log: r.log || [] });
        renderRuns();
        poll(r.id);
        snack("Run started");
        S.autoOutput = null;
        S.output = join(dirname(S.output), "peak3d_" + stamp());       // the next run gets its own folder
        if (S.output === r.output) S.output += "_2";
        S.autoOutput = S.output;
        outField.input.value = S.output;
      } catch (e) {
        if (e.status === 409 && e.body && e.body.confirm === "overwrite") {
          if (await confirmDialog("Replace existing results?", `${S.output} already holds peak3d results. Running again replaces them.`, "Replace")) return start(true);
        } else snack(e.message);
      } finally { refresh(); }
    }

    if (!S.input) {
      const last = P3.storeGet("peak3d-last-input");
      if (last) { inField.input.value = last; setInput(last); }
    } else setInput(S.input);
    refresh();
  }

  // ---------------------------------------------------------------- runs
  const STEPS = [["compile", "Compile"], ["pick", "Pick files"], ["align", "Align"], ["group", "Group"], ["fill", "Gap-fill"], ["report", "Report"]];
  const STAGE_TEXT = {
    compile: "Compiling the picker (the first run on a machine takes about a minute)…", pick: "Picking features in each file…",
    align: "Aligning retention times…", group: "Grouping features across files…", fill: "Filling gaps from the chromatograms…",
    write: "Writing tables and QC figures…", report: "Writing the report…", done: "Finished", failed: "Failed", cancelled: "Cancelled",
  };
  function renderRuns() {
    const box = document.getElementById("runs");
    if (!box) return;
    const list = [...S.runs.values()].sort((a, b) => b.status.started - a.status.started);
    if (!list.length) { box.replaceChildren(h("p.body-medium.muted", "No runs yet in this session.")); return; }
    box.replaceChildren(...list.map(runCard));
  }
  function runCard(R) {
    const s = R.status;
    const running = s.state === "running";
    const stIdx = Math.max(0, STEPS.findIndex(([k]) => k === (s.stage === "write" ? "fill" : s.stage)));
    const head = h("div.row",
      running ? P3.spinner(24) : icon(s.state === "done" ? "check_circle" : s.state === "failed" ? "error" : "cancel"),
      h("div", { style: { flex: 1, minWidth: 0 } }, h("div.title-medium", `${basename(s.input)} · ${int(s.n_files)} files`),
        h("div.body-small.muted.mono", { style: { overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }, title: s.output }, "→ " + s.output)),
      h("span.chip.static" + (s.state === "done" ? ".ok" : s.state === "failed" ? ".warn" : ".info"), s.state === "done" ? "Done" : s.state === "failed" ? "Failed" : s.state === "cancelled" ? "Cancelled" : "Running"),
      h("span.body-medium.muted.num", { style: { minWidth: "72px", textAlign: "right" } }, duration(s.elapsed)));
    const prog = h("div.progress" + (running && s.stage === "compile" ? ".indeterminate" : ""), h("div.bar", { style: { width: `${Math.round(100 * s.progress)}%` } }));
    const steps = h("div.steps", STEPS.map(([k, label], i) => {
      const done = s.state === "done" || i < stIdx, active = running && i === stIdx;
      return h("span.step" + (done ? ".done" : active ? ".active" : ""), done ? icon("check") : h("span.dot"),
        k === "pick" && s.picked != null && running && i === stIdx ? `${label} ${s.picked}/${s.n_files}` : label);
    }));
    const text = h("p.body-medium.muted", STAGE_TEXT[s.stage] || s.stage);
    const logPre = h("pre.log", R.log.slice(-400).join("\n"));
    const logBox = h("details.expand", { open: R.logOpen || null, ontoggle: (e) => (R.logOpen = e.target.open) }, h("summary.state", icon("description"), "Log", h("span.spacer"), icon("expand_more", "chev")), logPre);
    const actions = h("div.row.wrap", { style: { marginTop: "12px" } });
    if (running) actions.appendChild(h("button.btn.text.state", { type: "button", onclick: async () => { await post(`/api/runs/${s.id}/cancel`); snack("Cancelling…"); } }, icon("stop"), "Cancel"));
    if (s.state === "done") {
      actions.append(h("button.btn.filled.state", { type: "button", onclick: () => openResults(s.output) }, icon("insights"), "Open results"),
        h("a.btn.outlined.state", { href: `/api/results/report?path=${enc(s.output)}`, target: "_blank", rel: "noopener" }, icon("open_in_new"), "Report"));
    }
    const failed = s.state === "failed" ? h("div.banner.error", { style: { marginTop: "12px" } }, icon("error"), h("pre.mono", { style: { margin: 0, whiteSpace: "pre-wrap" } }, R.log.slice(-6).join("\n"))) : null;
    const el = h("section.card.elevated", head, h("div", { style: { marginTop: "16px" } }, prog), steps, text, failed, actions, logBox);
    requestAnimationFrame(() => { if (R.logOpen) logPre.scrollTop = logPre.scrollHeight; });
    return el;
  }
  function poll(id) {
    const R = S.runs.get(id);
    if (!R || R.polling) return;
    R.polling = true;
    const tick = async () => {
      try {
        const s = await api(`/api/runs/${id}?since=${R.log.length}`);
        R.log.push(...s.log);
        R.status = s;
        if (S.page === "run") renderRuns();
        if (s.state !== "running") {
          R.polling = false;
          if (s.state === "done") snack(`Run finished in ${duration(s.elapsed)}`, { label: "Open results", run: () => openResults(s.output) });
          else if (s.state === "failed") snack("Run failed; see the log");
          return;
        }
      } catch (e) { /* server restarting: keep trying */ }
      setTimeout(tick, 1000);
    };
    tick();
  }

  // ---------------------------------------------------------------- menu (Material 3)
  function menuButton(iconName, label, items) {
    const btn = h("button.icon-btn.state", { type: "button", title: label, "aria-label": label, "aria-haspopup": "menu" }, icon(iconName));
    let menu = null;
    const close = () => {
      if (!menu) return;
      menu.remove();
      menu = null;
      document.removeEventListener("pointerdown", outside, true);
      document.removeEventListener("keydown", esc, true);
    };
    const outside = (e) => { if (menu && !menu.contains(e.target) && !btn.contains(e.target)) close(); };
    const esc = (e) => { if (e.key === "Escape") close(); };
    btn.addEventListener("click", () => {
      if (menu) return close();
      menu = h("div.menu", { role: "menu" }, items.map((it) => (it === "-" ? h("hr.divider", { style: { margin: "8px 0" } }) :
        h(it.href ? "a.menu-item.state" : "button.menu-item.state", { role: "menuitem", href: it.href, download: it.href ? "" : null, type: it.href ? null : "button",
          onclick: () => { close(); if (it.run) it.run(); } }, icon(it.icon), h("span", it.label)))));
      document.body.appendChild(menu);
      const r = btn.getBoundingClientRect();
      menu.style.top = r.bottom + 4 + "px";
      menu.style.left = Math.max(8, Math.min(window.innerWidth - menu.offsetWidth - 8, r.right - menu.offsetWidth)) + "px";
      document.addEventListener("pointerdown", outside, true);
      document.addEventListener("keydown", esc, true);
      const first = menu.querySelector(".menu-item");
      if (first) first.focus();
    });
    return btn;
  }

  // ---------------------------------------------------------------- results page
  function openResults(path) { S.resultsPath = path; S.viewerTab = "overview"; go("results"); }
  async function resultsPage() {
    setTitle("Results");
    if (!S.resultsPath) {
      const done = [...S.runs.values()].filter((r) => r.status.state === "done");
      pageEl.appendChild(h("div.page.narrow", h("div.empty",
        icon("insights", "lg"), h("h2.headline-small", "Open results"),
        h("p.body-medium", "Choose a peak3d output folder (one with aligned_feature_table.tsv), or finish a run first."),
        h("button.btn.filled.state", { type: "button", onclick: () => browse("results", S.input || "", (p) => openResults(p)) }, icon("folder_open"), "Choose folder")),
        done.length ? h("section.card.elevated", h("div.card-head", h("h3.title-medium", "Finished in this session")),
          h("div.list", done.map((r) => h("button.list-item.state", { type: "button", onclick: () => openResults(r.status.output) }, icon("folder"),
            h("div.text", h("div.headline", basename(r.status.output)), h("div.supporting", r.status.output)), icon("chevron_right"))))) : null));
      return;
    }
    const path = S.resultsPath;
    setTitle("Results", path);
    const actions = document.getElementById("bar-actions");
    const reveal = { Darwin: "Show in Finder", Windows: "Show in File Explorer" }[S.info.platform] || "Open the output folder";
    const fileUrl = (name) => `/api/results/file?path=${enc(path)}&name=${name}`;
    const addActions = (quant) => {
      const other = quant === "area" ? "height" : "area";
      actions.replaceChildren(
        h("a.icon-btn.state", { href: `/api/results/report?path=${enc(path)}`, target: "_blank", rel: "noopener", title: "Open report.html in a new tab", "aria-label": "Open report" }, icon("open_in_new")),
        menuButton("download", "Download", [
          { icon: "table_rows", label: `Feature table (${quant}s)`, href: fileUrl("aligned_feature_table.tsv") },
          { icon: "table_rows", label: `Feature table (${other}s)`, href: fileUrl(`aligned_feature_${other}.tsv`) },
          { icon: "description", label: "Report (one HTML file to share)", href: `/api/results/report?path=${enc(path)}&download=1` },
          "-",
          { icon: "folder", label: reveal, run: () => post("/api/results/reveal", { path }).catch((e) => snack(e.message)) },
        ]),
        h("button.icon-btn.state", { type: "button", title: "Open other results", "aria-label": "Open other results", onclick: () => browse("results", dirname(path), (p) => openResults(p)) }, icon("folder_open")));
    };
    addActions("height");
    const holder = h("div");
    holder.appendChild(h("div.loading", P3.spinner(), h("span.body-medium", "Opening results…")));
    pageEl.appendChild(holder);
    try {
      const raw = await api(`/api/results/bundle?path=${enc(path)}`);
      const b = await P3.loadBundle(raw);
      if (S.page !== "results" || S.resultsPath !== path) return;
      addActions(b.header.quant === "area" ? "area" : "height");
      const exportTable = async (ids) => new Blob([await api("/api/results/table", { method: "POST", body: JSON.stringify({ path, ids }) })], { type: "text/tab-separated-values" });
      S.viewer = P3.viewer.mount(holder, b, { tab: S.viewerTab, onTab: (t) => (S.viewerTab = t), exportTable });
    } catch (e) {
      holder.replaceChildren(h("div.page", h("div.banner.error", { style: { marginTop: "16px" } }, icon("error"), h("div", e.message))));
    }
  }

  // ---------------------------------------------------------------- dialogs
  function browse(mode, start, done) {
    const dlg = document.getElementById("browse");
    const titles = { input: "Choose the mzML folder", output: "Choose where to write results", results: "Choose a results folder" };
    let cur = null;
    const pathField = textField("Folder", start || "", { lead: "folder_open", onchange: (v) => load(v.trim()) });
    pathField.input.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); load(pathField.input.value.trim()); } });
    const list = h("div.list");
    const status = h("div.body-small.muted", { style: { flex: 1 } });
    const ok = h("button.btn.filled.state", { type: "button" }, "Select");
    const cancel = h("button.btn.text.state", { type: "button", onclick: () => dlg.close() }, "Cancel");
    dlg.replaceChildren(h("div.dialog-head", h("h2.headline-small", titles[mode]), pathField.el), h("div.dialog-body", list), h("div.divider"), h("div.dialog-actions", status, cancel, ok));
    ok.addEventListener("click", () => { if (cur) { dlg.close(); done(cur.path); } });
    async function load(p) {
      list.replaceChildren(h("div.loading", { style: { padding: "32px" } }, P3.spinner()));
      try {
        const r = await api(`/api/ls?path=${enc(p)}`);
        cur = r;
        pathField.input.value = r.path;
        pathField.setError(null);
        const items = [];
        if (r.parent != null) items.push(h("button.list-item.state", { type: "button", onclick: () => load(r.parent) }, icon("arrow_back"), h("div.text", h("div.headline", "Up one level"))));
        for (const d of r.dirs) items.push(h("button.list-item.state", { type: "button", onclick: () => load(r.path ? join(r.path, d) : d) }, icon("folder"), h("div.text", h("div.headline", d)), icon("chevron_right")));
        if (!r.dirs.length) items.push(h("div.list-item", h("div.text.muted", "No subfolders")));
        list.replaceChildren(...items);
        if (mode === "input") { ok.disabled = !r.n_mzml; status.textContent = r.n_mzml ? `${int(r.n_mzml)} mzML files here` : "No mzML files here"; }
        else if (mode === "results") { ok.disabled = !r.is_output; status.textContent = r.is_output ? "peak3d results here" : "No peak3d results here"; }
        else { ok.disabled = !r.path || r.writable === false; status.textContent = r.writable === false ? "Not writable" : "Results go in a new subfolder here"; }
      } catch (e) {
        pathField.setError(e.message);
        list.replaceChildren();
        ok.disabled = true;
      }
    }
    dlg.showModal();
    load(start || "");
  }
  function confirmDialog(title, text, action) {
    return new Promise((resolve) => {
      const dlg = document.getElementById("confirm");
      const finish = (v) => { dlg.close(); resolve(v); };
      dlg.replaceChildren(h("div.dialog-head", h("h2.headline-small", title)), h("div.dialog-body.padded", h("p.body-medium.muted", text)),
        h("div.dialog-actions", h("button.btn.text.state", { type: "button", onclick: () => finish(false) }, "Cancel"), h("button.btn.filled.state", { type: "button", onclick: () => finish(true) }, action)));
      dlg.addEventListener("cancel", () => resolve(false), { once: true });
      dlg.showModal();
    });
  }

  // ---------------------------------------------------------------- start
  (async function init() {
    try {
      [S.info, S.schema] = await Promise.all([api("/api/info"), api("/api/schema")]);
    } catch (e) {
      pageEl.replaceChildren(h("div.page", h("div.banner.error", { style: { marginTop: "24px" } }, icon("error"), h("div", "Cannot reach the peak3d server: " + e.message))));
      return;
    }
    S.formErrors = {};
    for (const f of S.schema) S.form[f.dest] = f.kind === "list" ? [] : f.default;
    if ("cores" in S.form) S.form.cores = Math.max(1, Math.min(S.info.cpu_count || 1, 4));
    try {
      for (const r of await api("/api/runs")) { S.runs.set(r.id, { status: r, log: r.log || [] }); if (r.state === "running") poll(r.id); }
    } catch (e) { /* none */ }
    go("run");
  })();
})(window.P3 = window.P3 || {});
