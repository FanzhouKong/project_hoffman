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
  const S = { info: null, schemas: { peak3d: [], external: [] }, kernel: "peak3d", form: {}, input: "", output: "", inputInfo: null, runs: new Map(), page: "run", resultsPath: null, viewer: null, viewerTab: "overview" };
  // feature detection kernels: peak3d runs `peak3d process`, the others `peak3d external --kernel NAME` (their own
  // workflow, then the same tables, chromatograms, MS2 mapping and report)
  const KERNELS = [["peak3d", "Mosa", "scatter_plot"], ["asari", "asari", "dataset"], ["masscube", "MassCube", "experiment"]];
  const KNAME = Object.fromEntries(KERNELS.map(([k, n]) => [k, n]));
  const schemaOf = (k) => S.schemas[k === "peak3d" ? "peak3d" : "external"] || [];
  const kernelInfo = (k) => (S.info.kernels || {})[k] || { available: k === "peak3d" };
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
    library: "Spectral library (.msp or .pkl, optional)", similarity: "Similarity score", min_similarity: "Minimum similarity",
    precursor_ppm: "Precursor tolerance (ppm)", fragment_tol: "Fragment tolerance (Da)",
  };
  const HELP = {
    cores: "Files are picked in parallel; each worker holds one file in memory (about 0.5–1.5 GB).",
    no_rt_correction: "Align the runs on their own most credible features before grouping.",
    no_gap_fill: "Fill missing values from each file's chromatogram, and confirm real peaks there.",
    keep_cache: "Keeps .cache/ (large) so `peak3d align` can re-run alignment without re-picking.",
    set: "One NAME=VALUE per line; advanced picker parameters.",
    no_report: "A single HTML file with these views and chromatograms; opens in any browser, easy to share.",
    library: "MS2 spectra are mapped to the features (a consensus of all of them, and the highest-TIC one); with a library, both are searched after the run and the best consensus hit becomes each feature's annotation.",
  };
  const INVERTED = new Set(["no_rt_correction", "no_gap_fill", "no_report"]);
  const BASIC = ["cores", "polarity", "quant", "min_score", "ppm", "no_rt_correction", "no_gap_fill", "library", "similarity"];
  const SEGMENTS = { polarity: [[null, "Auto"], ["pos", "Positive"], ["neg", "Negative"]], quant: [["height", "Height"], ["area", "Area"]] };

  function stamp() {
    const d = new Date(), p = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`;
  }
  const join = (a, b) => (a.endsWith(S.info.sep) ? a + b : a + S.info.sep + b);
  const basename = (p) => p.split(/[\\/]/).filter(Boolean).pop() || p;
  const dirname = (p) => { const parts = p.split(/[\\/]/); parts.pop(); return parts.join(S.info.sep) || S.info.sep; };

  // ---------------------------------------------------------------- navigation
  const rail = document.querySelectorAll(".nav-item");
  rail.forEach((b) => {
    b.querySelector(".ic").appendChild(icon(b.dataset.page === "run" ? "play_arrow" : "insights"));
    b.addEventListener("click", () => go(b.dataset.page));
  });
  document.getElementById("logo").replaceWith(P3.logo());

  function go(page) {
    S.page = page;
    rail.forEach((b) => { if (b.dataset.page === page) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current"); });
    if (S.viewer) { S.viewer.destroy(); S.viewer = null; }
    document.getElementById("bar-actions").replaceChildren();
    pageEl.replaceChildren();
    scroll.scrollTop = 0;
    (page === "run" ? runPage : resultsPage)();
  }
  // the badge at the top right: the picker variant for peak3d, else the kernel and its version
  let variantNote = null;
  function paintVariant() {
    const chip = document.getElementById("variant");
    if (!chip) return;
    if (variantNote == null) variantNote = chip.title;
    const k = S.kernel, st = kernelInfo(k);
    chip.querySelector(".variant-k").textContent = k === "peak3d" ? "variant" : "kernel";
    document.getElementById("variant-v").textContent = k === "peak3d" ? S.info.variant : `${KNAME[k]} ${st.version || ""}`.trim();
    chip.title = k === "peak3d" ? variantNote : `Feature detection by ${KNAME[k]}${st.version ? " " + st.version : ""}${st.available ? "" : " (not available on this computer)"}`;
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
    return h("label.switch-row", h("div.text", h("div", label), help ? h("div.supporting", help) : null),
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
      return h("div", h("div.label-medium", { style: { color: "var(--ink-2)", marginBottom: "6px" } }, label),
        segmented(SEGMENTS[f.dest], S.form[f.dest], set), help && f.dest !== "quant" ? h("div.body-small.muted", { style: { marginTop: "5px" } }, help) : null);
    }
    if (f.dest === "library") {
      const t = textField(label, S.form.library, {
        lead: "description", help,
        trail: h("button.icon-btn.state", { type: "button", title: "Browse", "aria-label": "Browse for a .msp or .pkl library",
          onclick: () => browse("msp", S.form.library ? dirname(S.form.library) : S.input, (p) => { t.input.value = p; setLib(p); }) }, icon("folder_open")),
        oninput: (v) => setLib(v.trim()),
      });
      const setLib = (v) => { S.form.library = v || null; P3.storeSet("peak3d-id-library", v || ""); onAny(); };
      return t.el;
    }
    if (f.kind === "choice") {
      const names = f.dest === "similarity" ? S.info.methods || {} : {};
      const sel = h("select", f.choices.map((c) => h("option", { value: c }, names[c] || c)));
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
    const parts = ["peak3d", ...(S.kernel === "peak3d" ? ["process"] : ["external", "--kernel", S.kernel]), "--input", q(S.input || "<folder>"), "--output", q(S.output || "<folder>")];
    for (const f of schemaOf(S.kernel)) {
      const v = S.form[f.dest];
      if (f.kind === "flag") { if (v) parts.push(f.flag); }
      else if (f.kind === "list") (v || []).forEach((x) => parts.push(f.flag, q(x)));
      else if (v != null && String(v) !== "" && String(v) !== String(f.default)) parts.push(f.flag, q(v));
    }
    return parts.join(" ");
  }
  const q = (s) => (/^[\w./:=+@%,\\-]+$/.test(String(s)) ? String(s) : `"${String(s).replace(/"/g, '\\"')}"`);

  // ---------------------------------------------------------------- run page
  const INTRO = {
    peak3d: "Choose a folder of centroided .mzML files. Mosa picks every file, aligns the runs on their own features, and writes one feature table with an interactive report; MS2 spectra in the files are mapped to the features and can be searched against a spectral library.",
    asari: "Choose a folder of centroided .mzML files. asari detects and aligns the features with its own defaults; Mosa then adds chromatograms, maps the MS2 spectra and writes the same tables and report.",
    masscube: "Choose a folder of centroided .mzML files. MassCube runs its default untargeted workflow; Mosa then adds chromatograms, maps the MS2 spectra and writes the same tables and report.",
  };
  function runPage() {
    setTitle("New run");
    paintVariant();
    const page = h("div.page");
    pageEl.appendChild(page);
    const kst = kernelInfo(S.kernel);
    const tabs = h("div.tabs.kernel-tabs", { role: "tablist", "aria-label": "Kernel" }, KERNELS.map(([k, name, ic]) => {
      const st = kernelInfo(k);
      return h("button.tab.state", { type: "button", role: "tab", "aria-selected": String(k === S.kernel), title: st.available ? `${name} ${st.version || ""}`.trim() : `${name}: not available on this computer`,
        onclick: () => { if (k !== S.kernel) { S.kernel = k; P3.storeSet("peak3d-kernel", k); go("run"); } } },
        icon(ic), name, st.version && k !== "peak3d" ? h("span.tab-note", st.version) : null, st.available ? null : h("span.tab-note", "off"));
    }));
    page.appendChild(h("div.kernel-bar", h("span.label-medium.muted", "Kernel"), tabs));
    page.appendChild(h("p.page-intro", INTRO[S.kernel]));
    if (!kst.available) page.appendChild(h("div.banner.error", { style: { marginBottom: "16px" } }, icon("error"),
      h("div.body-medium", `${KNAME[S.kernel]} is not available on this computer. ${kst.note || ""}`)));
    const main = h("div.stack"), side = h("div.run-side");
    page.appendChild(h("div.run-layout", main, side));

    const runBtn = h("button.btn.filled.lg.block.state", { type: "button" }, icon("play_arrow"), "Run " + KNAME[S.kernel]);
    const cmd = h("code");
    const sumIn = h("dd"), sumOut = h("dd", { style: { fontFamily: "var(--mono)", fontSize: "12px" } }), sumCores = h("dd");
    const refresh = () => {
      cmd.textContent = argvPreview();
      const n = S.inputInfo && S.inputInfo.n_mzml;
      sumIn.textContent = n ? `${int(n)} mzML file${n > 1 ? "s" : ""}` : "No folder chosen";
      sumIn.style.color = n ? "" : "var(--ink-3)";
      sumOut.textContent = S.output ? basename(S.output) : "–";
      sumOut.title = S.output || "";
      sumCores.textContent = S.form.cores != null ? String(S.form.cores) : "–";
      const bad = !n || !S.output || !kst.available || schemaOf(S.kernel).some((f) => S.formErrors[f.dest]);
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
          if (!S.output || S.autoOutput === S.output) { S.output = S.autoOutput = join(info.path, "mosa_" + stamp()); outField.input.value = S.output; }
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
        snack(`Example data ready: ${r.n_mzml} synthetic runs. Press Run Mosa.`);
      } catch (e) {
        snack(e.message);
      } finally {
        exBtn.disabled = false;
        exBtn.replaceChildren(icon("science"), "Create example data");
      }
    });
    main.appendChild(h("section.card.elevated", h("div.card-head", h("h3.title-medium", "Data")), h("div.form", inField.el, outField.el),
      h("div.row", { style: { marginTop: "16px", gap: "12px", alignItems: "center" } }, exBtn,
        h("span.body-small.muted", { style: { flex: "1", minWidth: "200px" } }, "No mzML files at hand? This writes three small synthetic LC-MS runs into a peak3d_example folder."))));

    const basic = h("div.form"), adv = h("div.form");
    for (const f of schemaOf(S.kernel)) (BASIC.includes(f.dest) ? basic : adv).appendChild(fieldFor(f, refresh));
    main.appendChild(h("section.card.elevated", h("div.card-head", h("h3.title-medium", "Settings")), basic,
      h("details.expand", { style: { marginTop: "14px" } }, h("summary.state", icon("tune"), "Advanced options", h("span.spacer"), icon("expand_more", "chev")), h("div.form", { style: { padding: "12px 0 0" } }, adv))));

    const copyBtn = h("button.icon-btn.state", { type: "button", title: "Copy command", "aria-label": "Copy the equivalent command", onclick: () => {
      (navigator.clipboard ? navigator.clipboard.writeText(cmd.textContent) : Promise.reject()).then(() => snack("Command copied"), () => snack("Copy failed; select the text instead"));
    } }, icon("content_copy"));
    side.appendChild(h("section.card.elevated.summary", h("div.card-head", h("h3.title-medium", "Run")),
      h("dl.kv.body-medium", h("dt", "Kernel"), h("dd", `${KNAME[S.kernel]} ${S.kernel === "peak3d" ? "(" + S.info.variant + ")" : kst.version || ""}`.trim()),
        h("dt", "Input"), sumIn, h("dt", "Output"), sumOut, h("dt", "CPU cores"), sumCores),
      runBtn,
      h("div.label-medium.muted", { style: { margin: "16px 0 6px" } }, "Equivalent command"), h("div.cmd", cmd, copyBtn)));
    runBtn.addEventListener("click", () => start(false));

    const runsBox = h("div.stack#runs", { style: { gap: "12px" } });
    side.appendChild(h("div", h("h2.title-medium", { style: { margin: "4px 2px 10px" } }, "Runs"), runsBox));
    renderRuns();

    async function start(overwrite) {
      runBtn.disabled = true;
      try {
        const r = await post("/api/runs", { kernel: S.kernel, input: S.input, output: S.output, options: S.form, overwrite });
        S.runs.set(r.id, { status: r, log: r.log || [] });
        renderRuns();
        poll(r.id);
        snack("Run started");
        S.autoOutput = null;
        S.output = join(dirname(S.output), "mosa_" + stamp());       // the next run gets its own folder
        if (S.output === r.output) S.output += "_2";
        S.autoOutput = S.output;
        outField.input.value = S.output;
      } catch (e) {
        if (e.status === 409 && e.body && e.body.confirm === "overwrite") {
          if (await confirmDialog("Replace existing results?", `${S.output} already holds Mosa results. Running again replaces them.`, "Replace")) return start(true);
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
  const stepsOf = (k) => (!k || k === "peak3d" ? STEPS : [["pick", "Run " + KNAME[k]], ["group", "Convert"], ["fill", "Chromatograms & MS2"], ["report", "Report"]]);
  const EXTERNAL_TEXT = { compile: () => "Starting…", pick: (k) => `Running ${KNAME[k]} (its own progress is in the log)…`, group: () => "Converting the feature table…",
    fill: () => "Reading the files for chromatograms and MS2…" };
  const STAGE_TEXT = {
    compile: "Compiling the picker (the first run on a machine takes about a minute)…", pick: "Picking features in each file…",
    align: "Aligning retention times…", group: "Grouping features across files…", fill: "Filling gaps from the chromatograms…",
    write: "Writing tables and QC figures…", report: "Writing the report…", done: "Finished", failed: "Failed", cancelled: "Cancelled",
  };
  function renderRuns() {
    const box = document.getElementById("runs");
    if (!box) return;
    const list = [...S.runs.values()].sort((a, b) => b.status.started - a.status.started);
    if (!list.length) { box.replaceChildren(h("p.body-small.muted", { style: { margin: "0 2px" } }, "Runs you start appear here, with live progress.")); return; }
    box.replaceChildren(...list.map(runCard));
  }
  function runCard(R) {
    const s = R.status;
    const running = s.state === "running";
    const steps0 = stepsOf(s.kernel);
    const stIdx = Math.max(0, steps0.findIndex(([k]) => k === (s.stage === "write" ? "fill" : s.stage)));
    const head = h("div.row", { style: { alignItems: "flex-start" } },
      h("div", { style: { flex: 1, minWidth: 0 } }, h("div.title-small", { style: { overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" } },
        `${basename(s.input)} · ${int(s.n_files)} files${s.kernel && s.kernel !== "peak3d" ? " · " + KNAME[s.kernel] : ""}`),
        h("div.body-small.muted", { style: { overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }, title: s.output }, "→ " + basename(s.output))),
      h("span.chip.static" + (s.state === "done" ? ".ok" : s.state === "failed" ? ".warn" : ".info"), s.state === "done" ? "Done" : s.state === "failed" ? "Failed" : s.state === "cancelled" ? "Cancelled" : duration(s.elapsed)));
    const prog = h("div.progress" + (running && s.stage === "compile" ? ".indeterminate" : ""), h("div.bar", { style: { width: `${Math.round(100 * s.progress)}%` } }));
    const steps = h("div.steps", steps0.map(([k, label], i) => {
      const done = s.state === "done" || i < stIdx, active = running && i === stIdx;
      return h("span.step" + (done ? ".done" : active ? ".active" : ""), done ? icon("check") : h("span.dot"),
        k === "pick" && s.picked != null && running && i === stIdx ? `${label} ${s.picked}/${s.n_files}` : label);
    }));
    const ext = s.kernel && s.kernel !== "peak3d" && EXTERNAL_TEXT[s.stage];
    const text = h("p.body-small.muted", s.state === "done" ? `Finished in ${duration(s.elapsed)}` : ext ? ext(s.kernel) : STAGE_TEXT[s.stage] || s.stage);
    const logPre = h("pre.log", R.log.slice(-400).join("\n"));
    const logBox = h("details.expand", { open: R.logOpen || null, ontoggle: (e) => (R.logOpen = e.target.open) }, h("summary.state", icon("description"), "Log", h("span.spacer"), icon("expand_more", "chev")), logPre);
    const actions = h("div.row.wrap", { style: { marginTop: "10px" } });
    if (running) actions.appendChild(h("button.btn.text.state", { type: "button", onclick: async () => { await post(`/api/runs/${s.id}/cancel`); snack("Cancelling…"); } }, icon("stop"), "Cancel"));
    if (s.state === "done") {
      actions.append(h("button.btn.filled.state", { type: "button", onclick: () => openResults(s.output) }, "Open results"),
        h("a.btn.outlined.state", { href: `/api/results/report?path=${enc(s.output)}`, target: "_blank", rel: "noopener" }, icon("open_in_new"), "Report"));
    }
    const failed = s.state === "failed" ? h("div.banner.error", { style: { marginTop: "12px" } }, icon("error"), h("pre.mono", { style: { margin: 0, whiteSpace: "pre-wrap" } }, R.log.slice(-6).join("\n"))) : null;
    const el = h("section.card.elevated.run-card", head, h("div", { style: { marginTop: "12px" } }, prog), running ? steps : null, text, failed, actions, logBox);
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

  // ---------------------------------------------------------------- menu
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
        h("p.body-medium", "Choose a Mosa results folder (one with aligned_feature_table.tsv), or finish a run first."),
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
    const addActions = (quant, hd) => {
      const other = quant === "area" ? "height" : "area";
      hd = hd || {};
      actions.replaceChildren(
        h("a.icon-btn.state", { href: `/api/results/report?path=${enc(path)}`, target: "_blank", rel: "noopener", title: "Open report.html in a new tab", "aria-label": "Open report" }, icon("open_in_new")),
        menuButton("download", "Download", [
          { icon: "table_rows", label: `Feature table (${quant}s)`, href: fileUrl("aligned_feature_table.tsv") },
          ...(!hd.kernel || hd.kernel === "peak3d" ? [{ icon: "table_rows", label: `Feature table (${other}s)`, href: fileUrl(`aligned_feature_${other}.tsv`) }] : []),
          ...(hd.ms2c ? [{ icon: "bar_chart", label: "Consensus MS2 spectra, one per feature (.msp)", href: fileUrl("ms2_consensus.msp") }] : []),
          ...(hd.ms2 ? [{ icon: "bar_chart", label: "Highest-TIC MS2 spectra, one per feature (.msp)", href: fileUrl("ms2_spectra.msp") }] : []),
          ...(hd.ids ? [{ icon: "table_rows", label: "Library hits, both spectra (identifications.tsv)", href: fileUrl("identifications.tsv") }] : []),
          ...(hd.ids || hd.ann ? [{ icon: "table_rows", label: "Annotations (feature_annotations.tsv)", href: fileUrl("feature_annotations.tsv") }] : []),
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
      const ms2 = async () => P3.loadBundle(await api(`/api/results/ms2ids?path=${enc(path)}`));
      try {      // the MS2 and library hits as they are now (the report's copy can predate a search)
        const m = await ms2();
        for (const k of Object.keys(b.a)) if (/^(ms2_|ms2c_|id_|mk_)/.test(k)) delete b.a[k];
        Object.assign(b.a, m.a);
        for (const k of ["ms2", "ms2c", "ids", "ann"]) b.header[k] = m.header[k];
      } catch (e) { /* keep the bundle's own copy */ }
      if (S.page !== "results" || S.resultsPath !== path) return;
      const quant = b.header.quant === "area" ? "area" : "height";
      let cur = b.header;              // the header parts the download menu reflects (a search updates them)
      addActions(quant, cur);
      const exportTable = async (ids) => new Blob([await api("/api/results/table", { method: "POST", body: JSON.stringify({ path, ids }) })], { type: "text/tab-separated-values" });
      const identify = async (params) => {
        await post("/api/results/identify", Object.assign({ path }, params));
        const m = await ms2();
        cur = Object.assign({}, b.header, m.header);
        addActions(quant, cur);
        return m;
      };
      const browseLibrary = (start, done) => browse("msp", start ? dirname(start) : S.input || "", done);
      const noted = (r) => { cur = Object.assign({}, cur, { ann: true }); addActions(quant, cur); return r; };
      const annotate = async (group_id, hit) => noted(await post("/api/results/annotate", { path, group_id, hit }));
      const note = async (group_id, parts) => noted(await post("/api/results/note", Object.assign({ path, group_id }, parts)));
      S.viewer = P3.viewer.mount(holder, b, { tab: S.viewerTab, onTab: (t) => (S.viewerTab = t), exportTable, identify, annotate, note, browseLibrary, methods: S.info.methods || {} });
    } catch (e) {
      holder.replaceChildren(h("div.page", h("div.banner.error", { style: { marginTop: "16px" } }, icon("error"), h("div", e.message))));
    }
  }

  // ---------------------------------------------------------------- dialogs
  function browse(mode, start, done) {
    const dlg = document.getElementById("browse");
    const titles = { input: "Choose the mzML folder", output: "Choose where to write results", results: "Choose a results folder", msp: "Choose a spectral library (.msp or MassCube .pkl)" };
    let cur = null, file = null;
    const pathField = textField("Folder", start || "", { lead: "folder_open", onchange: (v) => load(v.trim()) });
    pathField.input.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); load(pathField.input.value.trim()); } });
    const list = h("div.list");
    const status = h("div.body-small.muted", { style: { flex: 1 } });
    const ok = h("button.btn.filled.state", { type: "button" }, "Select");
    const cancel = h("button.btn.text.state", { type: "button", onclick: () => dlg.close() }, "Cancel");
    dlg.replaceChildren(h("div.dialog-head", h("h2.headline-small", titles[mode]), pathField.el), h("div.dialog-body", list), h("div.divider"), h("div.dialog-actions", status, cancel, ok));
    ok.addEventListener("click", () => { if (mode === "msp" ? file : cur) { dlg.close(); done(mode === "msp" ? join(cur.path, file) : cur.path); } });
    async function load(p) {
      list.replaceChildren(h("div.loading", { style: { padding: "32px" } }, P3.spinner()));
      file = null;
      try {
        const r = await api(`/api/ls?path=${enc(p)}${mode === "msp" ? "&ext=msp" : ""}`);
        cur = r;
        pathField.input.value = r.path;
        pathField.setError(null);
        const items = [];
        if (r.parent != null) items.push(h("button.list-item.state", { type: "button", onclick: () => load(r.parent) }, icon("arrow_back"), h("div.text", h("div.headline", "Up one level"))));
        for (const d of r.dirs) items.push(h("button.list-item.state", { type: "button", onclick: () => load(r.path ? join(r.path, d) : d) }, icon("folder"), h("div.text", h("div.headline", d)), icon("chevron_right")));
        if (mode === "msp") for (const f of r.files || []) {
          const it = h("button.list-item.state", { type: "button", "aria-selected": "false",
            onclick: () => { file = f; list.querySelectorAll("[data-file]").forEach((e) => e.setAttribute("aria-selected", String(e === it))); ok.disabled = false; status.textContent = f; },
            ondblclick: () => { file = f; ok.click(); }, "data-file": f }, icon("description"), h("div.text", h("div.headline", f)));
          items.push(it);
        }
        if (!r.dirs.length && !(mode === "msp" && (r.files || []).length)) items.push(h("div.list-item", h("div.text.muted", mode === "msp" ? "No subfolders or library files" : "No subfolders")));
        list.replaceChildren(...items);
        if (mode === "msp") { ok.disabled = true; status.textContent = (r.files || []).length ? `${int(r.files.length)} library file${r.files.length > 1 ? "s" : ""} here: pick one` : "No .msp or .pkl files here"; }
        else if (mode === "input") { ok.disabled = !r.n_mzml; status.textContent = r.n_mzml ? `${int(r.n_mzml)} mzML files here` : "No mzML files here"; }
        else if (mode === "results") { ok.disabled = !r.is_output; status.textContent = r.is_output ? "Mosa results here" : "No Mosa results here"; }
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
      [S.info, S.schemas.peak3d, S.schemas.external] = await Promise.all([api("/api/info"), api("/api/schema"), api("/api/schema?kernel=asari")]);
    } catch (e) {
      pageEl.replaceChildren(h("div.page", h("div.banner.error", { style: { marginTop: "24px" } }, icon("error"), h("div", "Cannot reach the Mosa server: " + e.message))));
      return;
    }
    S.formErrors = {};
    for (const f of [...S.schemas.peak3d, ...S.schemas.external]) if (!(f.dest in S.form)) S.form[f.dest] = f.kind === "list" ? [] : f.default;
    if ("cores" in S.form) S.form.cores = Math.max(1, Math.min(S.info.cpu_count || 1, 4));
    if ("library" in S.form) S.form.library = P3.storeGet("peak3d-id-library") || null;     // search automatically with the last library
    const k = P3.storeGet("peak3d-kernel");
    if (k && KNAME[k]) S.kernel = k;
    try {
      for (const r of await api("/api/runs")) { S.runs.set(r.id, { status: r, log: r.log || [] }); if (r.state === "running") poll(r.id); }
    } catch (e) { /* none */ }
    go("run");
  })();
})(window.P3 = window.P3 || {});
