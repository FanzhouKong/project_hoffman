// Shared helpers for the peak3d viewer and GUI: DOM building, icons, the logo, number formats, toasts,
// and decoding of the results bundle (see peak3d/web/bundle.py for the layout).
(function (P3) {
  "use strict";

  // h("div.card.elevated", {onclick: f, style: "..."}, child, "text", [children]) -> Element
  function h(sel, attrs, ...kids) {
    const m = sel.match(/^([a-z0-9-]+)?((?:\.[\w-]+)*)(?:#([\w-]+))?$/i);
    const el = document.createElement((m && m[1]) || "div");
    if (m && m[2]) el.className = m[2].slice(1).split(".").join(" ");
    if (m && m[3]) el.id = m[3];
    if (attrs != null && (typeof attrs !== "object" || attrs instanceof Node || Array.isArray(attrs))) { kids.unshift(attrs); attrs = null; }
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
      else if (k === "html") el.innerHTML = v;
      else if (k in el && k !== "list" && typeof v !== "string") el[k] = v;
      else el.setAttribute(k, v === true ? "" : v);
    }
    append(el, kids);
    return el;
  }
  function append(el, kids) {
    for (const k of kids.flat(Infinity)) {
      if (k == null || k === false) continue;
      el.appendChild(k instanceof Node ? k : document.createTextNode(String(k)));
    }
    return el;
  }
  const SVG = "http://www.w3.org/2000/svg";
  function icon(name, cls) {
    const s = document.createElementNS(SVG, "svg");
    s.setAttribute("viewBox", "0 -960 960 960");
    s.setAttribute("class", "icon" + (cls ? " " + cls : ""));
    s.setAttribute("aria-hidden", "true");
    for (const d of P3.ICONS[name] || []) {
      const p = document.createElementNS(SVG, "path");
      p.setAttribute("d", d);
      s.appendChild(p);
    }
    return s;
  }

  function spinner(size) {
    const s = document.createElementNS(SVG, "svg");
    s.setAttribute("viewBox", "0 0 48 48");
    s.setAttribute("class", "spinner");
    if (size) { s.style.width = size + "px"; s.style.height = size + "px"; }
    const c = document.createElementNS(SVG, "circle");
    c.setAttribute("cx", "24"); c.setAttribute("cy", "24"); c.setAttribute("r", "20");
    s.appendChild(c);
    return s;
  }

  // ---- number formats
  const nf = (d) => new Intl.NumberFormat(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
  const NF = [0, 1, 2, 3, 4, 5].map(nf);
  function fixed(v, d) { return v == null || !isFinite(v) ? "–" : NF[d].format(v); }
  function int(v) { return v == null || !isFinite(v) ? "–" : NF[0].format(Math.round(v)); }
  function compact(v) {
    if (v == null || !isFinite(v)) return "–";
    const a = Math.abs(v);
    for (const [s, u] of [[1e12, "T"], [1e9, "B"], [1e6, "M"], [1e3, "K"]]) {
      if (a >= s) { const x = v / s; return trim(Math.abs(x) >= 100 ? x.toFixed(0) : Math.abs(x) >= 10 ? x.toFixed(1) : x.toFixed(2)) + u; }
    }
    return a >= 100 ? v.toFixed(0) : a >= 1 ? +v.toFixed(2) + "" : a === 0 ? "0" : +v.toPrecision(2) + "";
  }
  function trim(s) { return s.includes(".") ? s.replace(/0+$/, "").replace(/\.$/, "") : s; }
  function sci(v) { // intensities: 3 significant digits, compact suffix
    if (v == null || !isFinite(v)) return "–";
    return v === 0 ? "0" : compact(+v.toPrecision(3));
  }
  function duration(s) {
    if (s == null || !isFinite(s)) return "–";
    if (s < 60) return s.toFixed(s < 10 ? 1 : 0) + " s";
    const m = Math.floor(s / 60), r = Math.round(s - 60 * m);
    if (m < 60) return m + " min " + (r ? r + " s" : "");
    return Math.floor(m / 60) + " h " + (m % 60) + " min";
  }

  // light theme only; charts subscribe in case the palette ever changes at run time
  const listeners = new Set();
  function onTheme(f) { listeners.add(f); return () => listeners.delete(f); }
  function storeGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
  function storeSet(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } }

  // the mark: a chromatographic peak on a blue tile
  // the MOSA mark (logo.js); also the browser tab's icon
  function logo() {
    const el = h("span.logo", { "aria-hidden": "true" });
    if (P3.LOGO) el.append(h("img", { src: P3.LOGO, alt: "" }));
    return el;
  }
  if (P3.LOGO && document.head && !document.querySelector('link[rel="icon"]')) document.head.append(h("link", { rel: "icon", type: "image/png", href: P3.LOGO }));
  function css(name, el) { return getComputedStyle(el || document.documentElement).getPropertyValue(name).trim(); }

  // ---- downloads
  function saveBlob(blob, name) {
    const url = URL.createObjectURL(blob);
    const a = h("a", { href: url, download: name, style: { display: "none" } });
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 2000);
  }

  // ---- snackbar
  let snackTimer = null;
  function snack(text, action) {
    let s = document.querySelector(".snackbar");
    if (!s) { s = h("div.snackbar", { role: "status", "aria-live": "polite" }); document.body.appendChild(s); }
    s.replaceChildren(h("span", text));
    if (action) s.appendChild(h("button.btn.text.state", { type: "button", onclick: () => { s.classList.remove("show"); action.run(); } }, action.label));
    s.classList.add("show");
    clearTimeout(snackTimer);
    snackTimer = setTimeout(() => s.classList.remove("show"), action ? 8000 : 4000);
  }

  // ---- results bundle
  const DT = { f8: Float64Array, f4: Float32Array, i4: Int32Array, u4: Uint32Array, i2: Int16Array, u2: Uint16Array, i1: Int8Array, u1: Uint8Array, b1: Uint8Array };
  async function gunzip(bytes) {
    if (!("DecompressionStream" in window)) throw new Error("This browser is too old to open the report (needs DecompressionStream: Chrome 80+, Firefox 113+, Safari 16.4+).");
    const ds = new DecompressionStream("gzip");
    const stream = new Blob([bytes]).stream().pipeThrough(ds);
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }
  function b64bytes(text) {
    const bin = atob(text.replace(/\s+/g, ""));
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  function parseBundle(raw) {
    const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    const n = dv.getUint32(0, true);
    const header = JSON.parse(new TextDecoder().decode(raw.subarray(4, 4 + n)));
    let start = 4 + n; start += (8 - (start % 8)) % 8;
    const buf = raw.buffer.slice(raw.byteOffset + start, raw.byteOffset + raw.byteLength);   // 8-aligned copy
    const arrays = {};
    for (const [k, m] of Object.entries(header.arrays)) {
      const T = DT[m.dtype];
      if (!T) throw new Error("unknown dtype " + m.dtype);
      arrays[k] = new T(buf, m.offset, m.length);
    }
    delete header.arrays;
    return { header, a: arrays };
  }
  async function loadBundle(gzBytes) { return parseBundle(await gunzip(gzBytes)); }

  Object.assign(P3, { h, append, icon, spinner, saveBlob, logo, fixed, int, compact, sci, duration, onTheme, css, snack, loadBundle, b64bytes, storeGet, storeSet });
})(window.P3 = window.P3 || {});
