// Small canvas charts for the viewer: line (EICs, drift curves, histograms), scatter (feature map with box
// zoom) and bars (per-file values). Colours are CSS variables resolved at draw time, so charts repaint on a
// theme change. Marks follow the house rules: 2px lines, recessive hairline grid, tooltips on hover.
(function (P3) {
  "use strict";
  const { h, css } = P3;

  function niceStep(span, n) {
    const raw = span / Math.max(1, n);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const f = raw / mag;
    return mag * (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10);
  }
  function ticks(lo, hi, n) {
    if (!(hi > lo)) hi = lo + (Math.abs(lo) || 1);
    const step = niceStep(hi - lo, n);
    const out = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return { values: out, digits: Math.max(0, -Math.floor(Math.log10(step) + 1e-9)) };
  }
  function hexRgb(c) {
    const m = /^#?([0-9a-f]{6})$/i.exec(c.trim());
    if (!m) return [128, 128, 128];
    const n = parseInt(m[1], 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function mix(a, b, t) {
    const A = hexRgb(a), B = hexRgb(b);
    return "rgb(" + A.map((v, i) => Math.round(v + (B[i] - v) * t)).join(",") + ")";
  }
  function rgba(c, alpha) {
    if (c.startsWith("rgb(")) return c.replace("rgb(", "rgba(").replace(")", "," + alpha + ")");
    const [r, g, b] = hexRgb(c);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  class Chart {
    constructor(el, spec) {
      this.el = el;
      this.spec = spec;
      this.height = spec.height || 280;
      el.classList.add("chart");
      el.replaceChildren();
      this.canvas = h("canvas", { role: "img", "aria-label": spec.ariaLabel || spec.yLabel || "chart" });
      this.tip = h("div.tooltip");
      this.hint = h("div.chart-hint");
      el.append(this.canvas, this.tip, this.hint);
      this.ctx = this.canvas.getContext("2d");
      this.w = 0;
      this.ro = new ResizeObserver(() => this.resize());
      this.ro.observe(el);
      this.offTheme = P3.onTheme(() => this.draw());
      this.canvas.addEventListener("pointermove", (e) => this.onMove(this.pos(e), e));
      this.canvas.addEventListener("pointerleave", () => this.onLeave());
      this.canvas.addEventListener("click", (e) => this.onClick && this.onClick(this.pos(e), e));
    }
    pos(e) { const r = this.canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
    resize() {
      const w = Math.floor(this.el.clientWidth);
      if (!w || (w === this.w && this.canvas.height)) return;
      const dpr = window.devicePixelRatio || 1;
      this.w = w;
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(this.height * dpr);
      this.canvas.style.height = this.height + "px";
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.draw();
    }
    destroy() { this.ro.disconnect(); this.offTheme(); }
    update(spec) { this.spec = Object.assign(this.spec, spec); this.draw(); }
    pal() {
      const g = (n) => css(n, this.el);
      return { surface: g("--viz-surface"), ink: g("--viz-ink"), ink2: g("--viz-ink-2"), muted: g("--viz-muted"),
        grid: g("--viz-grid"), axis: g("--viz-axis"), s1: g("--viz-series-1"), s2: g("--viz-series-2"),
        from: g("--viz-ramp-from"), to: g("--viz-ramp-to"), neutral: g("--viz-neutral"),
        font: g("--font-viz") || "system-ui, sans-serif" };
    }
    color(c, P) { return !c ? P.s1 : c.startsWith("--") ? css(c, this.el) : c; }
    layout(P, yFmt, yt) {
      const ctx = this.ctx, s = this.spec;
      ctx.font = `400 11px ${P.font}`;
      let wmax = 0;
      for (const v of yt.values) wmax = Math.max(wmax, ctx.measureText(yFmt(v, yt.digits)).width);
      const l = Math.ceil(wmax) + (s.yLabel ? 30 : 14);
      this.box = { x0: l, x1: this.w - (s.marginRight || 16), y0: 12, y1: this.height - (s.xLabel ? 44 : 28) };
    }
    sx(v) { const b = this.box, d = this.xd; return b.x0 + (v - d[0]) / (d[1] - d[0] || 1) * (b.x1 - b.x0); }
    sy(v) { const b = this.box, d = this.yd; return b.y1 - (v - d[0]) / (d[1] - d[0] || 1) * (b.y1 - b.y0); }
    ix(px) { const b = this.box, d = this.xd; return d[0] + (px - b.x0) / (b.x1 - b.x0) * (d[1] - d[0]); }
    iy(py) { const b = this.box, d = this.yd; return d[0] + (b.y1 - py) / (b.y1 - b.y0) * (d[1] - d[0]); }
    clear(P) {
      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.w, this.height);
      ctx.fillStyle = P.surface;
      ctx.fillRect(0, 0, this.w, this.height);
    }
    axes(P, xt, yt, xFmt, yFmt, opts) {
      const ctx = this.ctx, b = this.box, s = this.spec;
      ctx.lineWidth = 1;
      ctx.strokeStyle = P.grid;
      ctx.beginPath();
      for (const v of yt.values) { const y = Math.round(this.sy(v)) + 0.5; ctx.moveTo(b.x0, y); ctx.lineTo(b.x1, y); }
      ctx.stroke();
      ctx.strokeStyle = P.axis;
      ctx.beginPath();
      ctx.moveTo(b.x0, Math.round(b.y1) + 0.5);
      ctx.lineTo(b.x1, Math.round(b.y1) + 0.5);
      ctx.stroke();
      ctx.fillStyle = P.muted;
      ctx.font = `400 11px ${P.font}`;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (const v of yt.values) ctx.fillText(yFmt(v, yt.digits), b.x0 - 8, this.sy(v));
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      if (!(opts && opts.noXTicks)) {
        let last = -Infinity;
        for (const v of xt.values) {
          const x = this.sx(v), t = xFmt(v, xt.digits), w = ctx.measureText(t).width;
          if (x - w / 2 < last + 8 || x < b.x0 - 1 || x > b.x1 + 1) continue;
          ctx.fillText(t, x, b.y1 + 6);
          last = x + w / 2;
        }
      }
      ctx.fillStyle = P.ink2;
      ctx.font = `500 12px ${P.font}`;
      if (s.xLabel) ctx.fillText(s.xLabel, (b.x0 + b.x1) / 2, this.height - 18);
      if (s.yLabel) {
        ctx.save();
        ctx.translate(12, (b.y0 + b.y1) / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textBaseline = "middle";
        ctx.fillText(s.yLabel, 0, 0);
        ctx.restore();
      }
    }
    showTip(px, py, html) {
      if (!html) return this.hideTip();
      const t = this.tip;
      t.innerHTML = html;
      t.classList.add("show");
      const tw = t.offsetWidth, th = t.offsetHeight;
      let x = px + 14, y = py - th - 10;
      if (x + tw > this.w) x = px - tw - 14;
      if (y < 0) y = py + 14;
      t.style.left = Math.max(0, x) + "px";
      t.style.top = y + "px";
    }
    hideTip() { this.tip.classList.remove("show"); }
    onLeave() { this.hideTip(); }
  }

  const ident = (v, d) => P3.fixed(v, Math.min(d, 4));

  // ---------------------------------------------------------------- line
  class Line extends Chart {
    constructor(el, spec) { super(el, spec); this.hover = null; }
    domains() {
      const s = this.spec;
      let x0 = Infinity, x1 = -Infinity, y0 = s.yZero === false ? Infinity : 0, y1 = -Infinity;
      for (const r of s.series) {
        for (let i = 0; i < r.x.length; i++) {
          const x = r.x[i], y = r.y[i];
          if (!isFinite(x) || !isFinite(y)) continue;
          if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
        }
      }
      if (!isFinite(x0)) { x0 = 0; x1 = 1; y0 = 0; y1 = 1; }
      if (s.xDomain) [x0, x1] = s.xDomain;
      if (s.yDomain) [y0, y1] = s.yDomain;
      if (y1 <= y0) y1 = y0 + 1;
      const pad = (y1 - y0) * 0.06;
      return [[x0, x1], [s.yZero === false ? y0 - pad : y0, y1 + pad]];
    }
    draw() {
      if (!this.w) return;
      const s = this.spec, P = this.pal(), ctx = this.ctx;
      [this.xd, this.yd] = this.domains();
      const xFmt = s.xFmt || ident, yFmt = s.yFmt || ((v) => P3.compact(v));
      const yt = ticks(this.yd[0], this.yd[1], Math.max(3, Math.floor((this.height - 60) / 50)));
      this.layout(P, yFmt, yt);
      const xt = ticks(this.xd[0], this.xd[1], Math.max(2, Math.floor((this.box.x1 - this.box.x0) / 90)));
      this.clear(P);
      this.axes(P, xt, yt, xFmt, yFmt);
      const b = this.box;
      ctx.save();
      ctx.beginPath();
      ctx.rect(b.x0, b.y0 - 4, b.x1 - b.x0, b.y1 - b.y0 + 4);
      ctx.clip();
      const active = this.hover != null ? this.hover : s.highlight != null ? s.series.findIndex((r) => r.id === s.highlight) : -1;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      const order = s.series.map((_, i) => i).filter((i) => i !== active);
      if (active >= 0) order.push(active);
      for (const i of order) {
        const r = s.series[i];
        const col = this.color(r.color, P);
        const dim = active >= 0 && i !== active && this.hover != null;
        if (r.fill) {
          ctx.beginPath();
          this.path(r, s.step);
          ctx.lineTo(this.sx(r.x[r.x.length - 1]), this.sy(Math.max(this.yd[0], 0)));
          ctx.lineTo(this.sx(r.x[0]), this.sy(Math.max(this.yd[0], 0)));
          ctx.closePath();
          ctx.fillStyle = rgba(col, dim ? 0.04 : 0.1);
          ctx.fill();
        }
        ctx.beginPath();
        this.path(r, s.step);
        ctx.strokeStyle = dim ? rgba(col, 0.22) : r.alpha != null && i !== active ? rgba(col, r.alpha) : col;
        ctx.lineWidth = i === active ? (r.width || 2) + 1 : r.width || 2;
        ctx.stroke();
      }
      if (s.vline != null && isFinite(s.vline)) {
        ctx.strokeStyle = P.muted;
        ctx.setLineDash([3, 3]);
        ctx.lineWidth = 1;
        const x = Math.round(this.sx(s.vline)) + 0.5;
        ctx.beginPath(); ctx.moveTo(x, b.y0); ctx.lineTo(x, b.y1); ctx.stroke();
        ctx.setLineDash([]);
      }
      // vertical marks {x, color, dash, width} (peak bounds, MS2 times) and dots {x, y, color} (apexes)
      for (const v of s.vlines || []) {
        if (!isFinite(v.x)) continue;
        ctx.strokeStyle = this.color(v.color || P.muted, P);
        ctx.setLineDash(v.dash || [4, 4]);
        ctx.lineWidth = v.width || 1;
        const x = Math.round(this.sx(v.x)) + 0.5;
        ctx.beginPath(); ctx.moveTo(x, b.y0); ctx.lineTo(x, b.y1); ctx.stroke();
      }
      ctx.setLineDash([]);
      for (const d of s.dots || []) {        // {x, y, color, r, shape: "dot" | "star"}
        if (!isFinite(d.x) || !isFinite(d.y)) continue;
        ctx.fillStyle = this.color(d.color, P);
        ctx.strokeStyle = P.surface;
        ctx.lineWidth = 1.5;
        const X = this.sx(d.x), Y = this.sy(d.y), r = d.r || 3.5;
        ctx.beginPath();
        if (d.shape === "star") {
          for (let k = 0; k < 10; k++) {
            const a = -Math.PI / 2 + k * Math.PI / 5, rr = k % 2 ? r * 0.45 : r;
            if (k) ctx.lineTo(X + rr * Math.cos(a), Y + rr * Math.sin(a)); else ctx.moveTo(X + rr * Math.cos(a), Y + rr * Math.sin(a));
          }
          ctx.closePath();
        } else ctx.arc(X, Y, r, 0, 2 * Math.PI);
        ctx.fill(); ctx.stroke();
      }
      ctx.restore();
      if (active >= 0 && s.label) {     // direct label on the active series, at its right end
        const r = s.series[active];
        let k = r.x.length - 1;
        while (k > 0 && !isFinite(r.y[k])) k--;
        const t = s.label(active);
        if (t) {
          ctx.font = `500 11px ${P.font}`;
          ctx.fillStyle = P.ink;
          ctx.textAlign = "right";
          ctx.textBaseline = "bottom";
          ctx.fillText(t, Math.min(this.sx(r.x[k]), b.x1) - 2, Math.max(this.sy(r.y[k]) - 6, b.y0 + 12));
        }
      }
      if (this.hoverPt) {
        const [hx, hy] = this.hoverPt;
        ctx.strokeStyle = P.axis;
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(Math.round(hx) + 0.5, b.y0); ctx.lineTo(Math.round(hx) + 0.5, b.y1); ctx.stroke();
        const r = s.series[this.hover];
        ctx.fillStyle = this.color(r.color, P);
        ctx.strokeStyle = P.surface;
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(hx, hy, 4.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      }
    }
    path(r, step) {
      const ctx = this.ctx;
      let pen = false;
      for (let i = 0; i < r.x.length; i++) {
        const y = r.y[i];
        if (!isFinite(y)) { pen = false; continue; }
        const X = this.sx(r.x[i]), Y = this.sy(y);
        if (step) {
          const half = (i + 1 < r.x.length ? r.x[i + 1] - r.x[i] : r.x[i] - r.x[i - 1]) / 2;
          const a = this.sx(r.x[i] - half), c = this.sx(r.x[i] + half);
          if (!pen) ctx.moveTo(a, Y); else ctx.lineTo(a, Y);
          ctx.lineTo(c, Y);
        } else if (!pen) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
        pen = true;
      }
    }
    nearest(px, py) {
      const s = this.spec, xv = this.ix(px);
      let best = null, bd = 28 * 28;
      s.series.forEach((r, si) => {
        let lo = 0, hi = r.x.length - 1;
        if (hi < 0) return;
        while (lo < hi) { const m = (lo + hi) >> 1; if (r.x[m] < xv) lo = m + 1; else hi = m; }
        for (let k = Math.max(0, lo - 2); k <= Math.min(r.x.length - 1, lo + 2); k++) {
          if (!isFinite(r.y[k])) continue;
          const dx = this.sx(r.x[k]) - px, dy = this.sy(r.y[k]) - py, d = dx * dx + dy * dy;
          if (d < bd) { bd = d; best = [si, k]; }
        }
      });
      return best;
    }
    onMove([px, py]) {
      if (!this.box) return;
      const n = this.nearest(px, py);
      const prev = this.hover;
      if (!n) { this.hover = null; this.hoverPt = null; this.hideTip(); if (prev != null) this.draw(); return; }
      const [si, k] = n, r = this.spec.series[si];
      this.hover = si;
      this.hoverPt = [this.sx(r.x[k]), this.sy(r.y[k])];
      this.draw();
      this.showTip(this.hoverPt[0], this.hoverPt[1], this.spec.tip ? this.spec.tip(si, k) : r.label);
      if (this.spec.onHover && prev !== si) this.spec.onHover(si);
    }
    onLeave() {
      this.hideTip();
      const had = this.hover != null;
      this.hover = null; this.hoverPt = null;
      if (had) { this.draw(); if (this.spec.onHover) this.spec.onHover(null); }
    }
  }

  // ---------------------------------------------------------------- scatter (feature map)
  class Scatter extends Chart {
    constructor(el, spec) {
      super(el, spec);
      this.view = null;
      this.drag = null;
      this.order = null;
      this.canvas.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) return;
        this.drag = { a: this.pos(e), b: this.pos(e) };
        this.canvas.setPointerCapture(e.pointerId);
      });
      this.canvas.addEventListener("pointerup", (e) => {
        const d = this.drag;
        this.drag = null;
        if (!d) return;
        const [ax, ay] = d.a, [bx, by] = d.b;
        if (Math.abs(ax - bx) > 6 && Math.abs(ay - by) > 6) {
          const xr = [this.ix(Math.min(ax, bx)), this.ix(Math.max(ax, bx))], yr = [this.iy(Math.max(ay, by)), this.iy(Math.min(ay, by))];
          if (this.spec.onBrush) this.spec.onBrush(xr, yr);       // brush mode: the box sets a filter, the view stays
          else this.view = [xr, yr];
          this.suppressClick = true;
        }
        this.draw();
      });
      this.canvas.addEventListener("dblclick", () => {
        if (this.spec.onBrush) this.spec.onBrush(null, null); else this.view = null;
        this.draw();
      });
      this.onClick = ([px, py]) => {
        if (this.suppressClick) { this.suppressClick = false; return; }
        const i = this.pick(px, py);
        if (i != null && this.spec.onClick) this.spec.onClick(i);
      };
    }
    full() {
      const { x, y } = this.spec;
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (let i = 0; i < x.length; i++) {
        if (x[i] < x0) x0 = x[i]; if (x[i] > x1) x1 = x[i]; if (y[i] < y0) y0 = y[i]; if (y[i] > y1) y1 = y[i];
      }
      if (!isFinite(x0)) return [[0, 1], [0, 1]];
      const px = (x1 - x0) * 0.02 || 1, py = (y1 - y0) * 0.03 || 1;
      return [[x0 - px, x1 + px], [y0 - py, y1 + py]];
    }
    draw() {
      if (!this.w) return;
      const s = this.spec, P = this.pal(), ctx = this.ctx;
      [this.xd, this.yd] = this.view || this.full();
      const xFmt = s.xFmt || ident, yFmt = s.yFmt || ident;
      const yt = ticks(this.yd[0], this.yd[1], Math.max(3, Math.floor((this.height - 60) / 50)));
      this.layout(P, yFmt, yt);
      const xt = ticks(this.xd[0], this.xd[1], Math.max(2, Math.floor((this.box.x1 - this.box.x0) / 90)));
      this.clear(P);
      this.axes(P, xt, yt, xFmt, yFmt);
      const b = this.box, n = s.x.length, v = s.v;
      if (!this.order) {   // faint first, intense on top
        this.order = Array.from({ length: n }, (_, i) => i);
        if (v) this.order.sort((a, c) => v[a] - v[c]);
      }
      const NB = 24, cols = [];
      for (let k = 0; k < NB; k++) cols.push(mix(P.from, P.to, k / (NB - 1)));
      const [v0, v1] = s.vDomain || [0, 1];
      const r = s.r || (n > 20000 ? 1.3 : n > 3000 ? 2 : n > 500 ? 3 : 4);
      ctx.save();
      ctx.beginPath(); ctx.rect(b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0); ctx.clip();
      const grid = new Map(), cell = 10;
      const mask = s.mask, emph = s.emph;     // emph: 1 = drawn in colour on top, 0 = faint grey underneath
      for (const pass of emph ? [0, 1] : [1]) {
        let lastC = -1;
        if (pass === 0) ctx.fillStyle = rgba(P.neutral, 0.35);
        for (const i of this.order) {
          if (mask && !mask[i]) continue;
          if (emph && (emph[i] ? 1 : 0) !== pass) continue;
          const X = this.sx(s.x[i]), Y = this.sy(s.y[i]);
          if (X < b.x0 - 3 || X > b.x1 + 3 || Y < b.y0 - 3 || Y > b.y1 + 3) continue;
          if (pass) {
            const c = v ? Math.max(0, Math.min(NB - 1, Math.round((v[i] - v0) / (v1 - v0 || 1) * (NB - 1)))) : NB - 1;
            if (c !== lastC) { ctx.fillStyle = cols[c]; lastC = c; }
          }
          if (n > 5000) ctx.fillRect(X - r, Y - r, 2 * r, 2 * r);           // dense maps: squares draw fastest
          else { ctx.beginPath(); ctx.arc(X, Y, r, 0, 2 * Math.PI); ctx.fill(); }
          const key = Math.floor(X / cell) + "," + Math.floor(Y / cell);
          const arr = grid.get(key);
          if (arr) arr.push(i); else grid.set(key, [i]);
        }
      }
      this.grid = grid;
      if (s.box) {           // the filter window [[x0, x1], [y0, y1]]; open ends run to the edge
        const [[bx0, bx1], [by0, by1]] = s.box;
        const X0 = isFinite(bx0) ? this.sx(bx0) : b.x0, X1 = isFinite(bx1) ? this.sx(bx1) : b.x1;
        const Y0 = isFinite(by1) ? this.sy(by1) : b.y0, Y1 = isFinite(by0) ? this.sy(by0) : b.y1;
        ctx.strokeStyle = P.s1;
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 3]);
        ctx.strokeRect(Math.round(X0) + 0.5, Math.round(Y0) + 0.5, Math.round(X1 - X0), Math.round(Y1 - Y0));
        ctx.setLineDash([]);
      }
      const sel = s.selected;
      if (sel != null && sel >= 0) {
        const X = this.sx(s.x[sel]), Y = this.sy(s.y[sel]);
        ctx.lineWidth = 2;
        ctx.strokeStyle = P.surface;
        ctx.beginPath(); ctx.arc(X, Y, 8, 0, 2 * Math.PI); ctx.stroke();
        ctx.strokeStyle = P.s2;
        ctx.beginPath(); ctx.arc(X, Y, 6.5, 0, 2 * Math.PI); ctx.stroke();
      }
      if (this.hoverI != null) {
        const X = this.sx(s.x[this.hoverI]), Y = this.sy(s.y[this.hoverI]);
        ctx.fillStyle = P.ink;
        ctx.strokeStyle = P.surface;
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(X, Y, 4.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      }
      if (this.drag) {
        const [ax, ay] = this.drag.a, [bx, by] = this.drag.b;
        ctx.fillStyle = rgba(P.s1, 0.12);
        ctx.strokeStyle = P.s1;
        ctx.lineWidth = 1;
        ctx.fillRect(Math.min(ax, bx), Math.min(ay, by), Math.abs(bx - ax), Math.abs(by - ay));
        ctx.strokeRect(Math.min(ax, bx) + 0.5, Math.min(ay, by) + 0.5, Math.abs(bx - ax), Math.abs(by - ay));
      }
      ctx.restore();
      this.hint.textContent = s.onBrush ? s.brushHint || "Drag to filter" : this.view ? "Double-click to reset zoom" : "Drag to zoom";
    }
    pick(px, py) {
      if (!this.grid) return null;
      const cell = 10, cx = Math.floor(px / cell), cy = Math.floor(py / cell);
      let best = null, bd = 9 * 9;
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
        const arr = this.grid.get(cx + dx + "," + (cy + dy));
        if (!arr) continue;
        for (const i of arr) {
          const X = this.sx(this.spec.x[i]) - px, Y = this.sy(this.spec.y[i]) - py, d = X * X + Y * Y;
          if (d <= bd) { bd = d; best = i; }
        }
      }
      return best;
    }
    onMove([px, py]) {
      if (this.drag) { this.drag.b = [px, py]; this.draw(); return; }
      const i = this.pick(px, py);
      if (i !== this.hoverI) { this.hoverI = i; this.draw(); }
      this.canvas.style.cursor = i != null ? "pointer" : "crosshair";
      if (i == null) return this.hideTip();
      this.showTip(this.sx(this.spec.x[i]), this.sy(this.spec.y[i]), this.spec.tip ? this.spec.tip(i) : "");
    }
    onLeave() { this.hideTip(); if (this.hoverI != null) { this.hoverI = null; this.draw(); } }
    update(spec) { if (spec.x || spec.v) { this.order = null; this.view = null; } super.update(spec); }
  }

  // ---------------------------------------------------------------- bars
  class Bars extends Chart {
    draw() {
      if (!this.w) return;
      const s = this.spec, P = this.pal(), ctx = this.ctx, n = s.values.length;
      let y1 = 0;
      for (const v of s.values) if (isFinite(v) && v > y1) y1 = v;
      this.xd = [-0.5, n - 0.5];
      this.yd = [0, (y1 || 1) * 1.06];
      const yFmt = s.yFmt || ((v) => P3.compact(v));
      const yt = ticks(this.yd[0], this.yd[1], Math.max(3, Math.floor((this.height - 60) / 50)));
      this.layout(P, yFmt, yt);
      this.clear(P);
      const xt = { values: [], digits: 0 };
      const band = (this.box.x1 - this.box.x0) / Math.max(1, n);
      const every = Math.max(1, Math.ceil(70 / band));
      if (s.xTick) for (let i = 0; i < n; i += every) xt.values.push(i);
      this.axes(P, xt, yt, (i) => (s.xTick ? s.xTick(i) : ""), yFmt, { noXTicks: !s.xTick });
      const bw = Math.max(1, Math.min(24, band - (band > 4 ? 2 : 0)));
      const base = this.sy(0);
      for (let i = 0; i < n; i++) {
        const v = s.values[i];
        if (!isFinite(v) || v <= 0) continue;
        const x = this.sx(i) - bw / 2, y = this.sy(v), hgt = base - y;
        const col = this.color(s.colors ? s.colors[i] : s.color, P);
        ctx.fillStyle = this.hover === i ? mixHover(col, P) : col;
        const rr = bw >= 8 ? Math.min(4, hgt) : 0;
        ctx.beginPath();
        if (rr > 0 && ctx.roundRect) ctx.roundRect(x, y, bw, hgt, [rr, rr, 0, 0]); else ctx.rect(x, y, bw, hgt);
        ctx.fill();
      }
    }
    onMove([px, py]) {
      if (!this.box) return;
      const i = Math.round(this.ix(px));
      const n = this.spec.values.length;
      if (i < 0 || i >= n || px < this.box.x0 || px > this.box.x1) { this.onLeave(); return; }
      if (i !== this.hover) { this.hover = i; this.draw(); }
      const v = this.spec.values[i];
      this.showTip(this.sx(i), Math.min(this.sy(isFinite(v) ? v : 0), this.box.y1 - 8), this.spec.tip ? this.spec.tip(i) : P3.compact(v));
    }
    onLeave() { this.hideTip(); if (this.hover != null) { this.hover = null; this.draw(); } }
  }
  function mixHover(col, P) {
    return /^#/.test(col) ? mix(col, P.ink.startsWith("#") ? P.ink : "#000000", 0.25) : col;
  }

  // ---------------------------------------------------------------- mirror (an MS2 spectrum against a library hit)
  // spec: top / bottom = { mz, inten } (bottom optional), tol = fragment tolerance (Da) for the matched colouring.
  // Each side is scaled to its own base peak (100); peaks with a partner on the other side within tol are drawn
  // in the series colour, the rest in the neutral grey.
  class Mirror extends Chart {
    prep() {
      const s = this.spec, sides = [s.top, s.bottom].filter(Boolean);
      this.sides = sides.map((r) => {
        let m = 0;
        for (let i = 0; i < r.inten.length; i++) if (r.inten[i] > m) m = r.inten[i];
        return { mz: r.mz, rel: Float32Array.from(r.inten, (v) => (m > 0 ? 100 * v / m : 0)), hit: new Uint8Array(r.mz.length) };
      });
      if (this.sides.length === 2 && s.tol > 0) {
        const [A, B] = this.sides;
        const oa = Array.from(A.mz.keys()).sort((i, j) => A.mz[i] - A.mz[j]), ob = Array.from(B.mz.keys()).sort((i, j) => B.mz[i] - B.mz[j]);
        let j = 0;
        for (const i of oa) {
          while (j < ob.length && B.mz[ob[j]] < A.mz[i] - s.tol) j++;
          for (let k = j; k < ob.length && B.mz[ob[k]] <= A.mz[i] + s.tol; k++) { A.hit[i] = 1; B.hit[ob[k]] = 1; }
        }
      }
    }
    draw() {
      if (!this.w) return;
      const s = this.spec, P = this.pal(), ctx = this.ctx;
      this.prep();
      const two = this.sides.length === 2;
      let x0 = Infinity, x1 = -Infinity;
      for (const r of this.sides) for (let i = 0; i < r.mz.length; i++) { if (r.mz[i] < x0) x0 = r.mz[i]; if (r.mz[i] > x1) x1 = r.mz[i]; }
      if (!isFinite(x0)) { x0 = 50; x1 = 500; }
      const pad = Math.max(5, (x1 - x0) * 0.05);
      this.xd = [Math.max(0, x0 - pad), x1 + pad];
      this.yd = two ? [-112, 112] : [0, 112];
      const yFmt = (v) => P3.fixed(Math.abs(v), 0);
      const yt = { values: two ? [-100, -50, 0, 50, 100] : [0, 25, 50, 75, 100], digits: 0 };
      this.layout(P, yFmt, yt);
      const xt = ticks(this.xd[0], this.xd[1], Math.max(2, Math.floor((this.box.x1 - this.box.x0) / 80)));
      this.clear(P);
      this.axes(P, xt, yt, (v, d) => P3.fixed(v, Math.min(d, 1)), yFmt);
      const b = this.box, zero = Math.round(this.sy(0)) + 0.5;
      ctx.save();
      ctx.beginPath(); ctx.rect(b.x0, b.y0 - 2, b.x1 - b.x0, b.y1 - b.y0 + 2); ctx.clip();
      const cols = [P.s1, P.s2];
      this.sides.forEach((r, si) => {
        const sign = si ? -1 : 1;
        ctx.lineWidth = 1.5;
        for (const matched of two ? [0, 1] : [1]) {      // unmatched first, matched on top
          ctx.strokeStyle = matched ? cols[si] : P.neutral;
          ctx.beginPath();
          for (let i = 0; i < r.mz.length; i++) {
            if (two && r.hit[i] !== matched) continue;
            const x = Math.round(this.sx(r.mz[i])) + 0.5;
            ctx.moveTo(x, zero); ctx.lineTo(x, this.sy(sign * r.rel[i]));
          }
          ctx.stroke();
        }
        // the four strongest peaks carry their m/z
        const top4 = Array.from(r.rel.keys()).sort((i, j) => r.rel[j] - r.rel[i]).slice(0, 4);
        ctx.font = `400 10px ${P.font}`;
        ctx.fillStyle = P.muted;
        ctx.textAlign = "center";
        ctx.textBaseline = si ? "top" : "bottom";
        for (const i of top4) if (r.rel[i] > 5) ctx.fillText(P3.fixed(r.mz[i], 4), this.sx(r.mz[i]), this.sy(sign * r.rel[i]) + (si ? 3 : -3));
      });
      ctx.strokeStyle = P.axis;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(b.x0, zero); ctx.lineTo(b.x1, zero); ctx.stroke();
      if (this.hoverPk) {
        const [si, i] = this.hoverPk, r = this.sides[si];
        const x = Math.round(this.sx(r.mz[i])) + 0.5;
        ctx.strokeStyle = P.ink;
        ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.moveTo(x, zero); ctx.lineTo(x, this.sy((si ? -1 : 1) * r.rel[i])); ctx.stroke();
      }
      ctx.restore();
      ctx.font = `500 11px ${P.font}`;
      ctx.fillStyle = P.ink2;
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      if (s.topLabel) ctx.fillText(s.topLabel, b.x0 + 6, b.y0 + 2);
      if (two && s.bottomLabel) { ctx.textBaseline = "bottom"; ctx.fillText(s.bottomLabel, b.x0 + 6, b.y1 - 2); }
    }
    onMove([px, py]) {
      if (!this.box || !this.sides) return;
      const upper = py <= this.sy(0), si = upper || this.sides.length < 2 ? 0 : 1, r = this.sides[si];
      let best = -1, bd = 7;
      for (let i = 0; i < r.mz.length; i++) { const d = Math.abs(this.sx(r.mz[i]) - px); if (d < bd) { bd = d; best = i; } }
      const prev = this.hoverPk;
      this.hoverPk = best >= 0 ? [si, best] : null;
      if (String(prev) !== String(this.hoverPk)) this.draw();
      if (best < 0) return this.hideTip();
      const name = si ? this.spec.bottomLabel || "library" : this.spec.topLabel || "query";
      this.showTip(this.sx(r.mz[best]), this.sy((si ? -1 : 1) * r.rel[best]),
        `<b>m/z ${P3.fixed(r.mz[best], 4)}</b><br>${P3.fixed(r.rel[best], 1)} % · ${name}${this.sides.length === 2 ? (r.hit[best] ? " · matched" : " · unmatched") : ""}`);
    }
    onLeave() { this.hideTip(); if (this.hoverPk) { this.hoverPk = null; this.draw(); } }
  }

  // ---------------------------------------------------------------- legend
  function legend(items) {
    return h("div.chart-legend", items.map((it) => {
      const col = it.color && it.color.startsWith("--") ? `var(${it.color})` : it.color;
      if (it.kind === "ramp") {
        return h("span.key", it.left || "", h("span.ramp", { style: { background: `linear-gradient(90deg, var(${it.from}), var(${it.to}))` } }), it.right || "");
      }
      if (it.kind === "line") return h("span.key", h("span.line-key" + (it.bold ? ".bold" : ""), { style: { background: col } }), it.label);
      if (it.kind === "star") return h("span.key", h("span.star-key", { style: { color: col } }, P3.icon("star")), it.label);
      if (it.kind === "dash") return h("span.key", h("span.dash-key" + (it.bold ? ".bold" : ""), { style: { borderColor: col } }), it.label);
      if (it.kind === "dot") return h("span.key", h("span.dot-key", { style: { background: col } }), it.label);
      if (it.kind === "text") return h("span.key", it.label);
      return h("span.key", h("span.swatch", { style: { background: col } }), it.label);
    }));
  }

  P3.plot = {
    line: (el, spec) => new Line(el, spec),
    scatter: (el, spec) => new Scatter(el, spec),
    bars: (el, spec) => new Bars(el, spec),
    mirror: (el, spec) => new Mirror(el, spec),
    legend, mix, ticks,
  };
})(window.P3 = window.P3 || {});
