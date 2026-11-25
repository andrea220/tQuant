// tqstudio market data page: pick a snapshot (date, set), an asset class (equity, rates, FX) and an item
// (underlying, curve, pair); fetch its view from the API and draw tables and charts.

const $ = (id) => document.getElementById(id);
const fmt = (x, d = 4) => (x === null || x === undefined ? "" : Number(x).toFixed(d));
const pct = (x, d = 2) => (x === null || x === undefined ? "" : (100 * x).toFixed(d));
const AXIS = { gridcolor: "#eef1f6", zerolinecolor: "#dde2ea", linecolor: "#dde2ea" };
const LAYOUT = { margin: { l: 55, r: 15, t: 38, b: 45 }, font: { size: 12, family: "Segoe UI, system-ui, sans-serif",
  color: "#18202f" }, showlegend: false, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  colorway: ["#1f4fa8", "#e8833a", "#12805c", "#a43fd1", "#c2362f"] };
const CLASSES = {
  equity: { label: "Underlying", noun: "underlyings" },
  rates: { label: "Curve", noun: "curves" },
  fx: { label: "Pair", noun: "pairs" },
};
const state = { snapshots: [], date: null, set: null, catalog: null, cls: "equity", item: null };

// ---------- helpers ----------
function table(columns, rows) {
  const head = "<tr>" + columns.map(([h]) => `<th>${h}</th>`).join("") + "</tr>";
  const body = rows.map((r) => "<tr>" + columns.map(([, f]) => `<td>${f(r)}</td>`).join("") + "</tr>").join("");
  return `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

function plot(id, traces, title, xTitle, yTitle, extra = {}) {
  Plotly.react($(id), traces, { ...LAYOUT, title: { text: title, font: { size: 13 } },
    xaxis: { title: xTitle, ...AXIS }, yaxis: { title: yTitle, ...AXIS }, ...extra }, { displaylogo: false, responsive: true });
}

const cards = (items) => items.map(([l, v], i) =>
  `<div class="card${i === 0 ? " big" : ""}"><div class="label">${l}</div><div class="value">${v}</div></div>`).join("");
const hue = (i, n) => `hsl(${220 - (160 * i) / Math.max(n - 1, 1)}, 65%, 45%)`;
const tenorLabel = (t) => (t < 1 ? `${Math.round(12 * t)}M` : `${+t.toFixed(1)}Y`);
// rate vols: normal in bp, lognormal in %
const volScale = (type) => (type === "normal" ? [1e4, "bp"] : [100, "%"]);

function smiles(id, x, matrix, names, scale, title, xTitle, yTitle) {
  plot(id, matrix.map((row, i) => ({ x, y: row.map((v) => scale * v), mode: "lines+markers", name: names[i],
    marker: { size: 4 }, line: { color: hue(i, matrix.length) } })), title, xTitle, yTitle,
  { showlegend: true, legend: { font: { size: 10 } } });
}

// clipboard: the async API needs a secure context (localhost / https); colleagues on http://<ip> use the fallback
async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const area = document.createElement("textarea");
  area.value = text;
  area.style.position = "fixed";
  area.style.opacity = "0";
  document.body.appendChild(area);
  area.select();
  document.execCommand("copy");
  document.body.removeChild(area);
}

async function copySurface() {
  if (!state.volCopy) return;
  const button = $("eq-vol-copy");
  try {
    await copyText(state.volCopy);
    button.textContent = "Copied";
  } catch (e) {
    button.textContent = "Copy failed";
  }
  setTimeout(() => { button.textContent = "Copy surface"; }, 1500);
}

// row labels of a vol surface: listed expiries as Nov26 (with the day, 20Nov26, when a month has several), the
// extrapolated yearly points (whole years) as 3Y, 4Y, ...
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
function expiryLabels(expiries, T) {
  const month = (e) => `${MONTHS[Number(e.slice(5, 7)) - 1]}${e.slice(2, 4)}`;
  const listed = expiries.filter((e, i) => Math.abs(T[i] - Math.round(T[i])) > 1e-9);
  const count = {};
  listed.forEach((e) => { count[month(e)] = (count[month(e)] || 0) + 1; });
  return expiries.map((e, i) => {
    if (Math.abs(T[i] - Math.round(T[i])) <= 1e-9 && T[i] >= 1) return `${Math.round(T[i])}Y`;
    return count[month(e)] > 1 ? `${Number(e.slice(8, 10))}${month(e)}` : month(e);
  });
}

function volTable(id, expiries, colLabels, matrix, scale, d = 2) {
  const cols = [["Expiry", (r) => r.expiry]].concat(colLabels.map((c, j) => [c, (r) => fmt(scale * r.row[j], d)]));
  $(id).innerHTML = table(cols, expiries.map((e, i) => ({ expiry: e, row: matrix[i] })));
}

// ---------- equity: fit to the listed options ----------
async function openFit() {
  $("eq-fit-panel").classList.remove("hidden");
  document.body.classList.add("modal-open");               // no page scroll behind the popup
  $("eq-fit-title").firstChild.textContent = `${state.item}: fit to the listed options `;
  $("eq-fit-status").textContent = "Repricing the listed options…";
  $("eq-fit-matrix").innerHTML = "";
  $("eq-fit-cards").innerHTML = "";
  try {
    const res = await fetch(`/api/market/equity/${encodeURIComponent(state.item)}/fit?${query()}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    state.fit = data;
    $("eq-fit-status").textContent = `${data.count} options`;
    renderFit();
  } catch (e) {
    $("eq-fit-status").innerHTML = `<span class="error">${e.message}</span>`;
  }
}

function renderFit() {
  const d = state.fit;
  if (!d) return;
  const view = state.fitView || "market";
  const vol = state.fitVol || "surface";
  document.querySelectorAll("#eq-fit-view button").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  document.querySelectorAll("#eq-fit-vol button").forEach((b) => b.classList.toggle("active", b.dataset.vol === vol));
  $("eq-fit-vol").classList.toggle("dim", view === "market" || view === "time");
  const s = d.summary[vol];
  const share = (x) => (x === null ? "–" : `${pct(x, 0)}%`);
  $("eq-fit-cards").innerHTML = cards([
    ["Within bid / ask", `${share(s.within_bid_ask)} <span class="muted">of ${s.two_sided}</span>`],
    ["RMS (% of spot)", fmt(s.rms_pct_spot, 3)], ["Max (% of spot)", fmt(s.max_pct_spot, 3)],
    ["RMS (vol pts)", fmt(s.rms_volpts, 2)], ["Max (vol pts)", fmt(s.max_volpts, 2)],
    [vol === "surface" ? "Model Valuation: within" : "Surface Valuation: within",
      share(d.summary[vol === "surface" ? "smile" : "surface"].within_bid_ask)],
  ]);
  const expiries = [...new Set(d.quotes.map((q) => q.expiry))].sort();
  const strikes = [...new Set(d.quotes.map((q) => q.strike))].sort((a, b) => a - b);
  const day = new Date(`${state.date}T00:00:00`);
  const T = expiries.map((e) => (new Date(`${e}T00:00:00`) - day) / (365 * 86400000));
  const labels = expiryLabels(expiries, T.map((t) => (Math.abs(t - Math.round(t)) < 1e-9 ? t + 1e-6 : t)));
  const at = {};
  d.quotes.forEach((q) => { at[`${q.expiry}|${q.strike}`] = q; });
  const dp = d.spot < 50 ? 3 : 2;
  const cell = (q) => {
    if (!q) return "<td></td>";
    const price = q[`${vol}_price`];
    const inside = q.bid - 1e-12 <= price && price <= q.ask + 1e-12;
    const cls = q.settlement ? "settle" : inside ? "inside" : "outside";
    const tip = `${q.option_type} ${q.strike} ${q.expiry} · bid ${fmt(q.bid, dp)} ask ${fmt(q.ask, dp)} · `
      + `market vol ${fmt(100 * q.vol_mid, 2)}% · ${vol === "surface" ? "surface" : "model"} vol `
      + `${fmt(100 * q[`${vol}_vol`], 2)}% · BS ${fmt(price, dp)}`;
    let text;
    if (view === "time") {
      const day = q.last_update ? `${Number(q.last_update.slice(8, 10))}-${MONTHS[Number(q.last_update.slice(5, 7)) - 1]}` : "";
      text = q.last_update_time && !q.previous_session ? q.last_update_time.slice(0, 5) : day;
      const tcls = q.settlement ? "settle" : q.previous_session ? "stale" : "";
      return `<td${tcls ? ` class="${tcls}"` : ""} title="${tip} · last bid update ${q.last_update || ""} `
        + `${q.last_update_time || ""}">${text}</td>`;
    }
    if (view === "market") text = fmt(q.mid, dp) + (q.settlement ? " <i>s</i>" : "");
    else if (view === "model") text = fmt(price, dp);
    else if (view === "diff") text = fmt((100 * (price - q.mid)) / d.spot, 3);
    else text = q.vol_mid === null ? "" : fmt(100 * (q[`${vol}_vol`] - q.vol_mid), 2);
    const color = view === "diff" || view === "dvol" ? ` class="${cls}"` : q.settlement ? ' class="settle-text"' : "";
    return `<td${color} title="${tip}">${text}</td>`;
  };
  const head = "<tr><th>Expiry</th>" + strikes.map((k) => `<th>${fmt(k, k < 100 ? 2 : 0)}`
    + `<br><span class="muted">${fmt((100 * k) / d.spot, 1)}%</span></th>`).join("") + "</tr>";
  const body = expiries.map((e, i) => `<tr><td>${labels[i]} <span class="muted">${e}</span></td>`
    + strikes.map((k) => cell(at[`${e}|${k}`])).join("") + "</tr>").join("");
  const timeNote = view !== "time" ? ""
    : `<p class="muted" style="margin: 0 0 8px">Quote date ${d.quote_date || ""}; ${d.previous_session} quotes of an `
      + "earlier session (orange). " + (d.has_times ? "Times in the terminal time zone."
        : "This snapshot has no update times (kept from the next download on): dates only.") + "</p>";
  $("eq-fit-matrix").innerHTML = `${timeNote}<table class="fit-matrix"><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

// ---------- equity ----------
function renderEquity(d) {
  $("eq-fit-panel").classList.add("hidden");               // the fit view belongs to the previous underlying
  document.body.classList.remove("modal-open");
  state.fit = null;
  const s = d.summary;
  const items = [["Spot", fmt(s.spot, 2)], ["Currency", s.currency], ["Repo 1Y", pct(s.repo, 3) + "%"],
    ["Dividends", s.dividend_model]];
  if (s.div_yield !== null) items.splice(4, 0, ["Div yield", pct(s.div_yield, 3) + "%"]);
  $("eq-summary").innerHTML = cards(items);
  calibratedLink(state.item);

  $("eq-forwards").innerHTML = table([
    ["Tenor", (r) => r.tenor], ["Forward", (r) => fmt(r.forward, 2)], ["Fwd / spot", (r) => fmt(r.forward_over_spot)],
    ["PV divs", (r) => fmt(r.pv_dividends, 2)], ["Impl. div yield %", (r) => pct(r.implied_div_yield)],
    ["DF", (r) => fmt(r.discount_factor)]], d.forwards);
  $("eq-dividends").innerHTML = d.dividends.length
    ? table([["Ex date", (r) => r.ex_date], ["Pay date", (r) => r.pay_date || ""], ["Amount", (r) => fmt(r.amount)],
        ["% spot", (r) => fmt(r.pct_spot, 2)]], d.dividends)
    : `<p class="muted">No discrete dividends (dividend model: ${s.dividend_model}).</p>`;
  const fc = d.forward_curve;
  plot("eq-chart-forward", [{ x: fc.T, y: fc.forward, mode: "lines" },
    { x: [0, fc.T[fc.T.length - 1]], y: [s.spot, s.spot], mode: "lines", line: { dash: "dash", color: "#888" } }],
  "Forward", "T (years)", "");
  plot("eq-chart-divyield", [{ x: fc.T, y: fc.implied_div_yield.map((v) => 100 * v), mode: "lines" }],
    "Implied dividend yield", "T (years)", "%");
  const r = d.repo;
  if (r) {
    plot("eq-chart-repo", [{ x: r.grid.T, y: r.grid.rate.map((v) => 100 * v), mode: "lines", name: "zero repo" },
      { x: r.pillars.map((p) => p.T), y: r.pillars.map((p) => 100 * p.rate), mode: "markers", name: "implied",
        text: r.pillars.map((p) => p.date), marker: { size: 6 } }],
    "Implied repo (zero rate to T)", "T (years)", "%");
  } else {
    Plotly.purge($("eq-chart-repo"));
    $("eq-chart-repo").innerHTML = `<p class="muted">Flat repo (${pct(s.repo, 3)}%): no term structure in this snapshot.</p>`;
  }

  const v = d.vol;
  if (v) {
    // header: moneyness and, below, the strike at the current spot (decimals by the spot's size)
    const kd = s.spot < 50 ? 2 : s.spot < 1000 ? 1 : 0;
    const rows = expiryLabels(v.expiries, v.T);       // label (Nov26, 3Y) and, beside it, the expiry date
    // copy as tab-separated text (Excel): moneyness, strikes at the current spot, then expiry, label, vols in %
    state.volCopy = [["Expiry", "Label", ...v.moneyness.map((m) => `${+(100 * m).toFixed(2)}%`)],
      ["Strike", "", ...v.moneyness.map((m) => (m * s.spot).toFixed(kd))],
      ...v.expiries.map((e, i) => [e, rows[i], ...v.matrix[i].map((x) => (100 * x).toFixed(4))])]
      .map((r) => r.join("\t")).join("\n");
    $("eq-vol-copy").disabled = false;
    // the labels in a fixed-width box so that the dates line up in one column
    volTable("eq-vol", rows.map((r, i) => `<span style="white-space: nowrap"><span style="display: inline-block; `
      + `min-width: 4.6em">${r}</span><span class="muted">${v.expiries[i]}</span></span>`), v.moneyness.map((m) => `${(100 * m).toFixed(1).replace(/\.0$/, "")}%`
      + `<br><span class="muted">${(m * s.spot).toFixed(kd)}</span>`), v.matrix, 100);
    smiles("eq-chart-smiles", v.moneyness.map((m) => 100 * m), v.matrix, rows, 100, "Smiles",
      "strike / spot (%)", "implied vol (%)");
    plot("eq-chart-atm", [{ x: v.T, y: v.atm_forward.map((x) => 100 * x), mode: "lines+markers" }],
      "ATM-forward term structure", "T (years)", "implied vol (%)");
  } else {
    state.volCopy = null;
    $("eq-vol-copy").disabled = true;
    $("eq-vol").innerHTML = '<p class="muted">No vol surface.</p>';
  }

  state.smile = d.smile_quotes;
  const sel = $("eq-smile-expiry");
  if (state.smile) {
    const labels = expiryLabels(state.smile.expiries.map((e) => e.expiry), state.smile.expiries.map((e) => e.t));
    sel.innerHTML = state.smile.expiries.map((e, i) => `<option value="${i}">${labels[i]} (${e.expiry})</option>`).join("");
    sel.value = String(state.smile.default);
    sel.disabled = false;
  } else {
    sel.innerHTML = "<option>no option quotes</option>";
    sel.disabled = true;
  }
  marketSmile();
  state.corr = d.correlations;
  const c = state.corr;
  const own = `EQ:${s.currency}:${state.item}`;                  // this page's underlying, picked at first
  state.corrPicked = new Set(c && c.names.includes(own) ? [own] : []);
  $("corr-pick").classList.toggle("hidden", !c);
  if (c) {
    $("corr-options").innerHTML = '<div class="actions"><button class="link" id="corr-all">All</button>'
      + '<button class="link" id="corr-none">None</button></div>'
      + c.names.map((n, i) => `<label><input type="checkbox" value="${n}"${state.corrPicked.has(n) ? " checked" : ""}>`
        + `${c.labels[i]} `
        + `<span class="muted">${n.split(":").slice(0, 2).join(":")}</span></label>`).join("");
    $("corr-options").querySelectorAll("input").forEach((box) => box.addEventListener("change", () => {
      box.checked ? state.corrPicked.add(box.value) : state.corrPicked.delete(box.value);
      correlationMatrix();
    }));
    const setAll = (on) => {
      $("corr-options").querySelectorAll("input").forEach((box) => { box.checked = on; });
      state.corrPicked = new Set(on ? c.names : []);
      correlationMatrix();
    };
    $("corr-all").addEventListener("click", (e) => { e.preventDefault(); setAll(true); });
    $("corr-none").addEventListener("click", (e) => { e.preventDefault(); setAll(false); });
  }
  correlationMatrix();
}

// sub-matrix of the picked factors, shaded: blue positive, red negative
function correlationMatrix() {
  const c = state.corr;
  if (!c) {
    $("eq-correlations").innerHTML = '<p class="muted">No correlation matrix in this snapshot.</p>';
    return;
  }
  const idx = c.names.map((n, i) => i).filter((i) => state.corrPicked.has(c.names[i]));
  $("corr-summary").textContent = idx.length ? `${idx.length} picked: ${idx.map((i) => c.labels[i]).join(", ")}`
    : "Pick underlyings";
  if (idx.length === 0) {
    $("eq-correlations").innerHTML = '<p class="muted">Pick the underlyings to see their correlation matrix.</p>';
    return;
  }
  const shade = (x) => (x >= 0 ? `rgba(53, 99, 233, ${0.08 + 0.5 * x})` : `rgba(194, 54, 47, ${0.08 + 0.5 * -x})`);
  const head = "<tr><th></th>" + idx.map((j) => `<th>${c.labels[j]}</th>`).join("") + "</tr>";
  const body = idx.map((i) => `<tr><td>${c.labels[i]}</td>` + idx.map((j) => {
    const x = c.matrix[i][j];
    return `<td style="background: ${shade(x)}">${x.toFixed(2)}</td>`;
  }).join("") + "</tr>").join("");
  $("eq-correlations").innerHTML = `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

// calibrated smile of one expiry against the quotes: bid / ask bars, mid points, model and stored surface
function marketSmile() {
  const s = state.smile;
  if (!s) {
    Plotly.purge($("eq-chart-smile-market"));
    $("eq-chart-smile-market").innerHTML = '<p class="muted">No option quotes in this snapshot.</p>';
    $("eq-smile-summary").innerHTML = "";
    return;
  }
  const e = s.expiries[Number($("eq-smile-expiry").value)];
  const bars = (pts, name, marker) => {
    const mid = pts.map((p) => 100 * p.mid);
    return { x: pts.map((p) => 100 * p.moneyness), y: mid, mode: "markers", name, marker,
      text: pts.map((p) => `${p.type} ${p.strike}`),
      error_y: { type: "data", symmetric: false, thickness: 1.2, width: 3,
        array: pts.map((p) => (p.ask === null ? 0 : 100 * (p.ask - p.mid))),
        arrayminus: pts.map((p) => (p.bid === null ? 0 : 100 * (p.mid - p.bid))) } };
  };
  const used = e.points.filter((p) => p.used && !p.settlement);
  const settle = e.points.filter((p) => p.used && p.settlement);
  const dropped = e.points.filter((p) => !p.used && p.mid !== null);
  const x = e.line.moneyness.map((m) => 100 * m);
  const traces = [
    { x, y: e.line.model.map((v) => (v === null ? null : 100 * v)), mode: "lines", name: "model (calibrated)",
      line: { color: "#1f4fa8", width: 2 } },
  ];
  if (e.line.surface) traces.push({ x, y: e.line.surface.map((v) => 100 * v), mode: "lines", name: "stored surface",
    line: { color: "#888", dash: "dash", width: 1.5 } });
  traces.push(bars(used, "bid / mid / ask", { color: "#18202f", size: 5 }));
  if (settle.length) traces.push(bars(settle, "settlement", { color: "#e8833a", size: 7, symbol: "diamond" }));
  if (dropped.length) traces.push(bars(dropped, "filtered out", { color: "#bbb", size: 5 }));
  plot("eq-chart-smile-market", traces, `Smile ${e.expiry} vs market`, "strike / spot (%)", "implied vol (%)",
    { showlegend: true, legend: { font: { size: 10 } } });
  $("eq-smile-summary").innerHTML = `<div class="cards">${cards([
    ["Expiry", `${e.expiry} (${e.t.toFixed(2)}y)`], ["Forward", fmt(e.forward, 2)], ["Repo", pct(e.repo, 3) + "%"],
    ["Quotes used", e.quotes], ["Fit rms (vol pts)", fmt(100 * e.fit_rms, 2)],
    ["Model inside bid-ask", e.inside === null ? "" : `${Math.round(100 * e.inside)}%`], ["Source", e.source]])}</div>`;
}

// link to the latest calibration set with this underlying (Calibration tab)
async function calibratedLink(ticker) {
  try {
    const data = await (await fetch("/api/calibration/sets")).json();
    const set = (data.sets || []).find((x) => (x.tickers || []).includes(ticker));
    if (!set || state.item !== ticker) return;
    const names = { Heston: "Heston", DisplacedDiffusion: "DD", HestonSLV: "SLV" };
    const models = [...set.models, ...(set.slv.includes(ticker) ? ["HestonSLV"] : [])].map((m) => names[m] || m).join(", ");
    $("eq-summary").insertAdjacentHTML("beforeend", `<div class="card"><div class="label">Calibrated</div><div class="value">`
      + `<a href="/calibration?date=${set.date}&set=${set.set}&ticker=${encodeURIComponent(ticker)}">${models} · set `
      + `${set.date} ${set.set}</a></div></div>`);
  } catch (e) { /* no calibration sets: nothing to show */ }
}

// ---------- rates ----------
let swaption = null;

function renderRates(d) {
  const s = d.summary;
  $("ir-summary").innerHTML = cards([["Curve", `${s.currency} ${s.curve}`], ["Pillars", s.pillars],
    ["Last pillar", s.last_pillar], ["Day count", s.daycount], ["Interpolation", s.interpolation], ["Key", s.key]]);
  const quotes = d.quotes || [];
  $("ir-quotes-panel").classList.toggle("hidden", !quotes.length);
  if (quotes.length) {
    const used = quotes.filter((q) => q.used).length;
    $("ir-quotes-note").textContent = `${quotes.length} quotes, ${used} used in the bootstrap (one pillar per end date)`;
    $("ir-quotes").innerHTML = table([
      ["Tenor", (q) => q.tenor],
      ["Instrument", (q) => ({ fixing: "Fixing", ois: "OIS", fra: "FRA", swap: "Swap" })[q.instrument] || q.instrument],
      ["Quote %", (q) => pct(q.quote, 4)],
      ["Bloomberg ticker", (q) => q.bloomberg || ""],
      ["Last update", (q) => q.last_update || ""],
      ["Pillar date", (q) => q.end_date || ""],
      ["DV01 (1 bp, 1M notional)", (q) => (q.quote_dv01 === null ? "" : fmt(q.quote_dv01 * 100, 0))],
      ["Used", (q) => (q.used ? "yes" : '<span class="muted">no</span>')],
    ], quotes);
  }
  $("ir-tenors").innerHTML = table([["Tenor", (r) => r.tenor], ["Zero %", (r) => pct(r.zero_rate, 4)],
    ["DF", (r) => fmt(r.discount_factor, 6)], ["3M fwd %", (r) => pct(r.forward_3m, 4)]], d.tenors);
  $("ir-pillars").innerHTML = table([["Date", (r) => r.date], ["T", (r) => fmt(r.T)],
    ["Zero %", (r) => pct(r.zero_rate, 4)], ["DF", (r) => fmt(r.discount_factor, 6)]], d.pillars);
  plot("ir-chart-curve", [
    { x: d.grid.T, y: d.grid.zero_rate.map((v) => 100 * v), mode: "lines", name: "zero rate" },
    { x: d.grid.T, y: d.grid.inst_forward.map((v) => 100 * v), mode: "lines", name: "inst. forward" },
    { x: d.pillars.map((p) => p.T), y: d.pillars.map((p) => 100 * p.zero_rate), mode: "markers", name: "pillars",
      marker: { size: 5, color: "#1f4fa8" } }],
  "Zero and instantaneous forward", "T (years)", "%", { showlegend: true, legend: { orientation: "h", y: -0.2 } });

  const cv = d.caplet_vol;
  $("ir-caplet-panel").classList.toggle("hidden", !cv);
  if (cv) {
    const [k, unit] = volScale(cv.vol_type);
    $("ir-caplet-unit").textContent = `(${cv.vol_type}, ${unit})`;
    volTable("ir-caplet", cv.expiries, cv.strikes.map((x) => pct(x, 2) + "%"), cv.matrix, k, cv.vol_type === "normal" ? 1 : 2);
    smiles("ir-chart-caplet", cv.strikes.map((x) => 100 * x), cv.matrix, cv.expiries, k, "Caplet smiles",
      "strike (%)", `vol (${unit})`);
  }

  swaption = d.swaption_vol;
  $("ir-swaption-panel").classList.toggle("hidden", !swaption);
  if (swaption) {
    const [k, unit] = volScale(swaption.vol_type);
    $("ir-swaption-unit").textContent = `(${swaption.vol_type}, ${unit})`;
    const atm = swaption.strike_offsets.indexOf(0);
    const atmGrid = swaption.cube.map((byTenor) => byTenor.map((row) => row[atm]));
    volTable("ir-swaption-atm", swaption.expiries.map(tenorLabel), swaption.tenors.map(tenorLabel), atmGrid, k,
      swaption.vol_type === "normal" ? 1 : 2);
    Plotly.react($("ir-chart-swaption-atm"), [{ type: "heatmap", z: atmGrid.map((r) => r.map((v) => k * v)),
      x: swaption.tenors.map(tenorLabel), y: swaption.expiries.map(tenorLabel), colorscale: "Blues",
      colorbar: { title: { text: unit } } }], { ...LAYOUT, title: { text: "ATM swaption vols", font: { size: 13 } },
      xaxis: { title: "tenor" }, yaxis: { title: "expiry", autorange: "reversed" } }, { displaylogo: false, responsive: true });
    $("ir-swaption-tenor").innerHTML = swaption.tenors.map((t, j) => `<option value="${j}">${tenorLabel(t)}</option>`).join("");
    swaptionSmile();
  }
}

function swaptionSmile() {
  const j = Number($("ir-swaption-tenor").value || 0);
  const [k, unit] = volScale(swaption.vol_type);
  smiles("ir-chart-swaption-smile", swaption.strike_offsets.map((x) => 1e4 * x),
    swaption.cube.map((byTenor) => byTenor[j]), swaption.expiries.map((t) => `${tenorLabel(t)} expiry`), k,
    `Smiles, ${tenorLabel(swaption.tenors[j])} tenor`, "strike offset from ATM (bp)", `vol (${unit})`);
}

// ---------- fx ----------
function renderFx(d) {
  const s = d.summary;
  $("fx-summary").innerHTML = cards([["Spot", fmt(s.spot, 4)], ["Pair", `${s.base}/${s.quote}`],
    [`${s.base} curve`, s.base_curve], [`${s.quote} curve`, s.quote_curve]]);
  $("fx-forwards").innerHTML = table([["Tenor", (r) => r.tenor], ["Forward", (r) => fmt(r.forward, 5)],
    ["Points", (r) => fmt(r.points, 1)], [`DF ${s.base}`, (r) => fmt(r.df_base, 5)],
    [`DF ${s.quote}`, (r) => fmt(r.df_quote, 5)]], d.forwards);
  const fc = d.forward_curve;
  if (!fc) {                        // a currency without its curve: spot only
    $("fx-forwards").innerHTML = `<p class="muted">No forwards: the ${s.base_curve === "missing" ? s.base : s.quote} curve is not in this snapshot.</p>`;
    Plotly.purge($("fx-chart-forward"));
  } else {
    const fc = d.forward_curve;
    plot("fx-chart-forward", [{ x: fc.T, y: fc.forward, mode: "lines" },
      { x: [0, fc.T[fc.T.length - 1]], y: [s.spot, s.spot], mode: "lines", line: { dash: "dash", color: "#888" } }],
    `${s.base}${s.quote} forward`, "T (years)", "");
  }
  const v = d.vol;
  $("fx-vol-panel").classList.toggle("hidden", !v);
  if (v) {
    volTable("fx-vol", v.expiries, v.moneyness.map((m) => `${(100 * m).toFixed(0)}%`), v.matrix, 100);
    smiles("fx-chart-smiles", v.moneyness.map((m) => 100 * m), v.matrix, v.expiries, 100, "Smiles",
      "strike / spot (%)", "implied vol (%)");
    plot("fx-chart-atm", [{ x: v.T, y: v.atm_forward.map((x) => 100 * x), mode: "lines+markers" }],
      "ATM-forward term structure", "T (years)", "implied vol (%)");
  }
}

// ---------- navigation ----------
const query = () => `date=${state.date}&set=${state.set}`;
const items = () => {
  const c = state.catalog;
  if (state.cls === "equity") return c.equity.map((t) => [t, t]);
  if (state.cls === "rates") return c.rates.map((r) => [r.id, `${r.ccy} ${r.name}`]);
  return c.fx.map((p) => [p.id, `${p.base}/${p.quote}`]);
};

function saveUrl() {
  history.replaceState(null, "", `?class=${state.cls}&item=${encodeURIComponent(state.item)}&${query()}`);
}

async function loadItem() {
  const id = state.item;
  const url = state.cls === "equity" ? `/api/market/equity/${encodeURIComponent(id)}`
    : state.cls === "rates" ? `/api/market/rates/${id.replace(":", "/")}` : `/api/market/fx/${id}`;
  $("status").textContent = "Loading…";
  try {
    const res = await fetch(`${url}?${query()}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    for (const c of Object.keys(CLASSES)) $(`view-${c}`).classList.toggle("hidden", c !== state.cls);
    ({ equity: renderEquity, rates: renderRates, fx: renderFx })[state.cls](data);
    $("status").textContent = "";
    saveUrl();
  } catch (e) {
    $("status").innerHTML = `<span class="error">Error: ${e.message}</span>`;
  }
}

function selectClass(cls, wanted) {
  state.cls = cls;
  document.querySelectorAll(".asset").forEach((b) => b.classList.toggle("active", b.dataset.class === cls));
  $("item-label").textContent = CLASSES[cls].label;
  const list = items();
  $("item").innerHTML = list.map(([id, label]) => `<option value="${id}">${label}</option>`).join("");
  state.item = list.some(([id]) => id === wanted) ? wanted : (list[0] || [null])[0];
  $("item").value = state.item;
  if (state.item) loadItem();
}

async function loadSnapshot(wantedClass, wantedItem) {
  $("status").textContent = "Loading snapshot…";
  const res = await fetch(`/api/catalog?${query()}`);
  state.catalog = await res.json();
  $("snapshot").textContent = `${state.source} · ${state.date} · ${state.set}`;
  for (const c of Object.keys(CLASSES)) {
    const n = state.catalog[c].length;
    $(`count-${c}`).textContent = `${n} ${CLASSES[c].noun}`;
  }
  selectClass(wantedClass || state.cls, wantedItem || state.item);
}

function fillSets(wanted) {
  const snap = state.snapshots.find((s) => s.date === state.date);
  $("snap-set").innerHTML = snap.sets.map((s) => `<option>${s}</option>`).join("");
  state.set = snap.sets.includes(wanted) ? wanted : snap.sets[0];
  $("snap-set").value = state.set;
}

async function init() {
  const p = new URLSearchParams(location.search);
  const snaps = await (await fetch("/api/snapshots")).json();
  state.snapshots = snaps.snapshots;
  state.source = snaps.source;
  $("snap-date").innerHTML = state.snapshots.map((s) => `<option>${s.date}</option>`).join("");
  if (state.snapshots.length) {
    state.date = state.snapshots.some((s) => s.date === p.get("date")) ? p.get("date") : state.snapshots[0].date;
    $("snap-date").value = state.date;
    fillSets(p.get("set"));
  }

  $("snap-date").addEventListener("change", () => { state.date = $("snap-date").value; fillSets(); loadSnapshot(); });
  $("snap-set").addEventListener("change", () => { state.set = $("snap-set").value; loadSnapshot(); });
  document.querySelectorAll(".asset").forEach((b) => b.addEventListener("click", () => selectClass(b.dataset.class)));
  $("item").addEventListener("change", () => { state.item = $("item").value; loadItem(); });
  $("ir-swaption-tenor").addEventListener("change", swaptionSmile);
  $("eq-smile-expiry").addEventListener("change", marketSmile);
  $("eq-vol-copy").addEventListener("click", copySurface);
  $("eq-fit-open").addEventListener("click", openFit);
  const closeFit = () => { $("eq-fit-panel").classList.add("hidden"); document.body.classList.remove("modal-open"); };
  $("eq-fit-close").addEventListener("click", closeFit);
  $("eq-fit-panel").addEventListener("click", (e) => { if (e.target === $("eq-fit-panel")) closeFit(); });   // backdrop
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeFit(); });
  document.querySelectorAll("#eq-fit-view button").forEach((b) => b.addEventListener("click", () => {
    state.fitView = b.dataset.view; renderFit(); }));
  document.querySelectorAll("#eq-fit-vol button").forEach((b) => b.addEventListener("click", () => {
    state.fitVol = b.dataset.vol; renderFit(); }));
  if (state.snapshots.length) await loadSnapshot(CLASSES[p.get("class")] ? p.get("class") : "equity", p.get("item"));
  else $("status").innerHTML = 'No snapshot yet: create one from the <a href="/">Home</a> (Universe > Create new snapshot).';
}

init();
