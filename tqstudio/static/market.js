// tqstudio market data page: pick a snapshot (date, set), an asset class (equity, rates, FX) and an item
// (underlying, curve, pair); fetch its view from the API and draw tables and charts.

const $ = (id) => document.getElementById(id);
const fmt = (x, d = 4) => (x === null || x === undefined ? "" : Number(x).toFixed(d));
const pct = (x, d = 2) => (x === null || x === undefined ? "" : (100 * x).toFixed(d));
const AXIS = { gridcolor: "#eef1f6", zerolinecolor: "#dde2ea", linecolor: "#dde2ea" };
const LAYOUT = { margin: { l: 55, r: 15, t: 38, b: 45 }, font: { size: 12, family: "Segoe UI, system-ui, sans-serif",
  color: "#18202f" }, showlegend: false, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  colorway: ["#3563e9", "#e8833a", "#12805c", "#a43fd1", "#c2362f"] };
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

function volTable(id, expiries, colLabels, matrix, scale, d = 2) {
  const cols = [["Expiry", (r) => r.expiry]].concat(colLabels.map((c, j) => [c, (r) => fmt(scale * r.row[j], d)]));
  $(id).innerHTML = table(cols, expiries.map((e, i) => ({ expiry: e, row: matrix[i] })));
}

// ---------- equity ----------
function renderEquity(d) {
  const s = d.summary;
  const items = [["Spot", fmt(s.spot, 2)], ["Currency", s.currency], ["Repo", pct(s.repo, 3) + "%"],
    ["Dividends", s.dividend_model], ["Discount curve", s.discount_curve]];
  if (s.div_yield !== null) items.splice(4, 0, ["Div yield", pct(s.div_yield, 3) + "%"]);
  $("eq-summary").innerHTML = cards(items);

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

  const v = d.vol;
  if (v) {
    volTable("eq-vol", v.expiries, v.moneyness.map((m) => `${(100 * m).toFixed(0)}%`), v.matrix, 100);
    smiles("eq-chart-smiles", v.moneyness.map((m) => 100 * m), v.matrix, v.expiries, 100, "Smiles",
      "strike / spot (%)", "implied vol (%)");
    plot("eq-chart-atm", [{ x: v.T, y: v.atm_forward.map((x) => 100 * x), mode: "lines+markers" }],
      "ATM-forward term structure", "T (years)", "implied vol (%)");
  } else {
    $("eq-vol").innerHTML = '<p class="muted">No vol surface.</p>';
  }

  const c = d.curve;
  $("eq-curve-name").textContent = c.name;
  $("eq-curve").innerHTML = table([["Date", (r) => r.date], ["T", (r) => fmt(r.T)],
    ["Zero rate %", (r) => pct(r.zero_rate, 4)]], c.pillars);
  plot("eq-chart-curve", [{ x: c.grid.T, y: c.grid.zero_rate.map((v) => 100 * v), mode: "lines" },
    { x: c.pillars.map((p) => p.T), y: c.pillars.map((p) => 100 * p.zero_rate), mode: "markers", marker: { size: 5 } }],
  "Zero curve", "T (years)", "%");
  $("eq-correlations").innerHTML = d.correlations.length
    ? table([["Factor", (r) => r.factor], ["Correlation", (r) => fmt(r.correlation)]], d.correlations)
    : '<p class="muted">Not in the correlation matrix.</p>';
}

// ---------- rates ----------
let swaption = null;

function renderRates(d) {
  const s = d.summary;
  $("ir-summary").innerHTML = cards([["Curve", `${s.currency} ${s.curve}`], ["Pillars", s.pillars],
    ["Last pillar", s.last_pillar], ["Day count", s.daycount], ["Key", s.key]]);
  $("ir-tenors").innerHTML = table([["Tenor", (r) => r.tenor], ["Zero %", (r) => pct(r.zero_rate, 4)],
    ["DF", (r) => fmt(r.discount_factor, 6)], ["3M fwd %", (r) => pct(r.forward_3m, 4)]], d.tenors);
  $("ir-pillars").innerHTML = table([["Date", (r) => r.date], ["T", (r) => fmt(r.T)],
    ["Zero %", (r) => pct(r.zero_rate, 4)], ["DF", (r) => fmt(r.discount_factor, 6)]], d.pillars);
  plot("ir-chart-curve", [
    { x: d.grid.T, y: d.grid.zero_rate.map((v) => 100 * v), mode: "lines", name: "zero rate" },
    { x: d.grid.T, y: d.grid.inst_forward.map((v) => 100 * v), mode: "lines", name: "inst. forward" },
    { x: d.pillars.map((p) => p.T), y: d.pillars.map((p) => 100 * p.zero_rate), mode: "markers", name: "pillars",
      marker: { size: 5, color: "#3563e9" } }],
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
  state.date = state.snapshots.some((s) => s.date === p.get("date")) ? p.get("date") : state.snapshots[0].date;
  $("snap-date").value = state.date;
  fillSets(p.get("set"));

  $("snap-date").addEventListener("change", () => { state.date = $("snap-date").value; fillSets(); loadSnapshot(); });
  $("snap-set").addEventListener("change", () => { state.set = $("snap-set").value; loadSnapshot(); });
  document.querySelectorAll(".asset").forEach((b) => b.addEventListener("click", () => selectClass(b.dataset.class)));
  $("item").addEventListener("change", () => { state.item = $("item").value; loadItem(); });
  $("ir-swaption-tenor").addEventListener("change", swaptionSmile);
  await loadSnapshot(CLASSES[p.get("class")] ? p.get("class") : "equity", p.get("item"));
}

init();
