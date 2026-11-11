// tqstudio Tradepad: asset class -> product type -> underlying / currency -> pricing date -> market inputs (spot and
// vol editable for equity) -> deal fields (from the product's field list, /api/tradepad/catalog) -> model ->
// POST /api/price; shows valuation, details, vega map and rho per pillar.

const $ = (id) => document.getElementById(id);
const fmtValue = (v, d) => (v === null || v === undefined ? "–"
  : typeof v === "number" ? v.toLocaleString("en-US", { minimumFractionDigits: d ?? 2, maximumFractionDigits: d ?? 2 })
    : v);
// numeric inputs are text fields in en-US format ("." decimals, "," thousands) whatever the browser locale
const parseNum = (text) => Number(String(text).replace(/,/g, "").trim());
const fmtInput = (v, d) => (d === undefined
  ? Number(v).toLocaleString("en-US", { maximumFractionDigits: 10 })
  : Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }));
document.addEventListener("focusout", (ev) => {           // re-format a numeric field when leaving it
  const e = ev.target;
  if (e.matches && e.matches("input[data-num]") && Number.isFinite(parseNum(e.value))) {
    e.value = fmtInput(parseNum(e.value), e.dataset.digits === undefined ? undefined : Number(e.dataset.digits));
  }
});
const AXIS = { gridcolor: "#eef1f6", zerolinecolor: "#c9d0db", linecolor: "#dde2ea" };
const FONT = { size: 12, family: "Segoe UI, system-ui, sans-serif", color: "#18202f" };
// market: the snapshot values of the editable inputs; edited: inputs the user changed (kept until reset)
const state = { catalog: null, dates: [], date: null, cls: null, product: null, market: {}, edited: {} };

const spec = () => state.catalog.asset_classes[state.cls][state.product];

// any change of an input clears the results: a price on screen always matches the inputs shown
function invalidate() {
  if (state.tradeReady || !$("trade-panel").classList.contains("hidden")) {
    state.tradeReady = false;
    $("trade-panel").classList.add("hidden");
  }
  updatePriceButton();
  staleValuation();
  if ($("results-section").classList.contains("hidden")) return;
  $("results-section").classList.add("hidden");
  $("empty").classList.remove("hidden");
  $("empty").innerHTML = spec().create_trade ? "Inputs changed: press <b>Build trade</b>, then <b>Price</b>."
    : "Inputs changed: press <b>Price</b>.";
}

// products with a schedule (autocallable) are created first: the trade is shown, then it can be priced
function updatePriceButton() {
  const needsTrade = Boolean(state.catalog && state.product && spec().create_trade);
  $("create").classList.toggle("hidden", !needsTrade);
  $("price").disabled = needsTrade && !state.tradeReady;
}

// the valuation bar in the header: always there, values blanked ("–") as soon as an input changes
function staleValuation() {
  const bar = $("valuation-bar");
  if (bar.classList.contains("stale") || !$("valuation").children.length) return;
  bar.classList.add("stale");
  $("valuation").querySelectorAll(".value").forEach((e) => { e.textContent = "–"; });
  $("model-info").textContent = "Inputs changed";
  $("override-info").classList.add("hidden");
}

function clearValuation() {
  $("valuation-bar").classList.remove("stale");
  $("valuation").innerHTML = "";
  $("model-info").textContent = "No valuation yet";
  $("override-info").classList.add("hidden");
}

const pct = (x, d = 2) => (x === null || x === undefined ? "" : `${Number(x).toFixed(d)}%`);

async function createTrade() {
  $("status").textContent = "Building the trade…";
  try {
    const res = await fetch("/api/tradepad/trade", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(dealRequest()) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    $("trade-summary").innerHTML = data.summary.map(([l, v, d]) =>
      `<div><span class="muted">${l}</span><b>${fmtValue(v, d)}</b></div>`).join("");
    const rows = data.schedule || [];
    $("trade-schedule").innerHTML = rows.length ? `<table><thead><tr><th>Fixing Date</th><th>Autocall Date</th>
      <th>Payment Date</th><th>Coupon Barrier</th><th>Coupon Amount</th><th>Autocall Barrier</th></tr></thead><tbody>${
      rows.map((r) => `<tr><td>${r.fixing_date}</td><td>${r.autocall_date || ""}</td><td>${r.payment_date}</td>
        <td>${pct(r.coupon_barrier)}</td><td>${pct(r.coupon_amount)}</td><td>${pct(r.autocall_barrier)}</td></tr>`).join("")
    }</tbody></table>` : "";
    $("trade-panel").classList.remove("hidden");
    state.tradeReady = true;
    updatePriceButton();
    $("status").textContent = "";
  } catch (e) {
    $("status").innerHTML = `<span class="error">Error: ${e.message}</span>`;
  }
}

// ---------- form ----------
function fieldHtml(f, value) {
  const id = `f-${f.name}`;
  const v = value ?? f.default;
  const input = f.type === "select"
    ? `<select id="${id}">${f.options.map((o) => `<option${String(o) === String(v) ? " selected" : ""}>${o}</option>`).join("")}</select>`
    : f.type === "number"
      ? `<input id="${id}" type="text" inputmode="decimal" data-num value="${fmtInput(v)}">`
      : `<input id="${id}" type="${f.type}" value="${v}">`;
  const wide = f.type === "date" || f.name === "ticker" ? " wide" : "";
  return `<div class="field${wide}"><label for="${id}">${f.label}</label>${input}</div>`;
}

function currentValues() {
  const out = {};
  document.querySelectorAll("[id^='f-']").forEach((e) => { out[e.id.slice(2)] = e.value; });
  return out;
}

function renderForm(keep = {}) {
  const s = spec();
  const [head, ...rest] = s.fields;                                   // underlying (or currency) first
  const dateField = `<div class="field wide"><label for="pricing-date">Pricing date</label><select id="pricing-date">${
    state.dates.map((d) => `<option${d === state.date ? " selected" : ""}>${d}</option>`).join("")}</select></div>`;
  $("head-fields").innerHTML = fieldHtml(head, keep[head.name]) + dateField;
  // fields with a group are drawn as sub-panels (in order of first appearance), the others as one grid
  const groups = [];
  for (const f of rest) {
    const g = groups.find((x) => x.name === (f.group || ""));
    if (g) g.fields.push(f); else groups.push({ name: f.group || "", fields: [f] });
  }
  $("fields").innerHTML = groups.map((g) => {
    const body = g.fields.map((f) => fieldHtml(f, keep[f.name])).join("");
    return g.name ? `<div class="subpanel"><div class="subpanel-title">${g.name}</div><div class="form deal-form">${body}</div></div>`
      : `<div class="form deal-form">${body}</div>`;
  }).join("");
  $("model").innerHTML = s.models.map((m) => `<option>${m}</option>`).join("");
  showModelFields();
  state.tradeReady = false;
  $("trade-panel").classList.add("hidden");
  updatePriceButton();
  clearValuation();
  $("pricing-date").addEventListener("change", () => changeDate($("pricing-date").value));
  $(`f-${head.name}`).addEventListener("change", () => { state.edited = {}; refreshMarket(); });
  rest.forEach((f) => $(`f-${f.name}`).addEventListener("change", refreshMarket));
  state.edited = {};
  refreshMarket();
}

function showModelFields() {
  document.querySelectorAll(".mc").forEach((e) => e.classList.toggle("hidden", $("model").value !== "Local vol"));
}

function selectProduct(product) {
  state.product = product;
  renderForm();
}

function selectClass(cls) {
  state.cls = cls;
  document.querySelectorAll(".segmented button").forEach((b) => b.classList.toggle("active", b.dataset.cls === cls));
  const products = Object.keys(state.catalog.asset_classes[cls]);
  $("product").innerHTML = products.map((p) => `<option>${p}</option>`).join("");
  selectProduct(products[0]);
}

async function loadCatalog() {
  state.catalog = await (await fetch(`/api/tradepad/catalog?date=${state.date}`)).json();
  $("snapshot").textContent = `snapshot ${state.catalog.snapshot}`;
}

async function changeDate(day) {
  const keep = currentValues();
  state.date = day;
  await loadCatalog();
  if (!state.catalog.asset_classes[state.cls][state.product]) { selectClass(state.cls); return; }
  renderForm(keep);
}

// ---------- market inputs ----------
const EDITABLE = { spot: { label: "Spot", scale: 1, step: 0.01, digits: 2 },
  vol: { label: "Vol at strike (%)", scale: 100, step: 0.1, digits: 2 } };

function dealRequest() {
  const fields = {};
  for (const f of spec().fields) {
    const v = $(`f-${f.name}`).value;
    fields[f.name] = f.type === "number" ? parseNum(v) : v;
  }
  return { asset_class: state.cls, product: state.product, fields, date: state.date };
}

let marketTimer = null;
function refreshMarket() {
  clearTimeout(marketTimer);
  marketTimer = setTimeout(loadMarket, 200);
}

async function loadMarket() {
  try {
    const res = await fetch("/api/tradepad/market", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(dealRequest()) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    state.market = data.editable;
    $("market-edit").innerHTML = Object.entries(data.editable).map(([k, v]) => {
      const e = EDITABLE[k];
      const shown = state.edited[k] ?? v * e.scale;
      return `<div class="field"><label for="m-${k}">${e.label}</label><input id="m-${k}" type="text" inputmode="decimal" `
        + `data-num data-digits="${e.digits}" value="${fmtInput(shown, e.digits)}" `
        + `class="${state.edited[k] !== undefined ? "edited" : ""}"></div>`;
    }).join("");
    Object.keys(data.editable).forEach((k) => $(`m-${k}`).addEventListener("input", (ev) => {
      state.edited[k] = parseNum(ev.target.value);
      ev.target.classList.add("edited");
      $("reset-market").classList.remove("hidden");
    }));
    $("reset-market").classList.toggle("hidden", !Object.keys(state.edited).length);
    $("market-info").innerHTML = data.info.map(([l, v, d]) =>
      `<div><span class="muted">${l}</span><b>${fmtValue(v, d)}</b></div>`).join("");
    renderDividends(data);
  } catch (e) {
    $("market-info").innerHTML = `<span class="error">${e.message}</span>`;
  }
}

// equity products: the underlying's dividends with ex-date up to the expiry
function renderDividends(data) {
  const show = state.cls === "Equity" && data.dividends !== undefined;
  $("div-panel").classList.toggle("hidden", !show);
  if (!show) return;
  const rows = data.dividends;
  const model = data.dividend_model || {};
  if (!rows.length) {
    $("div-total").textContent = "";
    $("div-table").innerHTML = model.model === "continuous"
      ? `<p class="muted">Continuous dividend yield ${fmtValue(100 * model.yield, 3)}%.</p>`
      : '<p class="muted">No dividends before the expiry.</p>';
    return;
  }
  const total = rows.reduce((a, r) => a + r.amount, 0), pv = rows.reduce((a, r) => a + r.pv, 0);
  $("div-total").textContent = `(${rows.length}: total ${fmtValue(total, 4)}, PV ${fmtValue(pv, 4)})`;
  $("div-table").innerHTML = `<table><thead><tr><th>Ex date</th><th>Pay date</th><th>Amount</th><th>% spot</th>
    <th>PV</th></tr></thead><tbody>${rows.map((r) => `<tr><td>${r.ex_date}</td><td>${r.pay_date}</td>
    <td>${fmtValue(r.amount, 4)}</td><td>${fmtValue(r.pct_spot, 2)}%</td><td>${fmtValue(r.pv, 4)}</td></tr>`).join("")
    }</tbody></table>`;
}

function overrides() {
  const out = {};
  for (const [k, v] of Object.entries(state.edited)) {
    const e = EDITABLE[k];
    if (Math.abs(v - state.market[k] * e.scale) > 1e-9) out[k] = v / e.scale;
  }
  return out;
}

// ---------- pricing ----------
const cards = (items, big = 0) => items.map(([l, v, d], i) =>
  `<div class="card${i < big ? " big" : ""}"><div class="label">${l}</div><div class="value">${fmtValue(v, d)}</div></div>`).join("");

function render(d) {
  $("model-info").textContent = `${d.product} · ${d.model} · ${d.snapshot}`;
  const ov = Object.entries(d.overrides || {});
  $("override-info").classList.toggle("hidden", !ov.length);
  $("override-info").textContent = ov.map(([k, v]) => k === "vol" ? `vol ${(100 * v).toFixed(2)}%` : `spot ${v}`).join(" · ");
  $("valuation-bar").classList.remove("stale");
  $("valuation").innerHTML = d.headline.map(([l, v, dd], i) =>
    `<div class="vb-card${i === 0 ? " main" : ""}"><div class="label">${l}</div><div class="value">${fmtValue(v, dd)}</div></div>`).join("");
  $("info").innerHTML = cards(d.info);
  const sens = d.sensitivities;
  $("sens-panel").classList.toggle("hidden", !sens);
  if (sens) {
    $("sens-qty").textContent = `(quantity ${fmtValue(sens.quantity, 2)})`;
    $("sens-table").innerHTML = `<table><thead><tr><th>Greek</th><th>Unit</th><th>Per unit</th>
      <th>Equivalent (x size)</th></tr></thead><tbody>${sens.rows.map((r) => {
      const d = r.greek === "Gamma" ? 6 : 4;
      return `<tr><td>${r.greek}</td><td>${r.unit}</td><td>${fmtValue(r.per_unit, d)}</td><td><b>${fmtValue(r.equivalent, 0)}</b></td></tr>`;
    }).join("")}</tbody></table>`;
  }

  $("vega-panel").classList.toggle("hidden", !d.vega);
  if (d.vega) {
    const v = d.vega;
    $("vega-title").textContent = v.title;
    const scale = Math.max(...v.z.flat().map(Math.abs), 1e-12);
    const text = v.z.map((row) => row.map((x) => (Math.abs(x) >= 0.01 * scale ? x.toPrecision(3) : "")));
    Plotly.react($("chart-vega"), [{ type: "heatmap", z: v.z, x: v.x, y: v.y, text, texttemplate: "%{text}",
      colorscale: "RdBu", reversescale: true, zmin: -scale, zmax: scale }],
    { margin: { l: 60, r: 20, t: 10, b: 50 }, font: FONT, paper_bgcolor: "rgba(0,0,0,0)",
      xaxis: { title: v.x_title }, yaxis: { title: v.y_title, autorange: "reversed" } },
    { displaylogo: false, responsive: true });
  }

  $("rho-panel").classList.toggle("hidden", !d.rho.length);
  if (d.rho.length) {
    Plotly.react($("chart-rho"), d.rho.map((r) => ({ type: "bar", name: r.curve, x: r.dates, y: r.rho })),
      { margin: { l: 60, r: 20, t: 10, b: 50 }, font: FONT, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
        barmode: "group", showlegend: true, legend: { orientation: "h", y: 1.12 }, xaxis: { title: "pillar", ...AXIS },
        yaxis: { title: "rho (1 bp)", ...AXIS }, colorway: ["#3563e9", "#e8833a", "#12805c"] },
      { displaylogo: false, responsive: true });
    const head = "<tr><th>Curve</th>" + d.rho[0].dates.map((x) => `<th>${x}</th>`).join("") + "<th>Total</th></tr>";
    const body = d.rho.map((r) => `<tr><td>${r.curve}</td>` + r.rho.map((x) => `<td>${fmtValue(x, 2)}</td>`).join("")
      + `<td><b>${fmtValue(r.total, 2)}</b></td></tr>`).join("");
    $("rho-table").innerHTML = `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
  }
  $("results-section").classList.remove("hidden");
  $("empty").classList.add("hidden");
}

async function price() {
  $("price").disabled = true;
  $("status").textContent = $("model").value === "Local vol" ? "Pricing… (first use calibrates the local vol)" : "Pricing…";
  try {
    const body = { ...dealRequest(), model: $("model").value, n_paths: Number($("n_paths").value), overrides: overrides() };
    const res = await fetch("/api/price", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    render(data);
    $("status").textContent = "";
  } catch (e) {
    $("status").innerHTML = `<span class="error">Error: ${e.message}</span>`;
  } finally {
    $("price").disabled = false;
    updatePriceButton();
  }
}

async function init() {
  state.dates = (await (await fetch("/api/snapshots")).json()).snapshots.map((s) => s.date);
  state.date = state.dates[0];
  await loadCatalog();
  const classes = Object.keys(state.catalog.asset_classes);
  $("asset-classes").innerHTML = classes.map((c) => `<button data-cls="${c}">${c}</button>`).join("");
  document.querySelectorAll(".segmented button").forEach((b) => b.addEventListener("click", () => selectClass(b.dataset.cls)));
  $("product").addEventListener("change", () => selectProduct($("product").value));
  $("model").addEventListener("change", showModelFields);
  $("price").addEventListener("click", price);
  $("create").addEventListener("click", createTrade);
  $("reset-market").addEventListener("click", () => { state.edited = {}; invalidate(); loadMarket(); });
  for (const id of ["inputs-left", "deal-panel"]) {
    $(id).addEventListener("input", invalidate);
    $(id).addEventListener("change", invalidate);
    $(id).addEventListener("click", (ev) => { if (ev.target.closest(".segmented button")) invalidate(); });
  }
  selectClass(classes[0]);
}

init();
