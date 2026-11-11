// tqstudio Tradepad: asset class -> product type -> underlying / currency -> pricing date -> market inputs (spot and
// vol editable for equity) -> deal fields (from the product's field list, /api/tradepad/catalog) -> model ->
// POST /api/price; shows valuation, details, vega map and rho per pillar.

const $ = (id) => document.getElementById(id);
const fmtValue = (v, d) => (v === null || v === undefined ? "–"
  : typeof v === "number" ? v.toLocaleString("en-US", { minimumFractionDigits: d ?? 2, maximumFractionDigits: d ?? 2 })
    : v);
const AXIS = { gridcolor: "#eef1f6", zerolinecolor: "#c9d0db", linecolor: "#dde2ea" };
const FONT = { size: 12, family: "Segoe UI, system-ui, sans-serif", color: "#18202f" };
// market: the snapshot values of the editable inputs; edited: inputs the user changed (kept until reset)
const state = { catalog: null, dates: [], date: null, cls: null, product: null, market: {}, edited: {} };

const spec = () => state.catalog.asset_classes[state.cls][state.product];

// any change of an input clears the results: a price on screen always matches the inputs shown
function invalidate() {
  if ($("results-section").classList.contains("hidden")) return;
  $("results-section").classList.add("hidden");
  $("empty").classList.remove("hidden");
  $("empty").innerHTML = "Inputs changed: press <b>Price</b>.";
}

// ---------- form ----------
function fieldHtml(f, value) {
  const id = `f-${f.name}`;
  const v = value ?? f.default;
  const input = f.type === "select"
    ? `<select id="${id}">${f.options.map((o) => `<option${String(o) === String(v) ? " selected" : ""}>${o}</option>`).join("")}</select>`
    : `<input id="${id}" type="${f.type}" value="${v}"${f.step ? ` step="${f.step}"` : ""}>`;
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
  $("fields").innerHTML = rest.map((f) => fieldHtml(f, keep[f.name])).join("");
  $("model").innerHTML = s.models.map((m) => `<option>${m}</option>`).join("");
  showModelFields();
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
    fields[f.name] = f.type === "number" ? Number(v) : v;
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
      return `<div class="field"><label for="m-${k}">${e.label}</label><input id="m-${k}" type="number" step="${e.step}" `
        + `value="${Number(shown).toFixed(e.digits)}" class="${state.edited[k] !== undefined ? "edited" : ""}"></div>`;
    }).join("");
    Object.keys(data.editable).forEach((k) => $(`m-${k}`).addEventListener("input", (ev) => {
      state.edited[k] = Number(ev.target.value);
      ev.target.classList.add("edited");
      $("reset-market").classList.remove("hidden");
    }));
    $("reset-market").classList.toggle("hidden", !Object.keys(state.edited).length);
    $("market-info").innerHTML = data.info.map(([l, v, d]) =>
      `<div><span class="muted">${l}</span><b>${fmtValue(v, d)}</b></div>`).join("");
  } catch (e) {
    $("market-info").innerHTML = `<span class="error">${e.message}</span>`;
  }
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
  $("valuation").innerHTML = cards(d.headline, 2);
  $("info").innerHTML = cards(d.info);

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
  $("reset-market").addEventListener("click", () => { state.edited = {}; invalidate(); loadMarket(); });
  for (const id of ["inputs-left", "deal-panel"]) {
    $(id).addEventListener("input", invalidate);
    $(id).addEventListener("change", invalidate);
    $(id).addEventListener("click", (ev) => { if (ev.target.closest(".segmented button")) invalidate(); });
  }
  selectClass(classes[0]);
}

init();
