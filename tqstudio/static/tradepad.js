// tqstudio Tradepad: asset class -> product type -> deal form (built from the product's field list, served by
// /api/tradepad/catalog) -> model -> POST /api/price; shows valuation, details, vega map and rho per pillar.

const $ = (id) => document.getElementById(id);
const fmtValue = (v, d) => (v === null || v === undefined ? "–"
  : typeof v === "number" ? v.toLocaleString("en-US", { minimumFractionDigits: d ?? 2, maximumFractionDigits: d ?? 2 })
    : v);
const AXIS = { gridcolor: "#eef1f6", zerolinecolor: "#c9d0db", linecolor: "#dde2ea" };
const FONT = { size: 12, family: "Segoe UI, system-ui, sans-serif", color: "#18202f" };
const state = { catalog: null, cls: null, product: null };

function spec() { return state.catalog.asset_classes[state.cls][state.product]; }

function fieldHtml(f) {
  const id = `f-${f.name}`;
  const input = f.type === "select"
    ? `<select id="${id}">${f.options.map((o) => `<option${o === f.default ? " selected" : ""}>${o}</option>`).join("")}</select>`
    : `<input id="${id}" type="${f.type}" value="${f.default}"${f.step ? ` step="${f.step}"` : ""}>`;
  const wide = f.type === "date" || f.name === "ticker" ? " wide" : "";
  return `<div class="field${wide}"><label for="${id}">${f.label}</label>${input}</div>`;
}

function showModelFields() {
  document.querySelectorAll(".mc").forEach((e) => e.classList.toggle("hidden", $("model").value !== "Local vol"));
}

function selectProduct(product) {
  state.product = product;
  const s = spec();
  $("fields").innerHTML = s.fields.map(fieldHtml).join("");
  $("model").innerHTML = s.models.map((m) => `<option>${m}</option>`).join("");
  showModelFields();
}

function selectClass(cls) {
  state.cls = cls;
  document.querySelectorAll(".segmented button").forEach((b) => b.classList.toggle("active", b.dataset.cls === cls));
  const products = Object.keys(state.catalog.asset_classes[cls]);
  $("product").innerHTML = products.map((p) => `<option>${p}</option>`).join("");
  selectProduct(products[0]);
}

function request() {
  const fields = {};
  for (const f of spec().fields) {
    const v = $(`f-${f.name}`).value;
    fields[f.name] = f.type === "number" ? Number(v) : v;
  }
  return { asset_class: state.cls, product: state.product, fields, model: $("model").value,
    n_paths: Number($("n_paths").value) };
}

const cards = (items, big = 0) => items.map(([l, v, d], i) =>
  `<div class="card${i < big ? " big" : ""}"><div class="label">${l}</div><div class="value">${fmtValue(v, d)}</div></div>`).join("");

function render(d) {
  $("model-info").textContent = `${d.product} · ${d.model}`;
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
    const res = await fetch("/api/price", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request()) });
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
  state.catalog = await (await fetch("/api/tradepad/catalog")).json();
  $("snapshot").textContent = `snapshot ${state.catalog.snapshot}`;
  const classes = Object.keys(state.catalog.asset_classes);
  $("asset-classes").innerHTML = classes.map((c) => `<button data-cls="${c}">${c}</button>`).join("");
  document.querySelectorAll(".segmented button").forEach((b) => b.addEventListener("click", () => selectClass(b.dataset.cls)));
  $("product").addEventListener("change", () => selectProduct($("product").value));
  $("model").addEventListener("change", showModelFields);
  $("price").addEventListener("click", price);
  selectClass(classes[0]);
}

init();
