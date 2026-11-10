// tqstudio Tradepad: define a trade, POST /api/price with the chosen models, show prices, greeks and vega maps.

const $ = (id) => document.getElementById(id);
const fmt = (x, d = 4) => (x === null || x === undefined ? "" : Number(x).toFixed(d));
const DEFAULT_MODELS = ["Black-Scholes", "Local vol"];
let last = null;

function showFields() {
  const product = $("product").value;
  document.querySelectorAll(".digital").forEach((e) => e.classList.toggle("hidden", product !== "Digital"));
  document.querySelectorAll(".barrier").forEach((e) => e.classList.toggle("hidden", product !== "Barrier"));
}

function request() {
  return {
    trade: {
      ticker: $("ticker").value, product: $("product").value, option_type: $("option_type").value,
      strike_pct: Number($("strike_pct").value), expiry: $("expiry").value, quantity: Number($("quantity").value),
      payoff: $("payoff").value, cash_amount: Number($("cash_amount").value),
      barrier_type: $("barrier_type").value, barrier_pct: Number($("barrier_pct").value),
    },
    models: [...document.querySelectorAll("#models input:checked")].map((e) => e.value),
    n_paths: Number($("n_paths").value),
  };
}

function renderResults(d) {
  const t = d.trade;
  $("trade-info").textContent = `spot ${fmt(t.spot, 2)} · strike ${fmt(t.strike, 2)} · expiry ${t.expiry} ` +
    `(${fmt(t.T, 2)}y) · forward ${fmt(t.forward, 2)} · vol at strike ${fmt(100 * t.vol_at_strike, 2)}% · ` +
    `MC ${t.n_paths.toLocaleString()} paths, ${t.n_steps} steps`;
  const ref = d.rows.find((r) => r.error === undefined);
  const head = ["Model", "Unit price", "s.e.", "Price", "vs " + (ref ? ref.model : ""), "Delta", "Vega", "Rho",
    "Calib. [s]", "Pricing [s]"];
  const body = d.rows.map((r, i) => {
    if (r.error !== undefined) {
      return `<tr><td>${r.model}</td><td colspan="${head.length - 1}" class="error">${r.error}</td></tr>`;
    }
    const cells = [r.model, fmt(r.unit_price), fmt(r.standard_error), fmt(r.price), fmt(r.price - ref.price),
      fmt(r.delta), fmt(r.vega), fmt(r.rho), fmt(r.calibration_s, 1), fmt(r.pricing_s, 1)];
    return `<tr class="selectable" data-i="${i}">` + cells.map((c) => `<td>${c}</td>`).join("") + "</tr>";
  }).join("");
  $("results").innerHTML = `<table><thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead>` +
    `<tbody>${body}</tbody></table>`;
  document.querySelectorAll("tr.selectable").forEach((tr) => tr.addEventListener("click", () => vegaMap(Number(tr.dataset.i))));
  $("results-section").classList.remove("hidden");
  const first = d.rows.findIndex((r) => r.error === undefined);
  if (first >= 0) vegaMap(first);
}

function vegaMap(i) {
  const r = last.rows[i];
  document.querySelectorAll("tr.selectable").forEach((tr) => tr.classList.toggle("selected", Number(tr.dataset.i) === i));
  const g = last.vega_grid;
  const scale = Math.max(...r.vega_nodes.flat().map(Math.abs), 1e-12);
  const x = g.moneyness.map((m) => `${(100 * m).toFixed(0)}%`);
  const y = g.expiries.map((t) => `${t.toFixed(2)}y`);
  const text = r.vega_nodes.map((row) => row.map((v) => (Math.abs(v) >= 0.005 * scale ? v.toFixed(3) : "")));
  Plotly.react($("chart-vega"), [{
    type: "heatmap", z: r.vega_nodes, x, y, text, texttemplate: "%{text}", colorscale: "RdBu", reversescale: true,
    zmin: -scale, zmax: scale, colorbar: { title: { text: "vega" } },
  }], {
    title: { text: `${r.model}: vega per surface node (1 vol point)`, font: { size: 13 } },
    margin: { l: 60, r: 20, t: 40, b: 45 }, xaxis: { title: "strike / spot" }, yaxis: { title: "expiry", autorange: "reversed" },
  }, { displaylogo: false, responsive: true });
}

async function price() {
  const req = request();
  if (!req.models.length) { $("status").textContent = "Select at least one model."; return; }
  $("price").disabled = true;
  $("status").textContent = "Pricing… (first use of a model calibrates it)";
  try {
    const res = await fetch("/api/price", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    last = data;
    renderResults(data);
    $("status").textContent = "";
  } catch (e) {
    $("status").innerHTML = `<span class="error">Error: ${e.message}</span>`;
  } finally {
    $("price").disabled = false;
  }
}

async function init() {
  const [opts, tick] = await Promise.all([fetch("/api/tradepad/options").then((r) => r.json()),
    fetch("/api/tickers").then((r) => r.json())]);
  $("snapshot").textContent = `snapshot ${opts.snapshot}`;
  $("ticker").innerHTML = tick.tickers.map((t) => `<option>${t}</option>`).join("");
  $("product").innerHTML = opts.products.map((p) => `<option>${p}</option>`).join("");
  $("barrier_type").innerHTML = opts.barrier_types.map((b) => `<option>${b}</option>`).join("");
  $("models").innerHTML = opts.models.map((m) =>
    `<label><input type="checkbox" value="${m}" ${DEFAULT_MODELS.includes(m) ? "checked" : ""}>${m}</label>`).join("");
  const expiry = new Date(opts.snapshot);
  expiry.setFullYear(expiry.getFullYear() + 1);
  $("expiry").value = expiry.toISOString().slice(0, 10);
  $("product").addEventListener("change", showFields);
  $("price").addEventListener("click", price);
  showFields();
}

init();
