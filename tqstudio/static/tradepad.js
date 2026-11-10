// tqstudio Tradepad: define a deal, pick one model, POST /api/price, show valuation, market inputs and the vega map.

const $ = (id) => document.getElementById(id);
const fmt = (x, d = 4) => (x === null || x === undefined ? "–" : Number(x).toFixed(d));
const pct = (x, d = 2) => (x === null || x === undefined ? "–" : (100 * x).toFixed(d) + "%");

function showFields() {
  const product = $("product").value;
  document.querySelectorAll(".digital").forEach((e) => e.classList.toggle("hidden", product !== "Digital"));
  document.querySelectorAll(".barrier").forEach((e) => e.classList.toggle("hidden", product !== "Barrier"));
  document.querySelectorAll(".mc").forEach((e) => e.classList.toggle("hidden", $("model").value !== "Local vol"));
}

function request() {
  return {
    trade: {
      ticker: $("ticker").value, product: $("product").value, option_type: $("option_type").value,
      strike_pct: Number($("strike_pct").value), expiry: $("expiry").value, quantity: Number($("quantity").value),
      payoff: $("payoff").value, cash_amount: Number($("cash_amount").value),
      barrier_type: $("barrier_type").value, barrier_pct: Number($("barrier_pct").value),
    },
    model: $("model").value,
    n_paths: Number($("n_paths").value),
  };
}

const cards = (items, big = 0) => items.map(([l, v], i) =>
  `<div class="card${i < big ? " big" : ""}"><div class="label">${l}</div><div class="value">${v}</div></div>`).join("");

function render(d) {
  const t = d.trade, r = d.result;
  $("model-info").textContent = r.model + (t.n_paths ? ` · ${t.n_paths.toLocaleString()} paths, ${t.n_steps} steps` : "");
  $("valuation").innerHTML = cards([
    ["Price", fmt(r.price)], ["Premium (% spot)", pct(r.unit_price / t.spot, 3)], ["Std error", fmt(r.standard_error)],
    ["Delta", fmt(r.delta)], ["Vega (1 vol pt)", fmt(r.vega)], ["Rho (1 bp)", fmt(r.rho)],
    ["Calibration", fmt(r.calibration_s, 1) + " s"], ["Pricing", fmt(r.pricing_s, 1) + " s"]], 2);
  const market = [["Spot", fmt(t.spot, 2)], ["Strike", fmt(t.strike, 2)], ["Expiry", `${t.expiry} (${fmt(t.T, 2)}y)`],
    ["Forward", fmt(t.forward, 2)], ["Rate (zero)", pct(t.rate, 3)], ["Discount factor", fmt(t.discount_factor, 5)],
    ["Repo", pct(t.repo, 3)], ["PV dividends", fmt(t.pv_dividends, 2)], ["Impl. div yield", pct(t.implied_div_yield, 3)],
    ["Vol at strike", pct(t.vol_at_strike)], ["Vol ATM forward", pct(t.vol_atm_forward)]];
  if (t.barrier !== null) market.splice(2, 0, ["Barrier", fmt(t.barrier, 2)]);
  $("market").innerHTML = cards(market);
  vegaMap(r, d.vega_grid);
  $("results-section").classList.remove("hidden");
  $("empty").classList.add("hidden");
}

function vegaMap(r, g) {
  const scale = Math.max(...r.vega_nodes.flat().map(Math.abs), 1e-12);
  const x = g.moneyness.map((m) => `${(100 * m).toFixed(0)}%`);
  const y = g.expiries.map((t) => `${t.toFixed(2)}y`);
  const text = r.vega_nodes.map((row) => row.map((v) => (Math.abs(v) >= 0.005 * scale ? v.toFixed(3) : "")));
  Plotly.react($("chart-vega"), [{
    type: "heatmap", z: r.vega_nodes, x, y, text, texttemplate: "%{text}", colorscale: "RdBu", reversescale: true,
    zmin: -scale, zmax: scale, colorbar: { title: { text: "vega" } },
  }], {
    margin: { l: 60, r: 20, t: 10, b: 45 }, xaxis: { title: "strike / spot" }, yaxis: { title: "expiry", autorange: "reversed" },
    font: { size: 12, family: "Segoe UI, system-ui, sans-serif" }, paper_bgcolor: "rgba(0,0,0,0)",
  }, { displaylogo: false, responsive: true });
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
  const [opts, tick] = await Promise.all([fetch("/api/tradepad/options").then((r) => r.json()),
    fetch("/api/tickers").then((r) => r.json())]);
  $("snapshot").textContent = `snapshot ${opts.snapshot}`;
  $("ticker").innerHTML = tick.tickers.map((t) => `<option>${t}</option>`).join("");
  $("product").innerHTML = opts.products.map((p) => `<option>${p}</option>`).join("");
  $("barrier_type").innerHTML = opts.barrier_types.map((b) => `<option>${b}</option>`).join("");
  $("model").innerHTML = opts.models.map((m) => `<option>${m}</option>`).join("");
  const expiry = new Date(opts.snapshot);
  expiry.setFullYear(expiry.getFullYear() + 1);
  $("expiry").value = expiry.toISOString().slice(0, 10);
  $("product").addEventListener("change", showFields);
  $("model").addEventListener("change", showFields);
  $("price").addEventListener("click", price);
  showFields();
}

init();
