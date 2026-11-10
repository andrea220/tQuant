// tqstudio market data page: pick an underlying, fetch /api/market/<ticker>, draw tables and charts.

const $ = (id) => document.getElementById(id);
const fmt = (x, d = 4) => (x === null || x === undefined ? "" : Number(x).toFixed(d));
const pct = (x, d = 2) => (x === null || x === undefined ? "" : (100 * x).toFixed(d));
const LAYOUT = { margin: { l: 55, r: 15, t: 35, b: 45 }, font: { size: 12 }, showlegend: false };

function table(columns, rows) {
  // columns: [[header, row => cell text], ...]
  const head = "<tr>" + columns.map(([h]) => `<th>${h}</th>`).join("") + "</tr>";
  const body = rows.map((r) => "<tr>" + columns.map(([, f]) => `<td>${f(r)}</td>`).join("") + "</tr>").join("");
  return `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

function plot(id, traces, title, xTitle, yTitle, extra = {}) {
  Plotly.react($(id), traces, { ...LAYOUT, title: { text: title, font: { size: 13 } },
    xaxis: { title: xTitle }, yaxis: { title: yTitle }, ...extra }, { displaylogo: false, responsive: true });
}

function render(d) {
  const s = d.summary;
  const cards = [["Spot", fmt(s.spot, 2)], ["Currency", s.currency], ["Repo", pct(s.repo, 3) + "%"],
    ["Dividends", s.dividend_model], ["Discount curve", s.discount_curve], ["Snapshot", s.snapshot]];
  if (s.div_yield !== null) cards.splice(4, 0, ["Div yield", pct(s.div_yield, 3) + "%"]);
  $("summary").innerHTML = cards.map(([l, v]) =>
    `<div class="card"><div class="label">${l}</div><div class="value">${v}</div></div>`).join("");

  $("forwards").innerHTML = table([
    ["Tenor", (r) => r.tenor], ["Forward", (r) => fmt(r.forward, 2)], ["Fwd / spot", (r) => fmt(r.forward_over_spot)],
    ["PV divs", (r) => fmt(r.pv_dividends, 2)], ["Impl. div yield %", (r) => pct(r.implied_div_yield)],
    ["DF", (r) => fmt(r.discount_factor)]], d.forwards);
  $("dividends").innerHTML = d.dividends.length
    ? table([["Ex date", (r) => r.ex_date], ["Pay date", (r) => r.pay_date || ""], ["Amount", (r) => fmt(r.amount)],
        ["% spot", (r) => fmt(r.pct_spot, 2)]], d.dividends)
    : `<p class="muted">No discrete dividends (dividend model: ${s.dividend_model}).</p>`;

  const fc = d.forward_curve;
  plot("chart-forward", [{ x: fc.T, y: fc.forward, mode: "lines", name: "forward" },
    { x: [0, fc.T[fc.T.length - 1]], y: [s.spot, s.spot], mode: "lines", line: { dash: "dash", color: "#888" }, name: "spot" }],
    "Forward", "T (years)", "");
  plot("chart-divyield", [{ x: fc.T, y: fc.implied_div_yield.map((v) => 100 * v), mode: "lines" }],
    "Implied dividend yield", "T (years)", "%");

  const c = d.curve;
  $("curve-name").textContent = c.name;
  $("curve").innerHTML = table([["Date", (r) => r.date], ["T", (r) => fmt(r.T)],
    ["Zero rate %", (r) => pct(r.zero_rate, 4)]], c.pillars);
  plot("chart-curve", [{ x: c.grid.T, y: c.grid.zero_rate.map((v) => 100 * v), mode: "lines" },
    { x: c.pillars.map((p) => p.T), y: c.pillars.map((p) => 100 * p.zero_rate), mode: "markers", marker: { size: 5 } }],
    "Zero curve", "T (years)", "%");

  const v = d.vol;
  if (v) {
    const cols = [["Expiry", (r) => r.expiry]].concat(
      v.moneyness.map((m, j) => [`${(100 * m).toFixed(0)}%`, (r) => fmt(100 * r.row[j], 2)]));
    $("vol").innerHTML = table(cols, v.expiries.map((e, i) => ({ expiry: e, row: v.matrix[i] })));
    const n = v.expiries.length;
    plot("chart-smiles", v.matrix.map((row, i) => ({
      x: v.moneyness.map((m) => 100 * m), y: row.map((x) => 100 * x), mode: "lines+markers", name: v.expiries[i],
      marker: { size: 4 }, line: { color: `hsl(${220 - (160 * i) / Math.max(n - 1, 1)}, 65%, 45%)` } })),
      "Smiles", "strike / spot (%)", "implied vol (%)", { showlegend: true, legend: { font: { size: 10 } } });
    plot("chart-atm", [{ x: v.T, y: v.atm_forward.map((x) => 100 * x), mode: "lines+markers" }],
      "ATM-forward term structure", "T (years)", "implied vol (%)");
  } else {
    $("vol").innerHTML = '<p class="muted">No vol surface.</p>';
  }

  $("correlations").innerHTML = d.correlations.length
    ? table([["Factor", (r) => r.factor], ["Correlation", (r) => fmt(r.correlation)]], d.correlations)
    : '<p class="muted">Not in the correlation matrix.</p>';
}

async function load(ticker) {
  $("status").textContent = `Loading ${ticker}…`;
  $("status").style.display = "";
  try {
    const res = await fetch(`/api/market/${encodeURIComponent(ticker)}`);
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    render(await res.json());
    $("status").style.display = "none";
    history.replaceState(null, "", `?ticker=${encodeURIComponent(ticker)}`);
  } catch (e) {
    $("status").textContent = `Error: ${e.message}`;
  }
}

async function init() {
  const res = await fetch("/api/tickers");
  const { snapshot, tickers } = await res.json();
  $("snapshot").textContent = `snapshot ${snapshot}`;
  const sel = $("ticker");
  sel.innerHTML = tickers.map((t) => `<option>${t}</option>`).join("");
  const wanted = new URLSearchParams(location.search).get("ticker");
  sel.value = tickers.includes(wanted) ? wanted : tickers[0];
  sel.addEventListener("change", () => load(sel.value));
  load(sel.value);
}

init();
