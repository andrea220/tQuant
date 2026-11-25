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
  state.job = null;                                  // the Projections belong to the last price only
  state.projections = null;
  setPriced(false);
  if (state.tradeReady || !$("trade-panel").classList.contains("hidden")) {
    state.tradeReady = false;
    $("trade-panel").classList.add("hidden");
    renderMarketData(null);
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
    const live = rows.some((r) => r.performance != null || r.outcome);   // a live trade: past fixings and outcomes
    $("trade-schedule").innerHTML = rows.length ? `<table><thead><tr><th>Fixing Date</th><th>Autocall Date</th>
      <th>Payment Date</th><th>Coupon Barrier</th><th>Coupon Amount</th><th>Autocall Barrier</th>${
      live ? "<th>Fixing (% strike)</th><th>Outcome</th>" : ""}</tr></thead><tbody>${
      rows.map((r) => `<tr><td>${r.fixing_date}</td><td>${r.autocall_date || ""}</td><td>${r.payment_date}</td>
        <td>${pct(r.coupon_barrier)}</td><td>${pct(r.coupon_amount)}</td><td>${pct(r.autocall_barrier)}</td>${
        live ? `<td>${r.performance == null ? "" : pct(r.performance)}</td><td>${r.outcome || ""}</td>` : ""}</tr>`).join("")
    }</tbody></table>` : "";
    renderMarketData(data.market_data);
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

// worst-of: the underlyings together, the first always there, the others added with "+ Add underlying"
const EXTRA = /^ticker_\d+$/;
const spotText = (ticker) => {
  const s = (spec().spots || {})[ticker];
  return s === undefined ? "" : `Spot ${fmtValue(s, 2)}`;
};

function underlyingRow(name, options, value, removable) {
  const sel = `<select id="f-${name}">${options.map((o) => `<option${String(o) === String(value) ? " selected" : ""}>${o}</option>`).join("")}</select>`;
  return `<div class="underlying-row" data-name="${name}"><div class="underlying-pick">${sel}${
    removable ? `<button type="button" class="link remove-underlying" title="Remove this underlying">&#x2715;</button>` : ""}</div>
    <span class="muted underlying-spot">${spotText(value)}</span></div>`;
}

function underlyingsHtml(head, keep) {
  const extras = Object.keys(keep).filter((k) => EXTRA.test(k)).sort((a, b) => parseInt(a.slice(7)) - parseInt(b.slice(7)));
  const rows = [underlyingRow(head.name, head.options, keep[head.name] ?? head.default, false),
    ...extras.map((k) => underlyingRow(k, head.options, keep[k], true))];
  return `<div class="subpanel"><div class="subpanel-title">Underlyings</div><div id="underlyings">${rows.join("")}</div>
    <button type="button" class="link add-underlying" id="add-underlying">+ Add underlying</button></div>`;
}

function wireUnderlyings(max) {
  const box = $("underlyings");
  const update = () => $("add-underlying").classList.toggle("hidden", box.children.length > max);
  const wire = (row) => {
    const rm = row.querySelector(".remove-underlying");
    if (rm) rm.addEventListener("click", () => { row.remove(); update(); invalidate(); refreshMarket(); });
    const sel = row.querySelector("select");
    sel.addEventListener("change", () => { row.querySelector(".underlying-spot").textContent = spotText(sel.value); });
    if (row.dataset.name !== "ticker") sel.addEventListener("change", refreshMarket);
  };
  $("add-underlying").addEventListener("click", () => {
    const options = spec().fields[0].options;
    const selects = [...box.querySelectorAll("select")];
    const free = options.find((o) => !selects.some((s) => s.value === o));
    if (free === undefined) return;
    const n = Math.max(1, ...selects.map((s) => parseInt(s.id.split("_")[1] || "1"))) + 1;
    box.insertAdjacentHTML("beforeend", underlyingRow(`ticker_${n}`, options, free, true));
    wire(box.lastElementChild);
    update();
    invalidate();
    refreshMarket();
  });
  box.querySelectorAll(".underlying-row").forEach(wire);
  update();
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
  // several underlyings (autocallable): all of them in the deal, on the right; on the left the pricing date only
  $("head-fields").innerHTML = (s.more_underlyings ? "" : fieldHtml(head, keep[head.name])) + dateField;
  // fields with a group are drawn as sub-panels (in order of first appearance), the others as one grid
  const groups = [];
  for (const f of rest) {
    const g = groups.find((x) => x.name === (f.group || ""));
    if (g) g.fields.push(f); else groups.push({ name: f.group || "", fields: [f] });
  }
  setupDealTabs();
  $("fields").innerHTML = (s.more_underlyings ? underlyingsHtml(head, keep) : "") + groups.map((g) => {
    const body = g.fields.map((f) => fieldHtml(f, keep[f.name])).join("");
    return g.name ? `<div class="subpanel"><div class="subpanel-title">${g.name}</div><div class="form deal-form">${body}</div></div>`
      : `<div class="form deal-form">${body}</div>`;
  }).join("");
  if (s.more_underlyings) wireUnderlyings(s.more_underlyings);
  $("market-section").classList.toggle("hidden", Boolean(s.more_underlyings));
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
  // Monte Carlo settings only for the Monte Carlo models (not Black-Scholes, discounting, analytic rates pricers)
  const closedForm = /^(Black-Scholes|Discounting|Analytic)/.test($("model").value);
  document.querySelectorAll(".mc").forEach((e) => e.classList.toggle("hidden", closedForm));
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
  $("snapshot").textContent = `${state.source || ""} · ${state.catalog.snapshot}`;
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
  document.querySelectorAll("#underlyings select").forEach((e) => {          // worst-of: the other underlyings
    if (EXTRA.test(e.id.slice(2))) fields[e.id.slice(2)] = e.value;
  });
  return { asset_class: state.cls, product: state.product, fields, date: state.date };
}

let marketTimer = null;
function refreshMarket() {
  clearTimeout(marketTimer);
  marketTimer = setTimeout(loadMarket, 200);
}

async function loadMarket() {
  if (spec().more_underlyings) {           // several underlyings: no single spot / vol to show or edit
    state.market = {};
    $("div-panel").classList.add("hidden");
    return;
  }
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
  loadLegs();
}

// swaps: the two legs and their cash flows on the market (fixed and floating, curves, PV), at every deal change
async function loadLegs() {
  const show = Boolean(spec().legs);
  $("legs-panel").classList.toggle("hidden", !show);
  if (!show) return;
  try {
    const res = await fetch("/api/tradepad/trade", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(dealRequest()) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    renderLegs(data);
  } catch (e) {
    $("legs").innerHTML = `<span class="error">${e.message}</span>`;
  }
}

function renderLegs(data) {
  $("legs-npv").textContent = `(NPV of the flows ${fmtValue(data.npv, 2)})`;
  $("legs-summary").innerHTML = data.summary.map(([l, v, d]) =>
    `<div><span class="muted">${l}</span><b>${fmtValue(v, d)}</b></div>`).join("");
  const num = (x, d) => (x === null || x === undefined ? "" : fmtValue(x, d));
  $("legs").innerHTML = data.legs.map((leg) => {
    const floating = leg.projection !== null;
    const head = ["Start", "End", "Payment", ...(floating ? ["Fixing"] : []), "Year frac.", "Rate %",
      ...(floating ? ["Spread bp"] : []), "Amount", "DF", "PV"];
    const rows = leg.rows.map((r) => `<tr><td>${r.start}</td><td>${r.end}</td><td>${r.payment}</td>`
      + (floating ? `<td>${r.fixing ? r.fixing + (r.known ? ' <span class="pill">fixed</span>' : "") : "compounded"}</td>` : "")
      + `<td>${num(r.year_fraction, 5)}</td><td>${num(100 * r.rate, 4)}</td>`
      + (floating ? `<td>${num(1e4 * r.spread, 1)}</td>` : "")
      + `<td>${num(r.amount, 2)}</td><td>${num(r.df, 6)}</td><td>${num(r.pv, 2)}</td></tr>`).join("");
    const dir = leg.direction === "Pay" ? "pay" : "receive";
    return `<div class="subpanel leg">
      <div class="leg-head"><span class="subpanel-title">${leg.name}</span><span class="pill ${dir}">${leg.direction}</span>
        <span class="leg-pv">PV <b>${fmtValue(leg.pv, 2)}</b></span></div>
      <div class="mini-info leg-info">
        <div><span class="muted">Conventions</span><b>${leg.conventions}</b></div>
        ${floating ? `<div><span class="muted">Index</span><b>${leg.index}</b></div>
        <div><span class="muted">Projection curve</span><b>${leg.projection}</b></div>` : ""}
        <div><span class="muted">Discount curve</span><b>${leg.discount}</b></div>
      </div>
      <div class="hscroll"><table class="leg-table"><thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead>
        <tbody>${rows}</tbody></table></div></div>`;
  }).join("");
}

// equity products: the underlying's dividends with ex-date up to the expiry
// ---------- Market Data tab (products built with "Build trade"): the market data the trade is priced on ----------
// Tabs (products built with "Build trade"): Deal, Market Data, Risk, Projections; without them, the deal and the
// results one below the other. Tab hiding uses its own class, apart from the panels hidden for lack of data.
function showTab(tab) {
  const tabbed = Boolean(state.catalog && state.product && spec().create_trade);
  state.tab = tabbed ? tab : "deal";
  document.querySelectorAll("#main-tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === state.tab));
  const t = state.tab;
  const set = (id, show) => $(id).classList.toggle("tab-hidden", tabbed && !show);
  set("deal-panel", t === "deal" || t === "market");
  set("trade-panel", t === "deal");
  set("legs-panel", t === "deal");
  set("div-panel", t === "deal");
  set("empty", t === "deal");
  set("results-section", t === "risk" || t === "projections");
  $("fields").classList.toggle("tab-hidden", t === "market");
  $("deal-title").classList.toggle("tab-hidden", t === "market");
  $("market-data").classList.toggle("hidden", t !== "market");
  $("risk-panels").classList.toggle("hidden", tabbed && t !== "risk");
  $("projections").classList.toggle("hidden", !(tabbed && t === "projections"));
  if (t === "projections") loadProjections();
}

function setPriced(priced) {                         // Risk and Projections exist after a price only
  document.querySelectorAll('#main-tabs button[data-tab="risk"], #main-tabs button[data-tab="projections"]')
    .forEach((b) => { b.disabled = !priced; });
  const projections = document.querySelector('#main-tabs button[data-tab="projections"]');
  if (projections) projections.disabled = !(priced && state.job);
  if (!priced && (state.tab === "risk" || state.tab === "projections")) showTab("deal");
}

function setupDealTabs() {
  $("main-tabs").classList.toggle("hidden", !spec().create_trade);
  setPriced(false);
  showTab("deal");
  renderMarketData(null);
}

function renderMarketData(md) {
  if (!md) {
    $("market-data").innerHTML = '<p class="muted">Press <b>Build trade</b> to see the market data of the trade.</p>';
    return;
  }
  const info = (rows) => `<div class="mini-info wide-info">${rows.map(([l, v, d]) =>
    `<div><span class="muted">${l}</span><b>${fmtValue(v, d)}</b></div>`).join("")}</div>`;
  const und = md.underlyings.map((u) => {
    const rows = [["Spot", u.spot, 2], ["Currency", u.ccy + (u.quanto ? " (quanto)" : ""), null],
      ["Level at the strike date", u.strike_level, 2], ["Source", u.strike_source, null],
      [`Forward at maturity (${md.maturity})`, u.forward, 2], ["Repo to maturity (%)", u.repo, 3],
      ["PV dividends to maturity", u.pv_dividends, 4]];
    if (u.quanto) rows.push(["Correlation with the FX rate", u.quanto.correlation, 3], ["FX vol (%)", u.quanto.fx_vol, 2]);
    const divs = u.dividends.length
      ? `<table><thead><tr><th>Ex date</th><th>Pay date</th><th>Amount</th><th>% spot</th><th>PV</th></tr></thead><tbody>${
        u.dividends.map((r) => `<tr><td>${r.ex_date}</td><td>${r.pay_date}</td><td>${fmtValue(r.amount, 4)}</td>
        <td>${fmtValue(r.pct_spot, 2)}%</td><td>${fmtValue(r.pv, 4)}</td></tr>`).join("")}</tbody></table>`
      : `<p class="muted">${u.dividend_model === "continuous" && u.div_yield !== null
        ? `Continuous dividend yield ${fmtValue(u.div_yield, 3)}%.` : "No dividends before the maturity."}</p>`;
    return `<div class="subpanel"><div class="subpanel-title">${u.ticker}</div>${info(rows)}
      <div class="subpanel-title" style="margin-top: 12px">Dividends to maturity (${u.dividends.length})</div>
      <div class="hscroll">${divs}</div></div>`;
  }).join("");
  const curves = md.curves.map((c) => `<div class="subpanel"><div class="subpanel-title">${c.label}
      <span class="muted">(${c.role})</span></div>${info([["Zero rate at maturity (%)", c.zero_at_maturity, 3],
      ["Discount factor at maturity", c.df_at_maturity, 5]])}
      <div class="hscroll" style="margin-top: 10px"><table><thead><tr><th>Pillar</th><th>Zero rate (%)</th>
      <th>Discount factor</th></tr></thead><tbody>${c.pillars.map((p) => `<tr><td>${p.date}</td>
      <td>${fmtValue(p.zero, 3)}</td><td>${fmtValue(p.df, 5)}</td></tr>`).join("")}</tbody></table></div></div>`).join("");
  $("market-data").innerHTML = `<h2>Underlyings</h2>${und}<h2 style="margin-top: 18px">Curves</h2>${curves}`;
}

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
  // products with tabs (autocallable): the details as one column, growing in height; the others: cards
  const compact = Boolean(spec().create_trade);
  $("info").className = compact ? "mini-info one-column" : "cards";
  $("info").innerHTML = compact
    ? d.info.map(([l, v, dd]) => `<div><span class="muted">${l}</span><b>${fmtValue(v, dd)}</b></div>`).join("")
    : cards(d.info);
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

  $("delta-panel").classList.toggle("hidden", !d.deltas);
  if (d.deltas) {
    $("delta-table").innerHTML = `<table><thead><tr><th>Underlying</th><th>Spot</th><th>Delta (per 1 of spot)</th>
      <th>Delta 1% (price change)</th><th>Delta 1% (% notional)</th></tr></thead><tbody>${d.deltas.map((r) =>
      `<tr><td>${r.ticker} <span class="muted">${r.ccy}</span></td><td>${fmtValue(r.spot, 2)}</td>
      <td>${fmtValue(r.delta, 6)}</td><td>${fmtValue(r.delta_1pct, 4)}</td><td>${fmtValue(r.delta_1pct_notional, 3)}%</td>
      </tr>`).join("")}</tbody></table>`;
  }
  $("vega-panel").classList.toggle("hidden", !d.vega);
  const maps = d.vega_maps || [];                  // several underlyings: one map at a time, a sub-tab each
  $("vega-tabs").classList.toggle("hidden", maps.length < 2);
  $("vega-tabs").innerHTML = maps.map((m, i) =>
    `<button type="button" data-i="${i}" class="${i === 0 ? "active" : ""}">${m.ticker}</button>`).join("");
  document.querySelectorAll("#vega-tabs button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("#vega-tabs button").forEach((x) => x.classList.toggle("active", x === b));
    drawVega(maps[Number(b.dataset.i)]);
  }));
  if (d.vega) drawVega(d.vega);


  $("rho-panel").classList.toggle("hidden", !d.rho.length);
  if (d.rho.length) {
    const t = d.sens_table;
    if (t) {                                     // pillars x (Par | Zero) x curves, total first
      const n = t.curves.length;
      const cell = (x, ok = true) => `<td>${!ok ? '<span class="muted">n/a</span>' : x === null ? "" : fmtValue(x, 2)}</td>`;
      const head = `<tr><th rowspan="2">Pillar</th><th colspan="${n}" class="group-start group-title">Par</th>`
        + `<th colspan="${n}" class="group-start group-title">Zero</th></tr><tr>`
        + t.curves.map((c, j) => `<th class="${j === 0 ? "group-start" : ""}">${c}</th>`).join("")
        + t.curves.map((c, j) => `<th class="${j === 0 ? "group-start" : ""}">${c}</th>`).join("") + "</tr>";
      const row = (label, par, zero, cls = "") => `<tr class="${cls}"><td>${label}</td>`
        + par.map((x, j) => cell(x, t.par_available[j]).replace("<td>", j === 0 ? '<td class="group-start">' : "<td>")).join("")
        + zero.map((x, j) => cell(x).replace("<td>", j === 0 ? '<td class="group-start">' : "<td>")).join("") + "</tr>";
      const body = row("Total", t.total.par, t.total.zero, "total-row")
        + t.rows.map((r) => row(r.label, r.par, r.zero)).join("");
      $("rho-table").innerHTML = `<table class="fit-grid sens-grid"><thead>${head}</thead><tbody>${body}</tbody></table>`;
      const missing = t.curves.filter((c, j) => !t.par_available[j]);
      $("rho-note").textContent = "Par: to the market quotes the curves are bootstrapped from (curve rebootstrapped; "
        + "the Ibor curve moves with its discount curve). Zero: to the zero rates of the pillars, the other curves held."
        + (missing.length ? ` No par sensitivity for ${missing.join(", ")} (no stored bootstrap Jacobian).` : "");
    } else {                                     // one row per pillar, one column per curve, the total first
      const dates = [...new Set(d.rho.flatMap((r) => r.dates))].sort();
      const value = (r, x) => { const i = r.dates.indexOf(x); return i < 0 ? "" : fmtValue(r.rho[i], 2); };
      const head = "<tr><th>Pillar</th>" + d.rho.map((r) => `<th>${r.curve}</th>`).join("") + "</tr>";
      const total = '<tr class="total-row"><td>Total</td>' + d.rho.map((r) => `<td>${fmtValue(r.total, 2)}</td>`).join("")
        + "</tr>";
      const body = dates.map((x) => `<tr><td>${x}</td>` + d.rho.map((r) => `<td>${value(r, x)}</td>`).join("")
        + "</tr>").join("");
      $("rho-table").innerHTML = `<table class="sens-grid"><thead>${head}</thead><tbody>${total}${body}</tbody></table>`;
      $("rho-note").textContent = "";
    }
  }
  $("results-section").classList.remove("hidden");
  $("empty").classList.add("hidden");
}

// ---------- Projections tab: expected flows, autocall probabilities and levels from the paths of the price ----------
async function loadProjections() {
  if (!state.job || state.projections) return;
  $("status").textContent = "Computing projections…";
  try {
    state.projections = await post("/api/price/projections", { job: state.job });
    renderProjections(state.projections);
    $("status").textContent = "";
  } catch (e) {
    $("status").innerHTML = `<span class="error">Error: ${e.message}</span>`;
  }
}

function renderProjections(p) {
  const layout = (y, extra = {}) => ({ margin: { l: 60, r: 20, t: 10, b: 50 }, font: FONT, paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)", yaxis: { title: y }, legend: { orientation: "h", y: -0.2 }, ...extra });
  const cfg = { displaylogo: false, responsive: true };
  const f = p.cash_flows;
  $("pv-info").innerHTML = [["PV coupons (% notional)", p.pv.coupons, 3], ["PV put (% notional)", p.pv.put, 3],
    ["Price (% notional)", p.pv.total, 3]]
    .map(([l, v, d]) => `<div><span class="muted">${l}</span><b>${fmtValue(v, d)}</b></div>`).join("");
  Plotly.react($("chart-flows"), [
    { type: "bar", name: "Coupons", x: f.map((r) => r.date), y: f.map((r) => r.coupon) },
    { type: "bar", name: "Redemption", x: f.map((r) => r.date), y: f.map((r) => r.redemption) },
    { type: "bar", name: "Put", x: f.map((r) => r.date), y: f.map((r) => r.put) }],
  layout("% of notional", { barmode: "relative" }), cfg);
  $("flows-table").innerHTML = `<table><thead><tr><th>Payment date</th><th>Coupon</th><th>Redemption</th><th>Put</th>
    <th>Total</th></tr></thead><tbody>${f.map((r) => `<tr><td>${r.date}</td><td>${fmtValue(r.coupon, 3)}%</td>
    <td>${fmtValue(r.redemption, 3)}%</td><td>${fmtValue(r.put, 3)}%</td><td><b>${fmtValue(r.total, 3)}%</b></td></tr>`)
    .join("")}</tbody></table>`;
  const m = p.maturity;
  $("maturity-info").innerHTML = [["Paths", p.n_paths, 0], ["Expected life (years)", m.expected_life, 2],
    ["Reaching maturity (%)", m.reached, 2], ["Knock-in at maturity (%)", m.knock_in, 2],
    ["Loss given knock-in (% notional)", m.loss_given_knock_in, 2]]
    .map(([l, v, d]) => `<div><span class="muted">${l}</span><b>${fmtValue(v, d)}</b></div>`).join("");

  const a = p.autocall;
  Plotly.react($("chart-autocall"), [
    { type: "bar", name: "Called on this date", x: a.map((r) => r.date), y: a.map((r) => r.probability) },
    { type: "scatter", mode: "lines+markers", name: "Called by this date", x: a.map((r) => r.date),
      y: a.map((r) => r.cumulative) }], layout("probability (%)"), cfg);
  const traces = [];
  const palette = ["#1f4fa3", "#d9822b", "#2e9e6a", "#b2416e", "#7a5cc2"];
  const rgba = (hex, a) => `rgba(${parseInt(hex.slice(1, 3), 16)}, ${parseInt(hex.slice(3, 5), 16)}, ${parseInt(hex.slice(5, 7), 16)}, ${a})`;
  p.forwards.forEach((u, k) => {
    const x = u.rows.map((r) => r.date);
    const name = u.ticker + (u.quanto ? " (quanto)" : "");
    const color = palette[k % palette.length];
    traces.push({ type: "scatter", mode: "lines", x, y: u.rows.map((r) => 100 * r.p90 / u.spot), line: { width: 0 },
      hoverinfo: "skip", showlegend: false, legendgroup: u.ticker });
    traces.push({ type: "scatter", mode: "lines", name: `${u.ticker} 10%-90%`, x, legendgroup: u.ticker,
      y: u.rows.map((r) => 100 * r.p10 / u.spot), fill: "tonexty", fillcolor: rgba(color, 0.15), line: { width: 0 } });
    traces.push({ type: "scatter", mode: "lines+markers", name: `${name} paths`, x, legendgroup: u.ticker,
      line: { color }, marker: { color }, y: u.rows.map((r) => 100 * r.mean / u.spot) });
    if (u.rows.some((r) => r.forward !== null)) traces.push({ type: "scatter", mode: "lines", name: `${u.ticker} forward`, x,
      legendgroup: u.ticker, y: u.rows.map((r) => (r.forward === null ? null : 100 * r.forward / u.spot)),
      line: { dash: "dash", color } });
  });
  Plotly.react($("chart-forwards"), traces, layout("% of spot"), cfg);
  const dates = p.forwards.length ? p.forwards[0].rows.map((r) => r.date) : [];
  $("forwards-table").innerHTML = dates.length ? `<table><thead><tr><th>Date</th>${p.forwards.map((u) =>
    `<th>${u.ticker} paths</th><th>${u.ticker} forward</th>`).join("")}</tr></thead><tbody>${dates.map((d, i) =>
    `<tr><td>${d}</td>${p.forwards.map((u) => `<td>${fmtValue(u.rows[i].mean, 2)}</td>
    <td>${fmtValue(u.rows[i].forward, 2)}</td>`).join("")}</tr>`).join("")}</tbody></table>` : "";
}

function drawVega(v) {
  $("vega-title").textContent = `${v.title} · total ${fmtValue(v.total, 2)}`;
  const scale = Math.max(...v.z.flat().map(Math.abs), 1e-12);
  const text = v.z.map((row) => row.map((x) => (Math.abs(x) >= 0.01 * scale ? x.toPrecision(3) : "")));
  Plotly.react($("chart-vega"), [{ type: "heatmap", z: v.z, x: v.x, y: v.y, text, texttemplate: "%{text}",
    colorscale: "RdBu", reversescale: true, zmin: -scale, zmax: scale }],
  { margin: { l: 60, r: 20, t: 10, b: 50 }, font: FONT, paper_bgcolor: "rgba(0,0,0,0)",
    xaxis: { title: v.x_title }, yaxis: { title: v.y_title, autorange: "reversed" } },
  { displaylogo: false, responsive: true });
}

// first phase: the price only, in the valuation bar; the greeks panels wait for the second phase
function renderPrice(d) {
  $("model-info").textContent = `${d.product} · ${d.model} · ${d.snapshot}`;
  $("valuation-bar").classList.remove("stale");
  $("valuation").innerHTML = d.headline.map(([l, v, dd], i) =>
    `<div class="vb-card${i === 0 ? " main" : ""}"><div class="label">${l}</div><div class="value">${fmtValue(v, dd)}</div></div>`).join("")
    + '<div class="vb-card"><div class="label">Greeks</div><div class="value muted">computing…</div></div>';
  ["sens-panel", "delta-panel", "vega-panel", "rho-panel"].forEach((id) => $(id).classList.add("hidden"));
}

async function post(url, body) {
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body) });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

async function price() {
  $("price").disabled = true;
  $("status").textContent = "Pricing…";
  try {
    const body = { ...dealRequest(), model: $("model").value, n_paths: Number($("n_paths").value), overrides: overrides(),
      phase: "price" };
    let data = await post("/api/price", body);
    state.job = data.has_projections ? data.job : null;     // projections: computed when their tab is opened
    state.projections = null;
    setPriced(true);
    showTab("risk");                                  // the results: the charts are drawn in a visible panel
    if (data.job) {                                   // price first, then the greeks of the same computation
      renderPrice(data);
      $("status").textContent = "Computing greeks…";
      data = await post("/api/price/greeks", { job: data.job });
    }
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
  const snaps = await (await fetch("/api/snapshots")).json();
  state.dates = snaps.snapshots.map((s) => s.date);
  state.source = snaps.source;
  state.date = state.dates[0];
  await loadCatalog();
  const classes = Object.keys(state.catalog.asset_classes);
  $("asset-classes").innerHTML = classes.map((c) => `<button data-cls="${c}">${c}</button>`).join("");
  document.querySelectorAll(".segmented button").forEach((b) => b.addEventListener("click", () => selectClass(b.dataset.cls)));
  $("product").addEventListener("change", () => selectProduct($("product").value));
  $("model").addEventListener("change", showModelFields);
  $("price").addEventListener("click", price);
  $("create").addEventListener("click", createTrade);
  document.querySelectorAll("#main-tabs button").forEach((b) => b.addEventListener("click", () => {
    if (!b.disabled) showTab(b.dataset.tab);
  }));
  $("reset-market").addEventListener("click", () => { state.edited = {}; invalidate(); loadMarket(); });
  for (const id of ["inputs-left", "deal-panel"]) {
    $(id).addEventListener("input", invalidate);
    $(id).addEventListener("change", invalidate);
    $(id).addEventListener("click", (ev) => { if (ev.target.closest(".segmented button")) invalidate(); });
  }
  selectClass(classes[0]);
}

init();
