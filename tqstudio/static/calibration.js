// tqstudio calibration tab: calibration sets (list, one set, one underlying: models, fit, test) and new runs.

const $ = (id) => document.getElementById(id);
const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x) ? "" : Number(x).toFixed(d));
const AXIS = { gridcolor: "#eef1f6", zerolinecolor: "#dde2ea", linecolor: "#dde2ea" };
const LAYOUT = { margin: { l: 55, r: 15, t: 38, b: 45 }, font: { size: 12, family: "Segoe UI, system-ui, sans-serif",
  color: "#18202f" }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)" };
const MODEL_COLORS = { Heston: "#1f4fa8", DisplacedDiffusion: "#e8833a", HestonSLV: "#12805c", LocalVol: "#a43fd1" };
const LABEL = { Heston: "Heston", DisplacedDiffusion: "Displaced diffusion", HestonSLV: "Heston SLV", LocalVol: "Local vol",
  Black: "Black" };
const QUALITY = { good: "#12805c", fair: "#e8833a", poor: "#c2362f" };
const state = { sets: [], set: null, und: null, model: null };

function table(columns, rows, attrs = "") {
  const head = "<tr>" + columns.map(([h]) => `<th>${h}</th>`).join("") + "</tr>";
  const body = rows.map((r) => `<tr ${typeof attrs === "function" ? attrs(r) : attrs}>`
    + columns.map(([, f]) => `<td>${f(r)}</td>`).join("") + "</tr>").join("");
  return `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
}
const cards = (items) => items.map(([l, v], i) =>
  `<div class="card${i === 0 ? " big" : ""}"><div class="label">${l}</div><div class="value">${v}</div></div>`).join("");
function plot(id, traces, title, xTitle, yTitle) {
  Plotly.react($(id), traces, { ...LAYOUT, title: { text: title, font: { size: 13 } }, showlegend: true,
    legend: { font: { size: 10 } }, xaxis: { title: xTitle, ...AXIS }, yaxis: { title: yTitle, ...AXIS } },
  { displaylogo: false, responsive: true });
}
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
function expiryLabels(expiries, T) {
  const month = (e) => `${MONTHS[Number(e.slice(5, 7)) - 1]}${e.slice(2, 4)}`;
  const count = {};
  expiries.forEach((e, i) => { if (Math.abs(T[i] - Math.round(T[i])) > 1e-9) count[month(e)] = (count[month(e)] || 0) + 1; });
  return expiries.map((e, i) => (Math.abs(T[i] - Math.round(T[i])) <= 1e-9 && T[i] >= 1 ? `${Math.round(T[i])}Y`
    : count[month(e)] > 1 ? `${Number(e.slice(8, 10))}${month(e)}` : month(e)));
}
async function getJSON(url, options) {
  const res = await fetch(url, options);
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}
const post = (url, body) => getJSON(url, { method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body) });
function saveUrl() {
  const p = new URLSearchParams();
  if (state.set) { p.set("date", state.set.date); p.set("set", state.set.set); }
  if (state.und) p.set("ticker", state.und.ticker);
  history.replaceState(null, "", p.toString() ? `?${p}` : location.pathname);
}

// ---------- list ----------
async function loadSets() {
  const data = await getJSON("/api/calibration/sets");
  state.sets = data.sets;
  showRun(data.run);
  $("sets").innerHTML = state.sets.length ? table([
    ["Date", (s) => s.date], ["Time", (s) => s.set], ["Snapshot", (s) => `${s.snapshot.date} ${s.snapshot.set}`],
    ["Models", (s) => s.models.map((m) => LABEL[m] || m).join(", ") + (s.slv.length ? ` + SLV (${s.slv.join(", ")})` : "")],
    ["Underlyings", (s) => s.tickers.length], ["Status", (s) => Object.entries(s.counts).map(([k, v]) => `${v} ${k}`).join(", ")],
    ["Time (s)", (s) => fmt(s.seconds, 0)]], state.sets, (s) => `class="clickable" data-date="${s.date}" data-set="${s.set}"`)
    : '<p class="muted">No calibration set yet: start one with "New calibration set".</p>';
  $("sets").querySelectorAll("tr.clickable").forEach((tr) => tr.addEventListener("click",
    () => openSet(tr.dataset.date, tr.dataset.set)));
}

// ---------- one set ----------
async function openSet(day, cset, ticker) {
  const data = await getJSON(`/api/calibration/set/${day}/${cset}`);
  state.set = { date: day, set: cset, ...data };
  state.und = null;
  $("set-panel").classList.remove("hidden");
  $("und-view").classList.add("hidden");
  const info = data.info;
  $("set-title").textContent = `Calibration set ${day} ${cset}`;
  $("set-info").innerHTML = cards([["Snapshot", `<a href="/market?class=equity&date=${info.snapshot.date}&set=${info.snapshot.set}">`
    + `${info.snapshot.date} ${info.snapshot.set}</a>`], ["Created", info.created.replace("T", " ")],
    ["Models", info.models.map((m) => LABEL[m] || m).join(", ")], ["Underlyings", info.tickers.length],
    ["Time", `${fmt(info.seconds, 0)} s`]]);
  const cell = (t, m) => {
    const r = data.rows.find((x) => x.ticker === t && x.model === m);
    if (!r) return '<span class="muted">-</span>';
    if (r.status === "error") return `<span class="error" title="${(r.error || "").replace(/"/g, "'")}">error</span>`;
    return `<span style="color: ${QUALITY[r.quality]}; font-weight: 600">${r.quality}</span> `
      + `<span class="muted">${fmt(r.rms_volpts, 2)} / ${fmt(r.max_volpts, 1)} vol pts · ${fmt(r.seconds, 1)} s</span>`;
  };
  $("set-grid").innerHTML = table([["Underlying", (t) => `<b>${t}</b>`], ...data.models.map((m) => [LABEL[m] || m, (t) => cell(t, m)])],
    data.tickers, (t) => `class="clickable" data-ticker="${t}"`)
    + '<p class="muted">rms / max of model - market over the nodes 60%-150% of the spot; click an underlying.</p>';
  $("set-grid").querySelectorAll("tr.clickable").forEach((tr) => tr.addEventListener("click", () => openUnderlying(tr.dataset.ticker)));
  saveUrl();
  if (ticker) await openUnderlying(ticker);
}

// ---------- one underlying ----------
async function openUnderlying(ticker) {
  $("und-title").textContent = `${ticker}: loading…`;
  $("und-view").classList.remove("hidden");
  const d = await getJSON(`/api/calibration/underlying/${state.set.date}/${state.set.set}/${encodeURIComponent(ticker)}`);
  state.und = d;
  d.labels = expiryLabels(d.expiries, d.T);
  $("und-title").textContent = `${ticker} · spot ${fmt(d.spot, 2)}`;
  const ok = d.models.filter((m) => m.status === "ok");
  state.model = ok.length ? ok[0].model : d.models[0].model;
  $("und-models").innerHTML = d.models.map((m) => `<button class="secondary compact" data-model="${m.model}">`
    + `${LABEL[m.model] || m.model}</button>`).join("");
  $("und-models").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { state.model = b.dataset.model; showModel(); }));
  $("und-expiry").innerHTML = d.expiries.map((e, i) => `<option value="${i}">${d.labels[i]} (${e})</option>`).join("");
  $("und-expiry").value = String(d.T.reduce((b, t, i) => (Math.abs(t - 0.75) < Math.abs(d.T[b] - 0.75) ? i : b), 0));
  showModel();
  smile();
  atm();
  $("test-table").innerHTML = "";
  $("test-vols").innerHTML = "";
  $("test-status").textContent = "";
  saveUrl();
  loadTest(false);
}

function showModel() {
  const d = state.und;
  const m = d.models.find((x) => x.model === state.model);
  $("und-models").querySelectorAll("button").forEach((b) => b.classList.toggle("active-model", b.dataset.model === state.model));
  if (m.status !== "ok") {
    $("und-cards").innerHTML = cards([["Status", `<span class="error">error</span>`], ["Message", m.error || ""]]);
    ["und-params", "und-report", "und-errors"].forEach((id) => { $(id).innerHTML = ""; });
    return;
  }
  const drift = m.drift ? [["Drift on " + m.drift.snapshot, `${fmt(m.drift.rms_volpts, 2)} vol pts rms`
    + (m.drift.stale ? ' <span class="error">to recalibrate</span>' : ' <span style="color:#12805c">ok</span>')]] : [];
  $("und-cards").innerHTML = cards([["Fit", `<span style="color:${QUALITY[m.quality]}">${m.quality}</span>`],
    ["rms (vol pts)", fmt(m.rms_volpts, 2)], ["max (vol pts)", fmt(m.max_volpts, 1)], ["Calibration", `${fmt(m.seconds, 1)} s`],
    ["Jacobian", m.jacobian ? "stored" : "none"], ...drift]);
  const params = Object.entries(m.parameters || {});
  $("und-params").innerHTML = table([["Parameter", ([k]) => k], ["Value", ([, v]) => v.length === 1 ? fmt(v[0], 5)
    : v.map((x) => fmt(x, 4)).join(", ")]], params);
  const rep = m.report || [];
  const cols = rep.length ? Object.keys(rep[0]) : [];
  $("und-report").innerHTML = rep.length ? table(cols.map((c) => [c, (r) => (typeof r[c] === "number" ? fmt(r[c], 4) : r[c] ?? "")]), rep)
    : '<p class="muted">No report.</p>';
  const shade = (x) => (x >= 0 ? `rgba(53, 99, 233, ${Math.min(0.08 + 0.12 * x, 0.6)})` : `rgba(194, 54, 47, ${Math.min(0.08 + 0.12 * -x, 0.6)})`);
  const head = "<tr><th>Expiry</th>" + d.moneyness.map((x, j) => `<th>${(100 * x).toFixed(1).replace(/\.0$/, "")}%`
    + `<br><span class="muted">${fmt(d.strikes[j], d.spot < 50 ? 2 : d.spot < 1000 ? 1 : 0)}</span></th>`).join("") + "</tr>";
  const body = m.errors.map((row, i) => `<tr><td>${d.labels[i]} <span class="muted">${d.expiries[i]}</span></td>`
    + row.map((x) => `<td style="background:${shade(x)}">${fmt(x, 2)}</td>`).join("") + "</tr>").join("");
  $("und-errors").innerHTML = `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

function smile() {
  const d = state.und;
  const i = Number($("und-expiry").value);
  const x = d.moneyness.map((m) => 100 * m);
  const traces = [{ x, y: d.market[i].map((v) => 100 * v), mode: "lines+markers", name: "market surface",
    line: { color: "#888", dash: "dash" }, marker: { size: 5 } }];
  d.models.filter((m) => m.status === "ok").forEach((m) => traces.push({ x, y: m.vols[i].map((v) => 100 * v), mode: "lines",
    name: LABEL[m.model] || m.model, line: { color: MODEL_COLORS[m.model], width: 2 } }));
  const q = (d.quotes || {})[d.expiries[i]] || [];
  if (q.length) {
    traces.push({ x: q.map((p) => (100 * p.strike) / d.spot), y: q.map((p) => 100 * p.iv_mid), mode: "markers",
      name: "quotes (bid / ask)", marker: { color: "#18202f", size: 5 }, error_y: { type: "data", symmetric: false,
        array: q.map((p) => (p.iv_ask === null ? 0 : 100 * (p.iv_ask - p.iv_mid))),
        arrayminus: q.map((p) => (p.iv_bid === null ? 0 : 100 * (p.iv_mid - p.iv_bid))), thickness: 1.2, width: 3 } });
  }
  plot("chart-smile", traces, `Smile ${d.labels[i]} (${d.expiries[i]})`, "strike / spot (%)", "implied vol (%)");
}

function atm() {
  const d = state.und;
  const traces = [{ x: d.T, y: d.atm_market.map((v) => 100 * v), mode: "lines+markers", name: "market",
    line: { color: "#888", dash: "dash" } }];
  d.models.filter((m) => m.atm).forEach((m) => traces.push({ x: d.T, y: m.atm.map((v) => 100 * v), mode: "lines+markers",
    name: LABEL[m.model] || m.model, line: { color: MODEL_COLORS[m.model] }, marker: { size: 4 } }));
  plot("chart-atm", traces, "ATM-forward term structure", "T (years)", "implied vol (%)");
}

// ---------- model test ----------
async function loadTest(rerun) {
  const d = state.und;
  $("test-run").disabled = true;
  $("test-status").textContent = rerun ? "Running (about 10-20 s)…" : "Loading…";
  try {
    const t = await post("/api/calibration/test", { date: state.set.date, set: state.set.set, ticker: d.ticker, rerun });
    const engines = ["Black", ...t.engines];
    const price = (name) => [LABEL[name] || name, (r) => (r[name] === null ? "" : fmt(r[name], 3)
      + (r[`${name}_se`] ? ` <span class="muted">±${fmt(r[`${name}_se`], 3)}</span>` : ""))];
    const head = [["Product", (r) => r.product], ["Type", (r) => r.option_type], ["T", (r) => fmt(r.t, 1)],
      ["Strike %", (r) => fmt(r.strike_pct, 0)], ["Barrier %", (r) => (r.barrier_pct === null ? "" : fmt(r.barrier_pct, 0))]];
    $("test-table").innerHTML = table([...head, ...engines.map(price)], t.rows);
    const vols = t.rows.filter((r) => r.product === "Vanilla");
    $("test-vols").innerHTML = table([...head.slice(1, 4), ...engines.map((n) => [LABEL[n] || n,
      (r) => fmt(100 * r[`${n}_vol`], 2)])], vols);
    const secs = Object.entries(t.seconds || {}).map(([k, v]) => `${LABEL[k] || k} ${fmt(v, 1)} s`).join(", ");
    $("test-status").textContent = t.cached ? "(stored results; Run test to recompute)" : `done: ${secs}`;
  } catch (e) {
    $("test-status").innerHTML = `<span class="error">Error: ${e.message}</span>`;
  } finally {
    $("test-run").disabled = false;
  }
}

// ---------- universe ----------
const CLASS_LABEL = { equity: "Equity", rates: "Rates", fx: "FX" };
async function loadUniverse(cls) {
  const u = await getJSON("/api/calibration/universe");
  state.universe = u;
  state.uvClass = cls || state.uvClass || "equity";
  $("uv-classes").innerHTML = Object.keys(u.classes).map((c) => `<button data-cls="${c}" class="${c === state.uvClass ? "active" : ""}">`
    + `${CLASS_LABEL[c] || c} <span class="muted">${u.classes[c].length}</span></button>`).join("");
  $("uv-classes").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => loadUniverse(b.dataset.cls)));
  const models = u.models[state.uvClass];
  const rows = u.classes[state.uvClass];
  const supported = new Set(u.supported);
  const cell = (r, m) => `<label class="flag"><input type="checkbox" data-name="${r.name}" data-model="${m}"`
    + `${r.flags[m] ? " checked" : ""}></label>`;
  $("uv-table").innerHTML = !rows.length ? '<p class="muted">No underlying of this class in the latest snapshot.</p>'
    : !models.length ? table([["Underlying", (r) => `<b>${r.name}</b>`], ["Models", () => '<span class="muted">none yet</span>']], rows)
    : table([["Underlying", (r) => `<b>${r.name}</b>`], ...models.map((m) => [(LABEL[m] || m)
      + (supported.has(m) ? "" : ' <span class="muted">(not in the sets yet)</span>'), (r) => cell(r, m)])], rows);
  $("uv-note").textContent = state.uvClass === "equity"
    ? "Flags saved at once in marketdata/config/calibration.yaml; New calibration set calibrates the flagged models."
    : state.uvClass === "rates" ? "Rates models are listed but not calibrated in the sets yet: keep them off."
    : "No FX model yet.";
  $("uv-table").querySelectorAll("input[type=checkbox]").forEach((box) => box.addEventListener("change", async () => {
    const row = state.universe.classes[state.uvClass].find((r) => r.name === box.dataset.name);
    row.flags[box.dataset.model] = box.checked;
    try {
      await post("/api/calibration/universe", { asset_class: state.uvClass, name: row.name, flags: row.flags });
      $("uv-saved").textContent = `saved ${row.name}`;
      setTimeout(() => { $("uv-saved").textContent = ""; }, 1500);
    } catch (e) {
      $("uv-saved").innerHTML = `<span class="error">${e.message}</span>`;
    }
  }));
}
// Heston SLV: when to recalibrate (cheap indicators at once, the Monte Carlo fit on request)
async function loadSlv() {
  $("slv-table").innerHTML = '<p class="muted">Loading…</p>';
  const d = await getJSON("/api/calibration/slv");
  $("slv-latest").textContent = `vs latest snapshot ${d.latest}`;
  const r = d.rules;
  $("slv-rules").textContent = `Recalibrate when older than ${r.age_days} days, the spot moved more than ${r.spot_move_pct}%, `
    + `the surface moved more than ${r.surface_change_volpts} vol pt (rms at 3M-3Y, 80-120% of the spot) or the Monte `
    + `Carlo fit on the latest market is above ${r.fit_volpts} vol pt rms (Check fit: a few seconds).`;
  if (!d.rows.length) { $("slv-table").innerHTML = '<p class="muted">No Heston SLV calibration yet: flag it above and run a calibration set.</p>'; return; }
  $("slv-table").innerHTML = table([["Underlying", (x) => `<b>${x.ticker}</b>`], ["Calibrated", (x) => `${x.set} <span class="muted">(${x.age_days} d)</span>`],
    ["Spot", (x) => (x.spot_now === undefined ? "" : `${fmt(x.spot_then, 2)} → ${fmt(x.spot_now, 2)} <span class="muted">(${fmt(x.spot_move_pct, 1)}%)</span>`)],
    ["ATM 1Y vol", (x) => (x.atm1y_now === undefined ? "" : `${fmt(100 * x.atm1y_then, 1)} → ${fmt(100 * x.atm1y_now, 1)}`)],
    ["Surface change (vol pts)", (x) => fmt(x.surface_change_volpts, 2)], ["Heston part drift", (x) => fmt(x.heston_drift_volpts, 2)],
    ["SLV fit then", (x) => fmt(x.fit_at_calibration, 2)],
    ["SLV fit now", (x) => `<span id="slv-fit-${x.ticker}"></span> <button class="link" data-ticker="${x.ticker}">Check fit</button>`],
    ["Verdict", (x) => `<span id="slv-verdict-${x.ticker}" style="color:${x.verdict === "OK" ? "#12805c" : "#c2362f"}; font-weight:600">`
      + `${x.verdict}</span> <span class="muted">${x.reasons.join(", ")}</span>`]], d.rows);
  $("slv-table").querySelectorAll("button.link").forEach((b) => b.addEventListener("click", async () => {
    const t = b.dataset.ticker;
    $(`slv-fit-${t}`).textContent = "…";
    try {
      const c = await post("/api/calibration/slv/check", { ticker: t });
      $(`slv-fit-${t}`).innerHTML = `${fmt(c.fit_volpts, 2)} <span class="muted">max ${fmt(c.max_volpts, 1)}</span>`;
      if (c.recalibrate) {
        const v = $(`slv-verdict-${t}`);
        v.textContent = "Recalibrate";
        v.style.color = "#c2362f";
        v.insertAdjacentHTML("afterend", ' <span class="muted">fit above the threshold</span>');
      }
    } catch (e) {
      $(`slv-fit-${t}`).innerHTML = `<span class="error">${e.message}</span>`;
    }
  }));
}

function showTab(tab) {
  document.querySelectorAll("#subtabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  $("tab-sets").classList.toggle("hidden", tab !== "sets");
  $("tab-universe").classList.toggle("hidden", tab !== "universe");
  if (tab === "universe") { loadUniverse(); loadSlv(); }
}

// ---------- new set ----------
function showRun(run) {
  if (!run || (!run.running && !run.result && !run.exit_code)) { $("run-status").textContent = ""; return; }
  if (run.running) {
    $("run-status").textContent = `Calibrating… ${run.progress || ""} ${run.current || ""} (${fmt(run.seconds, 0)} s)`;
  } else if (run.exit_code === 0) {
    $("run-status").textContent = `Last run done in ${fmt(run.seconds, 0)} s`;
  } else {
    $("run-status").innerHTML = `<span class="error">Last run failed: ${(run.errors || []).slice(-1)[0] || ""}</span>`;
  }
}
async function pollRun() {
  const run = await getJSON("/api/calibration/run");
  showRun(run);
  if (run.running) setTimeout(pollRun, 3000);
  else if (run.exit_code === 0) loadSets();
}
async function fillNewForm() {
  const u = await getJSON("/api/calibration/universe");
  const plan = u.classes.equity.map((r) => [r.name, u.models.equity.filter((m) => r.flags[m])]).filter(([, m]) => m.length);
  $("nf-plan").innerHTML = plan.length ? "Flagged: " + plan.map(([t, m]) => `<b>${t}</b> ${m.map((x) => LABEL[x] || x).join(", ")}`).join(" · ")
    : '<span class="error">No model flagged in Universe.</span>';
  const snaps = (await getJSON("/api/snapshots")).snapshots;
  $("nf-date").innerHTML = snaps.map((s) => `<option>${s.date}</option>`).join("");
  const sets = () => { const s = snaps.find((x) => x.date === $("nf-date").value); $("nf-set").innerHTML = s.sets.map((x) => `<option>${x}</option>`).join(""); };
  $("nf-date").onchange = sets;
  sets();
}
async function startRun() {
  try {
    await post("/api/calibration/run", { date: $("nf-date").value, set: $("nf-set").value });     // the universe flags
    $("new-form").classList.add("hidden");
    pollRun();
  } catch (e) {
    $("run-status").innerHTML = `<span class="error">${e.message}</span>`;
  }
}

async function init() {
  $("new-set").addEventListener("click", () => { $("new-form").classList.toggle("hidden"); fillNewForm(); });
  $("nf-start").addEventListener("click", startRun);
  document.querySelectorAll("#subtabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  $("und-expiry").addEventListener("change", smile);
  $("test-run").addEventListener("click", () => loadTest(true));
  await loadSets();
  pollRun();
  const p = new URLSearchParams(location.search);
  const first = state.sets[0];
  if (p.get("date") && p.get("set")) await openSet(p.get("date"), p.get("set"), p.get("ticker"));
  else if (p.get("ticker") && first) await openSet(first.date, first.set, p.get("ticker"));
}

init();
