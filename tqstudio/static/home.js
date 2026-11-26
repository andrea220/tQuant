// tqstudio home: the documents, then the reference data of the market data (every instrument and curve, active or not, the model flags of the
// equities, its status in the latest snapshot), "Create new snapshot" (download the selected or every active
// instrument, optionally calibrate), and the documents.

const $ = (id) => document.getElementById(id);
const fmt = (x, d = 4) => (x === null || x === undefined ? "" : Number(x).toFixed(d));
const pct = (x, d = 2) => (x === null || x === undefined ? "" : (100 * x).toFixed(d));
const state = { cls: "equity", data: null, selected: new Set() };

function table(columns, rows) {
  const head = "<tr>" + columns.map(([h]) => `<th>${h}</th>`).join("") + "</tr>";
  const body = rows.map((r) => "<tr>" + columns.map(([, f]) => `<td>${f(r)}</td>`).join("") + "</tr>").join("");
  return `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

async function postJson(url, body) {
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body) });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

// ---------- universe ----------
const dash = (x, f) => (x === null || x === undefined ? "–" : f(x));
const when = (r) => (r.last_download ? r.last_download.replace("T", " ").slice(0, 16) : "never");
const ADD_HINT = { equity: "Bloomberg ticker, e.g. UCG IM Equity or FTSEMIB Index",
                   rates: "Bloomberg fixing, e.g. EUR003M Index", fx: "Bloomberg pair, e.g. EURJPY Curncy" };

function renderUniverse() {
  const d = state.data;
  if (!d) return;
  document.querySelectorAll("#uv-classes button").forEach((b) => b.classList.toggle("active", b.dataset.cls === state.cls));
  const sorted = (rows) => [...rows].sort((a, b) => (b.active - a.active) || a.bbg.localeCompare(b.bbg));
  const lock = d.can_edit ? "" : " disabled";
  const pick = (name) => `<input type="checkbox" class="uv-select" data-name="${name}"${
    state.selected.has(name) ? " checked" : ""}>`;
  const flag = (name, on) => `<input type="checkbox" class="uv-active" data-name="${name}"${on ? " checked" : ""}${lock}>`;
  const lead = [["Select", (r) => pick(r.bbg)], ["Active", (r) => flag(r.bbg, r.active)]];
  const models = (d.models || []).map((m) => [m.replace("DisplacedDiffusion", "DD").replace("HestonSLV", "SLV")
    .replace("LocalVol", "LV"),
    (r) => `<input type="checkbox" class="uv-model" data-code="${r.code}" data-model="${m}"${
      (r.flags || {})[m] ? " checked" : ""}${lock}>`]);
  $("uv-ticker").placeholder = ADD_HINT[state.cls];
  $("uv-table2").innerHTML = "";
  if (state.cls === "equity") {
    $("uv-table").innerHTML = table([
      ...lead,
      ["Bloomberg ticker", (r) => r.bbg],
      ["Type", (r) => r.type],
      ["Ccy", (r) => r.ccy || ""],
      ["Last download", when],
      ["Settings", (r) => [r.option_root ? `options ${r.option_root}` : "", r.implied_spot ? "implied spot" : ""]
        .filter(Boolean).join(", ")],
      ...models,
    ], sorted(d.equity));
  } else if (state.cls === "rates") {
    $("uv-table").innerHTML = table([
      ["Active", (r) => flag(r.name, r.active)],
      ["Curve", (r) => r.name], ["Bloomberg (ICVS)", (r) => r.bbg], ["Kind", (r) => r.kind.toUpperCase()],
      ["Last download", when], ["Quotes", (r) => r.quotes], ["Last update", (r) => r.last_update || ""],
      ["Bootstrapped", (r) => (r.bootstrapped ? `${r.pillars} pillars` : "–")],
    ], [...d.curves].sort((a, b) => a.name.localeCompare(b.name)));
    $("uv-table2").innerHTML = table([
      ["Active", (r) => flag(r.bbg, r.active)], ["Fixing", (r) => r.bbg], ["Last download", when],
      ["Value (%)", (r) => dash(r.value, (x) => pct(x, 3))], ["Date", (r) => r.fixing_date || ""],
    ], sorted(d.fixing));
  } else {
    $("uv-table").innerHTML = table([
      ["Active", (r) => flag(r.bbg, r.active)], ["Bloomberg ticker", (r) => r.bbg],
      ["Last download", when], ["Spot", (r) => dash(r.spot, (x) => fmt(x, 4))],
    ], sorted(d.fx));
  }
  document.querySelectorAll("input.uv-select").forEach((b) => b.addEventListener("change", () => {
    if (b.checked) state.selected.add(b.dataset.name); else state.selected.delete(b.dataset.name);
  }));
  document.querySelectorAll("input.uv-active").forEach((b) =>
    b.addEventListener("change", () => setActive(b.dataset.name, b.checked)));
  document.querySelectorAll("input.uv-model").forEach((b) =>
    b.addEventListener("change", () => setModel(b.dataset.code)));
  $("uv-add").classList.toggle("hidden", !d.can_edit);
  $("snap-new").classList.toggle("hidden", d.source !== "live");     // pressed where not allowed: the popup says why
  $("uv-note").textContent = d.can_download
    ? "Select picks equities for a selected snapshot (the active rates, FX and curves are always downloaded); "
      + "Active ones make up a full snapshot. The model flags say what Calibrate calibrates."
    : d.source !== "live" ? "Snapshots are created with the live data (python -m tqstudio)."
      : !d.server_authorized
        ? "This tqstudio runs on a pc without a Bloomberg terminal: snapshots are created from the tqstudio of an "
          + "authorized pc (authorized_pcs in tqstudio/config.yaml); they appear here."
        : "The universe can be edited and snapshots created only from the authorized pcs (authorized_pcs in "
          + "tqstudio/config.yaml).";
}

async function loadUniverse() {
  try {
    const res = await fetch("/api/market/universe");
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    state.data = data;
    $("snapshot").textContent = data.snapshot ? `${data.source} · ${data.snapshot.date} · ${data.snapshot.set}`
      : `${data.source} · no snapshot`;
    renderUniverse();
    showDownload(data.download);
  } catch (e) {
    $("uv-table").innerHTML = `<span class="error">Error: ${e.message}</span>`;
  }
}

async function setActive(name, active) {
  try {
    await postJson("/api/market/universe/active", { name, active });
    $("dl-status").textContent = `${name} ${active ? "active: downloaded by a full snapshot" : "inactive"}`;
  } catch (e) {
    $("dl-status").innerHTML = `<span class="error">${e.message}</span>`;
  }
  await loadUniverse();
}

async function setModel(code) {
  const flags = {};
  document.querySelectorAll(`input.uv-model[data-code="${code}"]`).forEach((b) => { flags[b.dataset.model] = b.checked; });
  try {
    await postJson("/api/calibration/universe", { asset_class: "equity", name: code, flags });
  } catch (e) {
    $("dl-status").innerHTML = `<span class="error">${e.message}</span>`;
  }
  await loadUniverse();
}

function showAddForm(show) {
  $("uv-add-form").classList.toggle("hidden", !show);
  $("uv-add-msg").textContent = "";
  if (show) { $("uv-ticker").value = ""; $("uv-ticker").focus(); }
}

async function addTicker() {
  const ticker = $("uv-ticker").value.trim();
  if (!ticker) return;
  $("uv-add-msg").textContent = "Checking on Bloomberg…";
  try {
    const data = await postJson("/api/market/universe", { asset_class: state.cls, ticker });
    showAddForm(false);
    await loadUniverse();
    $("dl-status").textContent = `${data.ticker}${data.name ? ` (${data.name}, ${data.ccy})` : ""} `
      + `${data.action === "activated" ? "active again in" : "added to"} the reference data: downloaded by the next full snapshot`;
  } catch (e) {
    $("uv-add-msg").innerHTML = `<span class="error">${e.message}</span>`;
  }
}

// ---------- Create new snapshot ----------
function selectedEquities() {
  const eq = (state.data && state.data.equity) || [];
  return eq.filter((r) => state.selected.has(r.bbg)).map((r) => r.bbg);
}

function openNewSnapshot() {
  const picked = selectedEquities();
  $("ns-selected-label").textContent = `Download only the selected equities (${picked.length})`
    + (picked.length ? `: ${picked.join(", ")}` : "");
  $("ns-selected").disabled = !picked.length;
  $(picked.length ? "ns-selected" : "ns-all").checked = true;
  $("ns-msg").textContent = "";
  $("new-snapshot").classList.remove("hidden");
}

function closeNewSnapshot() { $("new-snapshot").classList.add("hidden"); }

async function createSnapshot() {
  const selected = $("ns-selected").checked ? selectedEquities() : null;
  const body = { calibrate: $("ns-calibrate").checked, ...(selected ? { select: selected } : {}) };
  try {
    const run = await postJson("/api/market/download", body);
    closeNewSnapshot();
    showDownload(run);
    setTimeout(pollDownload, 3000);
  } catch (e) {                               // not started: not authorized, no Python with blpapi, already running
    closeNewSnapshot();
    showDownloadError(e.message);
  }
}

// download errors: a popup, closed once (the run it belongs to is remembered: it does not come back on a reload)
const DISMISSED_KEY = "tqstudio.downloadErrorDismissed";
function dismissedRun() {
  try { return localStorage.getItem(DISMISSED_KEY); } catch (e) { return state.dismissedRun || null; }
}
function rememberDismissed(id) {
  state.dismissedRun = id;
  try { localStorage.setItem(DISMISSED_KEY, id); } catch (e) { /* private window: this page only */ }
}

function showDownloadError(message, log = [], runId = null) {
  const terminal = /ConnectionError|Bloomberg session|terminal running/i.test(message);
  const denied = /^Not authorized/.test(message);
  $("dl-error-title").textContent = denied ? "Not authorized"
    : terminal ? "Bloomberg terminal not reachable" : "Download failed";
  $("dl-error-text").textContent = denied ? message.replace(/^Not authorized:\s*/, "")
    : terminal
      ? `${message.replace(/^\w*Error:\s*/, "")}. Open the Bloomberg terminal on this pc, log in, then create the snapshot again.`
      : message;
  $("dl-error-log").textContent = log.join("\n");
  $("dl-error-details").classList.toggle("hidden", !log.length);
  $("dl-error").dataset.run = runId || "";
  $("dl-error").classList.remove("hidden");
}

function closeDownloadError() {
  const box = $("dl-error");
  if (box.classList.contains("hidden")) return;
  box.classList.add("hidden");
  if (box.dataset.run) rememberDismissed(box.dataset.run);
}

function showDownload(run) {
  const log = $("dl-log");
  $("snap-new").disabled = Boolean(run && run.running);
  log.classList.add("hidden");
  if (!run || (!run.running && run.exit_code === undefined)) return;
  const what = run.select ? `selected (${run.select.join(", ")})` : "all active";
  if (run.running) {
    $("dl-status").textContent = `Downloading ${what}… started ${run.started.slice(11)} (${fmt(run.seconds, 0)} s)`
      + (run.calibrate ? ", then calibration" : "");
    log.textContent = (run.log || []).join("\n");
    log.classList.toggle("hidden", !(run.log || []).length);
  } else if (run.exit_code === 0) {
    $("dl-status").textContent = `Last snapshot (${what}) done in ${fmt(run.seconds, 0)} s: ${run.result || ""}`
      + (run.after ? ` · ${run.after}` : "");
  } else {
    $("dl-status").textContent = "";
    if (run.started !== dismissedRun()) {
      showDownloadError((run.errors || []).slice(-1)[0] || `the download ended with exit code ${run.exit_code}`,
        run.log || [], run.started);
    }
  }
}

async function pollDownload() {
  const run = await (await fetch("/api/market/download")).json();
  showDownload(run);
  if (run.running) { setTimeout(pollDownload, 3000); return; }
  if (run.exit_code === 0) await loadUniverse();             // the new snapshot is the latest one
}

// ---------- documents ----------
async function loadDocuments() {
  const { documents } = await (await fetch("/api/documents")).json();
  $("documents").innerHTML = documents.map((d) => d.available
    ? `<a class="tile" href="/documents/${d.key}" target="_blank" rel="noopener">
         <div class="tile-title">${d.title} <span class="pill">PDF</span></div></a>`
    : `<div class="tile disabled"><div class="tile-title">${d.title}</div>
         <div class="tile-text">Not built on this machine: run <code>latexmk</code> in <code>study/docs</code>.</div></div>`
  ).join("");
}

async function init() {
  document.querySelectorAll("#uv-classes button").forEach((b) => b.addEventListener("click", () => {
    state.cls = b.dataset.cls; renderUniverse(); }));
  $("uv-add").addEventListener("click", () => showAddForm($("uv-add-form").classList.contains("hidden")));
  $("uv-add-ok").addEventListener("click", addTicker);
  $("uv-add-cancel").addEventListener("click", () => showAddForm(false));
  $("uv-ticker").addEventListener("keydown", (e) => { if (e.key === "Enter") addTicker(); });
  $("snap-new").addEventListener("click", openNewSnapshot);
  $("ns-start").addEventListener("click", createSnapshot);
  $("ns-cancel").addEventListener("click", closeNewSnapshot);
  $("new-snapshot").addEventListener("click", (e) => { if (e.target === $("new-snapshot")) closeNewSnapshot(); });
  $("dl-error-close").addEventListener("click", closeDownloadError);
  $("dl-error").addEventListener("click", (e) => { if (e.target === $("dl-error")) closeDownloadError(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") { closeDownloadError(); closeNewSnapshot(); } });
  loadDocuments();
  await loadUniverse();
  const run = await (await fetch("/api/market/download")).json();          // a download started earlier
  if (run.running) pollDownload();
}

init();
