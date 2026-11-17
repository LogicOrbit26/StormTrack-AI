"use strict";

/* =====================================================================
   CycloneAI frontend  ->  FastAPI backend integration
   Backend: http://127.0.0.1:8000   (uvicorn main:app --reload --port 8000)

   Endpoints used
     GET  /storms                                   -> storm list + tracks (cached once)
     GET  /vulnerability/{storm}/{step_index}       -> district risk table + overall risk
     GET  /alert-translate?stage&wind_speed_kmh&lang-> localized broadcast text + severity
     GET  /generate-bulletin/{storm}/{step_index}   -> PDF download
     GET  /generate-audio?storm_name&step_index&lang-> MP3 voice broadcast
     POST /predict-image (multipart field "file")   -> satellite image analysis
   ===================================================================== */

// ---------------------------------------------------------------------
// 1. Config, state and small helpers
// ---------------------------------------------------------------------
const API_BASE = "http://127.0.0.1:8000";
const KMH_PER_KNOT = 1.852;
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024; // same limit as the backend

const state = {
  storms: {},          // { "Fani": {...storm from GET /storms} }
  key: null,           // selected storm name (exact backend spelling)
  idx: 0,              // selected step_index (0-based, matches backend)
  layerColor: "#ef4444",
  vuln: null,          // last GET /vulnerability response
  alert: null,         // last GET /alert-translate response
};

const $ = (id) => document.getElementById(id);
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const kmhToKt = (kmh) => Math.round(kmh / KMH_PER_KNOT);

// Escape text before putting it into innerHTML templates.
const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// "2019-05-03T00:00:00Z" -> "2019-05-03 00:00 UTC"
const fmtTime = (iso) => iso.replace("T", " ").replace(/:\d\d(\.\d+)?Z$/, " UTC");

const currentStorm = () => state.storms[state.key] || null;
const currentPoint = () => {
  const s = currentStorm();
  return s ? s.trajectory[state.idx] : null;
};

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

// ---------------------------------------------------------------------
// 2. API layer
// ---------------------------------------------------------------------
class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}

// FastAPI errors are {"detail": "..."} or {"detail": [{msg: "..."}]} (422)
async function readErrorMessage(res) {
  try {
    const body = await res.clone().json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d) => d.msg).join("; ");
  } catch (_) { /* body was not JSON */ }
  return `HTTP ${res.status} ${res.statusText}`;
}

async function apiFetch(path, { params, method = "GET", body } = {}) {
  const url = new URL(API_BASE + path);
  if (params) Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v));
  let res;
  try {
    res = await fetch(url, { method, body });
  } catch (_) {
    throw new ApiError(`Cannot reach the backend at ${API_BASE}. Is uvicorn running?`, 0);
  }
  if (!res.ok) throw new ApiError(await readErrorMessage(res), res.status);
  return res;
}

const apiJson = async (path, opts) => (await apiFetch(path, opts)).json();

// ---------------------------------------------------------------------
// 3. Map setup (unchanged look: dark Carto tiles)
// ---------------------------------------------------------------------
const map = L.map("map").setView([16.5, 86.5], 5);
L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}", {
  attribution: "Tiles &copy; Esri",
  maxZoom: 16,
}).addTo(map);
const mapLayers = L.layerGroup().addTo(map);

// ---------------------------------------------------------------------
// 4. Rendering that only needs the cached GET /storms data (instant)
// ---------------------------------------------------------------------
function setBanner(cssClass, title, text) {
  const banner = $("alert-banner");
  banner.className = `alert-banner ${cssClass}`;
  const strong = document.createElement("strong");
  strong.textContent = title;
  const span = document.createElement("span");
  span.id = "alert-text";
  span.textContent = " " + text;
  banner.replaceChildren(strong, span);
}

function renderMap(storm, idx) {
  const traj = storm.trajectory;
  mapLayers.clearLayers();

  // Blue = track up to the selected point, orange dashed = selected point + next 4 steps
  const past = traj.slice(0, idx + 1).map((p) => [p.lat, p.lon]);
  const ahead = traj.slice(idx, idx + 5);

  if (past.length > 1) L.polyline(past, { color: "#38bdf8", weight: 4 }).addTo(mapLayers);
  if (ahead.length > 1) {
    L.polyline(ahead.map((p) => [p.lat, p.lon]),
      { color: "#f97316", weight: 3.5, dashArray: "8, 8" }).addTo(mapLayers);
  }

  // Uncertainty cone: real per-step radius from the backend (metres)
  ahead.forEach((p) => {
    L.circle([p.lat, p.lon], {
      radius: p.uncertainty_radius_m,
      color: state.layerColor,
      fillColor: state.layerColor,
      fillOpacity: 0.16,
      weight: 1.5,
    }).addTo(mapLayers);
  });

  // Landfall marker (storm metadata from the backend)
  if (storm.landfall) {
    L.circleMarker([storm.landfall.lat, storm.landfall.lon], {
      radius: 6, color: "#ffffff", fillColor: "#facc15", fillOpacity: 1, weight: 2,
    }).addTo(mapLayers).bindTooltip(`Landfall: ${storm.landfall.place}`);
  }

  // Current eye marker
  const pt = traj[idx];
  L.circleMarker([pt.lat, pt.lon], {
    radius: 9, color: "#ffffff", fillColor: state.layerColor, fillOpacity: 1, weight: 3,
  }).addTo(mapLayers).bindPopup(
    `<b>${esc(storm.name)}</b><br>${esc(pt.imd_category)}<br>Wind: ${pt.wind_speed_kmh} km/h`
  );

  map.setView([pt.lat, pt.lon], 5);
}

function renderProjections(traj, idx) {
  const cur = traj[idx];
  const next = traj.slice(idx + 1, idx + 5);
  const tbody = $("projections-tbody");
  if (!next.length) {
    tbody.innerHTML = `<tr><td colspan="6">End of track: no further forecast positions.</td></tr>`;
    return;
  }
  tbody.innerHTML = next.map((p) => `
    <tr>
      <td class="mono">+${p.hours_since_start - cur.hours_since_start} Hours${p.is_forecast ? "" : " (obs)"}</td>
      <td class="mono">${p.lat.toFixed(2)}° N</td>
      <td class="mono">${p.lon.toFixed(2)}° E</td>
      <td class="mono">${p.wind_speed_kmh} km/h (${kmhToKt(p.wind_speed_kmh)} kt)</td>
      <td class="mono">${p.pressure_hpa} hPa</td>
      <td>${esc(p.stage)} · ${esc(p.imd_category)}</td>
    </tr>`).join("");
}

// Wind change versus the nearest earlier step that is >= 12 h before the selected one
function intensityTrend(traj, idx) {
  const cur = traj[idx];
  let ref = null;
  for (let i = idx - 1; i >= 0; i--) {
    if (cur.hours_since_start - traj[i].hours_since_start >= 12) { ref = traj[i]; break; }
  }
  if (!ref && idx > 0) ref = traj[idx - 1];
  if (!ref) return null;
  return {
    hours: cur.hours_since_start - ref.hours_since_start,
    delta: cur.wind_speed_kmh - ref.wind_speed_kmh,
  };
}

const EYEWALL_BY_STAGE = {
  Formation: "Disorganised convection, no eye (inferred from stage)",
  Developing: "Partial banding, no closed eye-wall (inferred from stage)",
  Intensifying: "Eye-wall forming / contracting (inferred from stage)",
  Mature: "Closed concentric ring expected (inferred from stage)",
  Weakening: "Eye-wall eroding, open ring (inferred from stage)",
};

function renderRI(traj, idx) {
  const pt = traj[idx];

  const trend = intensityTrend(traj, idx);
  $("ri-trend-label").textContent = trend ? `${trend.hours}h Intensity Trend` : "Intensity Trend";
  $("ri-trend").textContent = trend ? `${trend.delta >= 0 ? "+" : ""}${trend.delta} km/h` : "N/A (first point)";

  const pct = pt.ri_probability * 100;
  const level = pt.ri_probability >= 0.6 ? "HIGH" : pt.ri_probability >= 0.3 ? "MODERATE" : "LOW";
  const probEl = $("ri-prob");
  probEl.textContent = `${pct.toFixed(1)}% (${level})`;
  probEl.className = "mono " + (level === "HIGH" ? "text-red" : level === "MODERATE" ? "text-amber" : "text-green");

  $("ri-sst").textContent = pt.sst_c >= 28.5
    ? `${pt.sst_c}°C (Favorable > 28.5°C)`
    : `${pt.sst_c}°C (Below 28.5°C threshold, less favorable)`;

  const shearNote = pt.wind_shear_kt < 10 ? "Low Shear - Supports Deepening"
    : pt.wind_shear_kt < 20 ? "Moderate Shear" : "High Shear - Inhibits Intensification";
  $("ri-shear").textContent = `${pt.wind_shear_kt} knots (${shearNote})`;

  $("ri-eye").textContent = EYEWALL_BY_STAGE[pt.stage] || "—";
}

function renderLocal() {
  const storm = currentStorm();
  if (!storm) return;
  const traj = storm.trajectory;
  state.idx = clamp(state.idx, 0, traj.length - 1);
  const pt = traj[state.idx];

  // Slider + ticks follow the real number of backend steps
  const slider = $("timeline-slider");
  slider.min = 1;
  slider.max = traj.length;
  slider.value = state.idx + 1;
  $("tick-start").textContent = "1";
  $("tick-mid").textContent = String(Math.ceil(traj.length / 2));
  $("tick-end").textContent = String(traj.length);
  $("point-counter").textContent =
    `Point ${state.idx + 1} / ${traj.length}${pt.is_forecast ? " · Forecast" : ""}`;

  // KPIs
  $("kpi-time").textContent = fmtTime(pt.timestamp);
  $("kpi-wind").textContent = `${pt.wind_speed_kmh} km/h (${kmhToKt(pt.wind_speed_kmh)} kt)`;
  $("kpi-pressure").textContent = `${pt.pressure_hpa} hPa`;
  $("kpi-stage").textContent = `${pt.stage} (${pt.imd_category})`;

  renderMap(storm, state.idx);
  renderProjections(traj, state.idx);
  renderRI(traj, state.idx);
  renderBanner();
}

// ---------------------------------------------------------------------
// 5. Panels that come from the backend (vulnerability, alert, banner)
// ---------------------------------------------------------------------
function renderBanner() {
  const pt = currentPoint();
  const storm = currentStorm();
  if (!pt || !storm) return;

  const sev = state.alert && state.alert.severity;
  const risk = state.vuln && state.vuln.overall_risk;
  const base = `${pt.imd_category} ${storm.name} active (${pt.wind_speed_kmh} km/h).`;
  const riskText = risk ? ` Regional coastal risk: ${risk}.` : "";

  if (sev === "danger") {
    const tail = risk === "Low Risk"
      ? "No monitored coastal district is within the damaging-wind zone at this point."
      : "Catastrophic wind damage and extreme storm surge possible near landfall.";
    setBanner("alert-high", "🚨 HIGH SEVERITY ALERT:", `${base}${riskText} ${tail}`);
  } else if (sev === "warning") {
    setBanner("alert-moderate", "⚠️ CYCLONE WARNING:",
      `${base}${riskText} Coastal authorities advised to maintain high readiness.`);
  } else if (sev === "advisory") {
    setBanner("alert-moderate", "ℹ️ CYCLONE ADVISORY:", `${base}${riskText} Continue to monitor official updates.`);
  } else {
    setBanner("alert-moderate", "⏳ UPDATING:", "Fetching alert level from the backend…");
  }
}

function etaText(d) {
  if (d.eta_hours === null) return "Not approaching";
  if (d.eta_hours < 1) return "< 1 hr";
  return `${d.eta_hours} hrs`;
}

function renderVulnerability(v) {
  $("vulnerability-tbody").innerHTML = v.districts.map((d) => {
    const pill = d.risk.includes("High") ? "risk-high" : d.risk.includes("Moderate") ? "risk-mod" : "risk-low";
    return `
      <tr>
        <td><strong>${esc(d.district)}</strong></td>
        <td>${esc(d.state)}</td>
        <td class="mono">${d.distance_km} km ${esc(d.storm_direction_from_district)}</td>
        <td class="mono">${etaText(d)}</td>
        <td><span class="risk-pill ${pill}">${esc(d.risk)}</span></td>
      </tr>`;
  }).join("");
}

let vulnSeq = 0;
async function refreshVulnerability() {
  const storm = currentStorm();
  if (!storm) return;
  const seq = ++vulnSeq; // ignore out-of-order responses while the slider is moving
  const tbody = $("vulnerability-tbody");
  if (!state.vuln) tbody.innerHTML = `<tr><td colspan="5">Loading vulnerability data…</td></tr>`;
  try {
    const v = await apiJson(`/vulnerability/${encodeURIComponent(storm.name)}/${state.idx}`);
    if (seq !== vulnSeq) return;
    state.vuln = v;
    renderVulnerability(v);
    renderBanner();
  } catch (err) {
    if (seq !== vulnSeq) return;
    tbody.innerHTML = `<tr><td colspan="5">⚠️ ${esc(err.message)}</td></tr>`;
  }
}

let alertSeq = 0;
async function refreshAlert() {
  const pt = currentPoint();
  if (!pt) return;
  const seq = ++alertSeq;
  const box = $("broadcast-text");
  try {
    const a = await apiJson("/alert-translate", {
      params: { stage: pt.stage, wind_speed_kmh: pt.wind_speed_kmh, lang: $("lang-select").value },
    });
    if (seq !== alertSeq) return;
    state.alert = a;
    box.textContent = a.message;
    renderBanner();
  } catch (err) {
    if (seq !== alertSeq) return;
    box.textContent = `⚠️ ${err.message}`;
  }
}

// One debounced entry point so dragging the slider doesn't flood the API
const refreshBackendPanels = debounce(() => { refreshVulnerability(); refreshAlert(); }, 150);

// ---------------------------------------------------------------------
// 6. Bulletin download (GET /generate-bulletin)
// ---------------------------------------------------------------------
function setBusy(btn, busy, busyLabel) {
  if (busy) {
    btn.dataset.label = btn.textContent;
    btn.textContent = busyLabel;
    btn.disabled = true;
  } else {
    btn.textContent = btn.dataset.label || btn.textContent;
    btn.disabled = false;
  }
}

function setStatus(id, text, isError = false) {
  const el = $(id);
  el.textContent = text;
  el.classList.toggle("error", isError);
}

async function downloadBulletin() {
  const storm = currentStorm();
  if (!storm) return;
  const btn = $("download-bulletin-btn");
  setBusy(btn, true, "⏳ Generating PDF bulletin…");
  setStatus("bulletin-status", "");
  try {
    const res = await apiFetch(`/generate-bulletin/${encodeURIComponent(storm.name)}/${state.idx}`);
    const blob = await res.blob();
    const match = /filename="?([^";]+)"?/i.exec(res.headers.get("Content-Disposition") || "");
    const filename = match ? match[1] : `IMD_Bulletin_${storm.name}_step${state.idx}.pdf`;

    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
    setStatus("bulletin-status", `✅ Downloaded ${filename}`);
  } catch (err) {
    setStatus("bulletin-status", `⚠️ ${err.message}`, true);
  } finally {
    setBusy(btn, false);
  }
}

// ---------------------------------------------------------------------
// 7. Siren + server-generated voice broadcast (GET /generate-audio)
// ---------------------------------------------------------------------
let activeAudio = null;
let activeAudioUrl = null;

function stopAudio() {
  if (activeAudio) { activeAudio.pause(); activeAudio = null; }
  if (activeAudioUrl) { URL.revokeObjectURL(activeAudioUrl); activeAudioUrl = null; }
}

// Same WebAudio siren as before; resolves when it has finished (~1.3 s)
function playSiren() {
  return new Promise((resolve) => {
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sawtooth";
      osc.frequency.setValueAtTime(520, ctx.currentTime);
      osc.frequency.linearRampToValueAtTime(880, ctx.currentTime + 0.6);
      osc.frequency.linearRampToValueAtTime(520, ctx.currentTime + 1.2);
      gain.gain.setValueAtTime(0.15, ctx.currentTime);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 1.3);
      osc.onended = () => { ctx.close(); resolve(); };
    } catch (_) { resolve(); }
    setTimeout(resolve, 1500); // safety net if onended never fires
  });
}

async function playBroadcast() {
  const storm = currentStorm();
  if (!storm) return;
  const btn = $("play-siren-btn");
  const lang = $("lang-select").value;
  stopAudio();
  setBusy(btn, true, "⏳ Generating voice broadcast…");
  setStatus("audio-status", "");
  try {
    // Fetch the MP3 while the siren plays, so there is no extra waiting
    const [res] = await Promise.all([
      apiFetch("/generate-audio", {
        params: { storm_name: storm.name, step_index: state.idx, lang },
      }),
      playSiren(),
    ]);
    const blob = await res.blob();
    activeAudioUrl = URL.createObjectURL(blob);
    activeAudio = new Audio(activeAudioUrl);
    activeAudio.addEventListener("ended", () => {
      setStatus("audio-status", "Broadcast finished.");
      stopAudio();
    });
    await activeAudio.play();

    // gTTS has no Odia voice: the backend falls back to English
    const fellBack = res.headers.get("X-TTS-Language-Fallback") || lang === "or";
    setStatus("audio-status", fellBack && lang === "or"
      ? "🔊 Playing. Odia voice is not supported by gTTS, so the broadcast is in English."
      : "🔊 Playing voice broadcast…");
  } catch (err) {
    const blocked = err.name === "NotAllowedError";
    setStatus("audio-status",
      blocked ? "⚠️ Browser blocked autoplay. Click the button again." : `⚠️ ${err.message}`, true);
  } finally {
    setBusy(btn, false);
  }
}

// ---------------------------------------------------------------------
// 8. Satellite image upload (POST /predict-image)
// ---------------------------------------------------------------------
let selectedFile = null;
let previewUrl = null;

function showInference(html, isError = false) {
  const box = $("inference-result");
  box.hidden = false;
  box.classList.toggle("error", isError);
  box.innerHTML = html;
}

function onImageSelected() {
  const file = $("image-input").files[0];
  $("inference-result").hidden = true;
  if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null; }

  if (!file) { selectedFile = null; $("analyze-image-btn").disabled = true; return; }

  if (!file.type.startsWith("image/")) {
    selectedFile = null;
    $("analyze-image-btn").disabled = true;
    showInference("Please choose an image file (JPG / PNG).", true);
    return;
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    selectedFile = null;
    $("analyze-image-btn").disabled = true;
    showInference("Image is larger than 10 MB.", true);
    return;
  }

  selectedFile = file;
  previewUrl = URL.createObjectURL(file);
  const preview = $("image-preview");
  preview.src = previewUrl;
  preview.hidden = false;
  $("upload-filename").textContent = `🖼️ ${file.name}`;
  $("upload-box").classList.add("has-file");
  $("analyze-image-btn").disabled = false;
}

function renderInference(r) {
  const row = (k, v) => `<div class="row"><span>${k}</span><strong class="mono">${esc(v)}</strong></div>`;
  const conf = `${(r.confidence * 100).toFixed(1)}%`;
  $("ai-confidence").textContent = conf;
  showInference(
    row("Cyclone detected", r.cyclone_detected ? "YES" : "NO") +
    row("Stage", r.stage || "N/A") +
    row("Confidence", conf) +
    row("Est. wind speed", `${r.estimated_wind_speed_kmh} km/h`) +
    row("IMD category", r.imd_category) +
    row("Model", r.model)
  );
}

async function analyzeImage() {
  if (!selectedFile) return;
  const btn = $("analyze-image-btn");
  setBusy(btn, true, "⏳ Analysing image…");
  try {
    const form = new FormData();
    form.append("file", selectedFile); // field name must be "file"
    const result = await apiJson("/predict-image", { method: "POST", body: form });
    renderInference(result);
  } catch (err) {
    showInference(`⚠️ ${esc(err.message)}`, true);
  } finally {
    setBusy(btn, false);
    btn.disabled = !selectedFile;
  }
}

// ---------------------------------------------------------------------
// 9. Event listeners
// ---------------------------------------------------------------------
$("storm-select").addEventListener("change", (e) => {
  state.key = e.target.value;
  const storm = currentStorm();
  state.idx = storm.forecast_start_index - 1;   // last observed point = "now"
  state.vuln = null;
  state.alert = null;
  renderLocal();
  refreshBackendPanels();
});

$("timeline-slider").addEventListener("input", (e) => {
  state.idx = parseInt(e.target.value, 10) - 1;
  renderLocal();            // instant (local data)
  refreshBackendPanels();   // debounced (API)
});

$("lang-select").addEventListener("change", refreshAlert);

document.querySelectorAll(".fusion-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".fusion-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const layer = btn.getAttribute("data-layer");
    state.layerColor = layer === "sar" ? "#ef4444" : layer === "ir" ? "#a855f7" : "#10b981";
    renderLocal();
  });
});

$("download-bulletin-btn").addEventListener("click", downloadBulletin);
$("play-siren-btn").addEventListener("click", playBroadcast);
$("image-input").addEventListener("change", onImageSelected);
$("analyze-image-btn").addEventListener("click", analyzeImage);

// ---------------------------------------------------------------------
// 10. Startup: load all storms once from GET /storms
// ---------------------------------------------------------------------
async function init() {
  try {
    const data = await apiJson("/storms");
    state.storms = {};
    data.storms.forEach((s) => { state.storms[s.name] = s; }); // keyed by exact backend name

    // Rebuild the dropdown from the backend so names always match exactly
    const select = $("storm-select");
    select.innerHTML = "";
    data.storms.forEach((s) => {
      const opt = document.createElement("option");
      opt.value = s.name;
      opt.textContent = `${s.name} (${s.year} - ${s.basin})`;
      select.appendChild(opt);
    });

    state.key = data.storms[0].name;
    select.value = state.key;
    state.idx = currentStorm().forecast_start_index - 1;

    renderLocal();
    refreshVulnerability();
    refreshAlert();
  } catch (err) {
    setBanner("alert-high", "❌ BACKEND UNAVAILABLE:",
      `${err.message} Start it with "uvicorn main:app --reload --port 8000" inside the backend folder, then reload this page.`);
  }
}

init();