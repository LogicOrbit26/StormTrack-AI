// 1. Storm Datasets (Historical & AI Trajectory Data)
const storms = {
  AMPHAN: {
    name: "AMPHAN",
    basin: "Bay of Bengal",
    center: [16.5, 87.2],
    track: [
      { time: "2020-05-16 06:00 UTC", lat: 10.4, lon: 87.0, wind: 45, pressure: 996, stage: "Cyclonic Storm" },
      { time: "2020-05-16 18:00 UTC", lat: 11.2, lon: 86.7, wind: 65, pressure: 988, stage: "Severe Cyclonic Storm" },
      { time: "2020-05-17 06:00 UTC", lat: 12.0, lon: 86.5, wind: 85, pressure: 978, stage: "Very Severe Cyclonic Storm" },
      { time: "2020-05-17 18:00 UTC", lat: 13.1, lon: 86.3, wind: 110, pressure: 960, stage: "Extremely Severe Cyclonic Storm" },
      { time: "2020-05-18 06:00 UTC", lat: 14.3, lon: 86.4, wind: 130, pressure: 932, stage: "Super Cyclonic Storm" },
      { time: "2020-05-18 18:00 UTC", lat: 15.6, lon: 86.7, wind: 140, pressure: 920, stage: "Super Cyclonic Storm" },
      { time: "2020-05-19 06:00 UTC", lat: 17.1, lon: 87.1, wind: 135, pressure: 925, stage: "Super Cyclonic Storm" },
      { time: "2020-05-19 18:00 UTC", lat: 18.8, lon: 87.6, wind: 120, pressure: 942, stage: "Extremely Severe Cyclonic Storm" },
      { time: "2020-05-20 06:00 UTC", lat: 20.6, lon: 88.1, wind: 105, pressure: 954, stage: "Very Severe Cyclonic Storm" },
      { time: "2020-05-20 12:00 UTC", lat: 21.9, lon: 88.4, wind: 95, pressure: 962, stage: "Landfall - Sundarbans" }
    ],
    districts: [
      { name: "South 24 Parganas", state: "West Bengal", dist: "18 km", eta: "14 hrs", risk: "High Risk" },
      { name: "Purba Medinipur", state: "West Bengal", dist: "42 km", eta: "16 hrs", risk: "High Risk" },
      { name: "Balasore", state: "Odisha", dist: "95 km", eta: "12 hrs", risk: "Moderate Watch" },
      { name: "Kendrapara", state: "Odisha", dist: "130 km", eta: "10 hrs", risk: "Moderate Watch" },
      { name: "Visakhapatnam", state: "Andhra Pradesh", dist: "460 km", eta: "Passed", risk: "Low Risk" }
    ],
    riTrend: "+45 knots",
    riProb: "84.2% (EXTREME)",
    sst: "30.8°C (Favorable > 28.5°C)",
    shear: "6 knots (Very Low Shear)"
  },
  BIPARJOY: {
    name: "BIPARJOY",
    basin: "Arabian Sea",
    center: [19.5, 67.8],
    track: [
      { time: "2023-06-11 06:00 UTC", lat: 17.5, lon: 67.3, wind: 85, pressure: 976, stage: "Very Severe Cyclonic Storm" },
      { time: "2023-06-12 06:00 UTC", lat: 18.6, lon: 67.5, wind: 105, pressure: 962, stage: "Extremely Severe Cyclonic Storm" },
      { time: "2023-06-13 06:00 UTC", lat: 19.8, lon: 67.7, wind: 100, pressure: 966, stage: "Very Severe Cyclonic Storm" },
      { time: "2023-06-14 06:00 UTC", lat: 21.1, lon: 68.0, wind: 90, pressure: 972, stage: "Very Severe Cyclonic Storm" },
      { time: "2023-06-15 12:00 UTC", lat: 22.8, lon: 68.6, wind: 75, pressure: 980, stage: "Landfall - Jakhau Port" }
    ],
    districts: [
      { name: "Kutch (Mandvi/Jakhau)", state: "Gujarat", dist: "12 km", eta: "18 hrs", risk: "High Risk" },
      { name: "Devbhumi Dwarka", state: "Gujarat", dist: "55 km", eta: "14 hrs", risk: "High Risk" },
      { name: "Jamnagar", state: "Gujarat", dist: "98 km", eta: "20 hrs", risk: "Moderate Watch" },
      { name: "Porbandar", state: "Gujarat", dist: "145 km", eta: "12 hrs", risk: "Moderate Watch" }
    ],
    riTrend: "+30 knots",
    riProb: "72.0% (HIGH)",
    sst: "29.9°C (Warm Arabian Sea)",
    shear: "9 knots (Low Shear)"
  },
  FANI: {
    name: "FANI",
    basin: "Bay of Bengal",
    center: [16.8, 85.2],
    track: [
      { time: "2019-04-30 06:00 UTC", lat: 13.4, lon: 84.5, wind: 80, pressure: 980, stage: "Very Severe Cyclonic Storm" },
      { time: "2019-05-01 06:00 UTC", lat: 15.1, lon: 84.3, wind: 105, pressure: 964, stage: "Extremely Severe Cyclonic Storm" },
      { time: "2019-05-02 06:00 UTC", lat: 17.2, lon: 84.9, wind: 125, pressure: 940, stage: "Extremely Severe Cyclonic Storm" },
      { time: "2019-05-03 03:00 UTC", lat: 19.7, lon: 85.8, wind: 115, pressure: 950, stage: "Landfall - Puri Coast" }
    ],
    districts: [
      { name: "Puri", state: "Odisha", dist: "8 km", eta: "10 hrs", risk: "High Risk" },
      { name: "Khordha (Bhubaneswar)", state: "Odisha", dist: "38 km", eta: "12 hrs", risk: "High Risk" },
      { name: "Ganjam", state: "Odisha", dist: "75 km", eta: "8 hrs", risk: "Moderate Watch" },
      { name: "Srikakulam", state: "Andhra Pradesh", dist: "160 km", eta: "Passed", risk: "Low Risk" }
    ],
    riTrend: "+40 knots",
    riProb: "81.0% (HIGH)",
    sst: "30.5°C (Favorable > 28.5°C)",
    shear: "7 knots (Low Shear)"
  },
  MICHAUNG: {
    name: "MICHAUNG",
    basin: "Bay of Bengal",
    center: [13.5, 81.2],
    track: [
      { time: "2023-12-03 06:00 UTC", lat: 11.4, lon: 82.1, wind: 50, pressure: 994, stage: "Cyclonic Storm" },
      { time: "2023-12-04 06:00 UTC", lat: 13.1, lon: 80.7, wind: 60, pressure: 988, stage: "Severe Cyclonic Storm" },
      { time: "2023-12-05 06:00 UTC", lat: 15.6, lon: 80.3, wind: 55, pressure: 990, stage: "Landfall - Bapatla" }
    ],
    districts: [
      { name: "Bapatla", state: "Andhra Pradesh", dist: "15 km", eta: "12 hrs", risk: "High Risk" },
      { name: "Chennai", state: "Tamil Nadu", dist: "45 km", eta: "Now", risk: "High Risk" },
      { name: "Nellore", state: "Andhra Pradesh", dist: "60 km", eta: "6 hrs", risk: "Moderate Watch" }
    ],
    riTrend: "+15 knots",
    riProb: "44.0% (MODERATE)",
    sst: "29.1°C",
    shear: "14 knots (Moderate Shear)"
  },
  TAUKTAE: {
    name: "TAUKTAE",
    basin: "Arabian Sea",
    center: [17.8, 71.5],
    track: [
      { time: "2021-05-15 06:00 UTC", lat: 14.5, lon: 72.8, wind: 75, pressure: 982, stage: "Very Severe Cyclonic Storm" },
      { time: "2021-05-16 06:00 UTC", lat: 17.2, lon: 71.9, wind: 100, pressure: 965, stage: "Extremely Severe Cyclonic Storm" },
      { time: "2021-05-17 06:00 UTC", lat: 19.8, lon: 71.2, wind: 115, pressure: 950, stage: "Extremely Severe Cyclonic Storm" },
      { time: "2021-05-17 18:00 UTC", lat: 20.9, lon: 71.1, wind: 100, pressure: 962, stage: "Landfall - Una, Gujarat" }
    ],
    districts: [
      { name: "Gir Somnath (Una)", state: "Gujarat", dist: "10 km", eta: "11 hrs", risk: "High Risk" },
      { name: "Amreli", state: "Gujarat", dist: "35 km", eta: "13 hrs", risk: "High Risk" },
      { name: "Mumbai Suburban", state: "Maharashtra", dist: "140 km", eta: "Passed", risk: "Moderate Watch" }
    ],
    riTrend: "+35 knots",
    riProb: "76.8% (HIGH)",
    sst: "30.2°C",
    shear: "8 knots (Low Shear)"
  }
};

// 2. Initialize Leaflet Satellite-Style Dark Map
const map = L.map('map').setView([16.5, 86.5], 5);
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; OpenStreetMap &copy; CARTO'
}).addTo(map);

let mapLayers = L.layerGroup().addTo(map);
let currentStormKey = "AMPHAN";
let currentIdx = 6;
let activeLayerColor = "#ef4444";

// 3. Multi-Language Broadcast Templates
const translations = {
  en: (s, pt) => `URGENT CYCLONE WARNING: ${s.name} (${pt.stage}) located at ${pt.lat}°N, ${pt.lon}°E with winds of ${pt.wind} knots (${Math.round(pt.wind * 1.852)} km/h). Move immediately to designated cyclone shelters.`,
  hi: (s, pt) => `अत्यावश्यक चक्रवात चेतावनी: चक्रवात ${s.name} (${pt.lat}°N, ${pt.lon}°E) ${Math.round(pt.wind * 1.852)} किमी/घंटा की हवाओं के साथ आगे बढ़ रहा है। तटीय निवासी तुरंत सुरक्षित आश्रयों में जाएं।`,
  or: (s, pt) => `ଜରୁରୀ ବାତ୍ୟା ସତର୍କତା: ବାତ୍ୟା ${s.name} (${pt.lat}°N, ${pt.lon}°E) ଘଣ୍ଟାପ୍ରତି ${Math.round(pt.wind * 1.852)} କିମି ବେଗରେ ମାଡ଼ି ଆସୁଛି। ତୁରନ୍ତ ବାତ୍ୟା ଆଶ୍ରୟସ୍ଥଳୀକୁ ଯାଆନ୍ତୁ।`,
  bn: (s, pt) => `জরুরি ঘূর্ণিঝড় সতর্কতা: ঘূর্ণিঝড় ${s.name} (${pt.lat}°N, ${pt.lon}°E) ঘণ্টায় ${Math.round(pt.wind * 1.852)} কিমি বেগে উপকূলের দিকে অগ্রসর হচ্ছে। অবিলম্বে নিরাপদ আশ্রয়কেন্দ্রে যান।`
};

// 4. Render Entire Dashboard State
function updateDashboard() {
  const storm = storms[currentStormKey];
  const slider = document.getElementById('timeline-slider');
  slider.max = storm.track.length;
  if (currentIdx >= storm.track.length) currentIdx = storm.track.length - 1;
  slider.value = currentIdx + 1;

  document.getElementById('point-counter').textContent = `Point ${currentIdx + 1} / ${storm.track.length}`;

  const pt = storm.track[currentIdx];
  const kmh = Math.round(pt.wind * 1.852);

  // Update Top KPIs
  document.getElementById('kpi-time').textContent = pt.time;
  document.getElementById('kpi-wind').textContent = `${pt.wind} knots (${kmh} km/h)`;
  document.getElementById('kpi-pressure').textContent = `${pt.pressure} hPa`;
  document.getElementById('kpi-stage').textContent = pt.stage;

  // Update Alert Banner
  const banner = document.getElementById('alert-banner');
  const alertText = document.getElementById('alert-text');
  if (pt.wind >= 90) {
    banner.className = "alert-banner alert-high";
    alertText.innerHTML = `<strong>🚨 HIGH SEVERITY ALERT:</strong> ${pt.stage} active. Catastrophic wind damage (${kmh} km/h) and extreme storm surge expected near landfall.`;
  } else {
    banner.className = "alert-banner alert-moderate";
    alertText.innerHTML = `<strong>⚠️ CYCLONE WATCH:</strong> ${pt.stage} active (${kmh} km/h). Coastal authorities advised to maintain high readiness.`;
  }

  // Update Map Tracks & Concentric Uncertainty Rings
  mapLayers.clearLayers();
  const pastCoords = storm.track.slice(0, currentIdx + 1).map(p => [p.lat, p.lon]);
  const futureCoords = storm.track.slice(currentIdx).map(p => [p.lat, p.lon]);

  if (pastCoords.length > 1) {
    L.polyline(pastCoords, { color: '#38bdf8', weight: 4 }).addTo(mapLayers);
  }
  if (futureCoords.length > 1) {
    L.polyline(futureCoords, { color: '#f97316', weight: 3.5, dashArray: '8, 8' }).addTo(mapLayers);
  }

  // Concentric Wind & Uncertainty Cones along future points
  storm.track.slice(currentIdx).forEach((fp, i) => {
    L.circle([fp.lat, fp.lon], {
      radius: 75000 + i * 45000,
      color: activeLayerColor,
      fillColor: activeLayerColor,
      fillOpacity: 0.16,
      weight: 1.5
    }).addTo(mapLayers);
  });

  // Current Eye Marker
  L.circleMarker([pt.lat, pt.lon], {
    radius: 9,
    color: '#ffffff',
    fillColor: activeLayerColor,
    fillOpacity: 1,
    weight: 3
  }).addTo(mapLayers).bindPopup(`<b>${storm.name}</b><br>${pt.stage}<br>Wind: ${kmh} km/h`);

  map.setView([pt.lat, pt.lon], 5);

  // Update 24-Hour Projections Table
  const projTbody = document.getElementById('projections-tbody');
  projTbody.innerHTML = "";
  const horizons = ["+06 Hours", "+12 Hours", "+18 Hours", "+24 Hours"];
  horizons.forEach((h, i) => {
    const fLat = (pt.lat + (i + 1) * 0.85).toFixed(2);
    const fLon = (pt.lon + (i + 1) * 0.25).toFixed(2);
    const fWind = Math.max(45, pt.wind - i * 5);
    const fPres = pt.pressure + i * 4;
    projTbody.innerHTML += `
      <tr>
        <td class="mono">${h}</td>
        <td class="mono">${fLat}° N</td>
        <td class="mono">${fLon}° E</td>
        <td class="mono">${fWind} kt (${Math.round(fWind * 1.852)} km/h)</td>
        <td class="mono">${fPres} hPa</td>
        <td>${fWind >= 90 ? "Extremely Severe" : "Very Severe"}</td>
      </tr>
    `;
  });

  // Update Coastal District Vulnerability Table
  const vulnTbody = document.getElementById('vulnerability-tbody');
  vulnTbody.innerHTML = storm.districts.map(d => {
    const pillClass = d.risk.includes("High") ? "risk-high" : d.risk.includes("Moderate") ? "risk-mod" : "risk-low";
    return `
      <tr>
        <td><strong>${d.name}</strong></td>
        <td>${d.state}</td>
        <td class="mono">${d.dist}</td>
        <td class="mono">${d.eta}</td>
        <td><span class="risk-pill ${pillClass}">${d.risk}</span></td>
      </tr>
    `;
  }).join("");

  // Update Broadcast Text & RI Box
  const lang = document.getElementById('lang-select').value;
  document.getElementById('broadcast-text').textContent = translations[lang](storm, pt);
  document.getElementById('ri-trend').textContent = storm.riTrend;
  document.getElementById('ri-prob').textContent = storm.riProb;
  document.getElementById('ri-sst').textContent = storm.sst;
  document.getElementById('ri-shear').textContent = storm.shear;
}

// 5. Event Listeners
document.getElementById('storm-select').addEventListener('change', (e) => {
  currentStormKey = e.target.value;
  currentIdx = Math.floor(storms[currentStormKey].track.length / 2);
  updateDashboard();
});

document.getElementById('timeline-slider').addEventListener('input', (e) => {
  currentIdx = parseInt(e.target.value, 10) - 1;
  updateDashboard();
});

document.getElementById('lang-select').addEventListener('change', updateDashboard);

document.querySelectorAll('.fusion-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.fusion-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const layer = btn.getAttribute('data-layer');
    activeLayerColor = layer === "sar" ? "#ef4444" : layer === "ir" ? "#a855f7" : "#10b981";
    updateDashboard();
  });
});

// Download / Print Official IMD Bulletin
document.getElementById('download-bulletin-btn').addEventListener('click', () => {
  window.print();
});

// Emergency Siren + Speech Synthesis Voice Broadcast
document.getElementById('play-siren-btn').addEventListener('click', () => {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(520, ctx.currentTime);
    osc.frequency.linearRampToValueAtTime(880, ctx.currentTime + 0.6);
    osc.frequency.linearRampToValueAtTime(520, ctx.currentTime + 1.2);
    gain.gain.setValueAtTime(0.15, ctx.currentTime);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 1.3);
  } catch (err) {}

  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();
    const msg = new SpeechSynthesisUtterance(document.getElementById('broadcast-text').textContent);
    setTimeout(() => window.speechSynthesis.speak(msg), 1350);
  }
});

// Initial Render
updateDashboard();
