// 1. Initialize Dark Command-Center Map centered on Bay of Bengal
const map = L.map('map', {
  zoomControl: true
}).setView([15.8, 84.8], 6);

L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; OpenStreetMap &copy; CARTO',
  maxZoom: 18
}).addTo(map);

// 2. Time-Series Cyclone Dataset (Past Observations + AI Forecast)
// Later, your teammates' Python backend will send an array just like this!
const stormTimeline = [
  { time: "T-36h (OBSERVED)", lat: 12.1, lon: 88.5, wind: 110, pressure: 982, cat: "CATEGORY 1", desc: "Severe Cyclonic Storm", conf: "98.1%" },
  { time: "T-24h (OBSERVED)", lat: 13.3, lon: 87.4, wind: 145, pressure: 968, cat: "CATEGORY 2", desc: "Very Severe Cyclonic Storm", conf: "97.5%" },
  { time: "T-12h (OBSERVED)", lat: 14.5, lon: 86.3, wind: 185, pressure: 954, cat: "CATEGORY 3", desc: "Very Severe Cyclonic Storm", conf: "97.0%" },
  { time: "T+00h (CURRENT EYE)", lat: 15.8, lon: 85.4, wind: 215, pressure: 942, cat: "CATEGORY 4", desc: "Extremely Severe Cyclonic Storm", conf: "96.4%" },
  { time: "T+12h (AI FORECAST)", lat: 17.0, lon: 84.6, wind: 225, pressure: 936, cat: "CATEGORY 4", desc: "Peak Intensity Approaching Coast", conf: "91.8%" },
  { time: "T+24h (AI FORECAST)", lat: 18.2, lon: 83.9, wind: 195, pressure: 950, cat: "CATEGORY 3", desc: "Landfall Imminent near Visakhapatnam", conf: "86.2%" },
  { time: "T+48h (AI FORECAST)", lat: 19.6, lon: 83.1, wind: 120, pressure: 978, cat: "CATEGORY 1", desc: "Overland Weakening Phase", conf: "79.5%" }
];

// 3. Draw Historical Track (Indices 0 to 3) & Predicted Track (Indices 3 to 6)
const pastCoords = stormTimeline.slice(0, 4).map(p => [p.lat, p.lon]);
const futureCoords = stormTimeline.slice(3).map(p => [p.lat, p.lon]);

L.polyline(pastCoords, {
  color: '#38bdf8',
  weight: 3.5,
  opacity: 0.9
}).addTo(map);

L.polyline(futureCoords, {
  color: '#ef4444',
  weight: 3.5,
  dashArray: '8, 8',
  opacity: 0.9
}).addTo(map);

// Plot small nodes along the track
stormTimeline.forEach((pt, idx) => {
  L.circleMarker([pt.lat, pt.lon], {
    radius: 4,
    color: idx <= 3 ? '#38bdf8' : '#ef4444',
    fillColor: '#0f172a',
    fillOpacity: 1,
    weight: 2
  }).addTo(map).bindTooltip(`${pt.time}: ${pt.wind} km/h`);
});

// 4. Create Pulsing Cyclone Eye Marker & Wind Radii Zones
const eyeIcon = L.divIcon({
  className: 'cyclone-eye-icon',
  html: '<div class="eye-ring"></div><div class="eye-core"></div>',
  iconSize: [48, 48],
  iconAnchor: [24, 24]
});

let currentPoint = stormTimeline[3];
const eyeMarker = L.marker([currentPoint.lat, currentPoint.lon], { icon: eyeIcon }).addTo(map);

// Outer Gale-Force Wind Ring (Orange)
const galeRing = L.circle([currentPoint.lat, currentPoint.lon], {
  radius: 220000, // 220 km
  color: '#f97316',
  fillColor: '#f97316',
  fillOpacity: 0.08,
  weight: 1.5,
  dashArray: '4, 4'
}).addTo(map);

// Inner Hurricane-Force Wind Ring (Red)
const hurricaneRing = L.circle([currentPoint.lat, currentPoint.lon], {
  radius: 95000, // 95 km
  color: '#ef4444',
  fillColor: '#ef4444',
  fillOpacity: 0.22,
  weight: 2
}).addTo(map);

// 5. Interactive Timeline Scrubber Logic
const slider = document.getElementById('time-slider');
slider.addEventListener('input', (e) => {
  const index = parseInt(e.target.value, 10);
  const data = stormTimeline[index];

  // Move Marker & Wind Rings
  const newLatLng = [data.lat, data.lon];
  eyeMarker.setLatLng(newLatLng);
  galeRing.setLatLng(newLatLng);
  hurricaneRing.setLatLng(newLatLng);

  // Scale wind ring radius dynamically based on wind speed
  hurricaneRing.setRadius(data.wind * 450);
  galeRing.setRadius(data.wind * 1050);

  // Update Sidebar Telemetry Readouts
  document.getElementById('time-label').textContent = data.time;
  document.getElementById('storm-category').textContent = data.cat;
  document.getElementById('storm-desc').textContent = data.desc;
  document.getElementById('confidence').textContent = `${data.conf} CONF`;
  document.getElementById('wind').innerHTML = `${data.wind} <small>km/h</small>`;
  document.getElementById('pressure').innerHTML = `${data.pressure} <small>hPa</small>`;
  document.getElementById('lat').textContent = `${data.lat.toFixed(2)}° N`;
  document.getElementById('lon').textContent = `${data.lon.toFixed(2)}° E`;

  map.panTo(newLatLng, { animate: true });
});

// 6. Multi-Source Satellite Sensor Mode Switcher
const sensorColors = {
  sar: { inner: '#ef4444', outer: '#f97316' }, // Radar Red/Orange
  ir:  { inner: '#a855f7', outer: '#38bdf8' }, // Thermal Purple/Cyan
  wv:  { inner: '#10b981', outer: '#06b6d4' }  // Moisture Emerald/Teal
};

document.querySelectorAll('.sensor-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.sensor-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    const mode = btn.getAttribute('data-mode');
    const palette = sensorColors[mode];

    hurricaneRing.setStyle({ color: palette.inner, fillColor: palette.inner });
    galeRing.setStyle({ color: palette.outer, fillColor: palette.outer });
  });
});
