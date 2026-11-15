"""
CycloneAI - AI Cyclone Intelligence & Early Warning System (educational mini project)
=====================================================================================

A single-file FastAPI backend that serves:

    GET  /storms                                   historical + 24h forecast tracks (5 cyclones)
    POST /predict-image                            satellite image -> cyclone stage (placeholder model)
    GET  /vulnerability/{storm}/{step_index}       distance / ETA / risk for coastal districts
    GET  /alert-translate                          localized emergency messages (en, hi, or, bn)
    GET  /generate-bulletin/{storm}/{step_index}   IMD-style PDF warning bulletin (fpdf2)
    GET  /generate-audio                           MP3 voice broadcast (gTTS)

HOW TO RUN
----------
1. Create and activate a virtual environment (recommended):

       python -m venv .venv
       source .venv/bin/activate          # Windows: .venv\\Scripts\\activate

2. Install dependencies:

       pip install fastapi "uvicorn[standard]" python-multipart "fpdf2>=2.7.6" gTTS Pillow

3. Start the server from inside the `backend/` folder:

       cd backend
       uvicorn main:app --reload --host 0.0.0.0 --port 8000

4. Open the interactive API docs (Swagger UI) in the browser:

       http://127.0.0.1:8000/docs

NOTES
-----
* /generate-audio needs internet access (gTTS calls Google's TTS service).
* All cyclone tracks are APPROXIMATE, simplified reconstructions for teaching.
  The 24-hour "forecast" segment is the last 4 six-hourly steps of each series.
  Nothing produced by this project is an official IMD product.
"""

from __future__ import annotations

import math
import random
import zlib
from datetime import datetime, timedelta, timezone
from enum import Enum
from io import BytesIO
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from fpdf.fonts import FontFace
from PIL import Image, ImageStat, UnidentifiedImageError

try:  # gTTS is imported lazily so the rest of the API still works without it
    from gtts import gTTS
    from gtts.lang import tts_langs
except ImportError:  # pragma: no cover
    gTTS = None
    tts_langs = None


# =============================================================================
# App setup
# =============================================================================

app = FastAPI(
    title="CycloneAI - Cyclone Intelligence & Early Warning System",
    description="Educational backend for cyclone tracking, risk assessment, "
    "multilingual alerts, PDF bulletins and voice broadcasts.",
    version="1.0.0",
)

# CORS: allow every origin (fine for a college project / local front-end dev).
# NOTE: wildcard origins cannot be combined with credentials, so they stay off.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],  # lets JS read the download filename
)


# =============================================================================
# Shared enums & helpers
# =============================================================================

class Stage(str, Enum):
    Formation = "Formation"
    Developing = "Developing"
    Intensifying = "Intensifying"
    Mature = "Mature"
    Weakening = "Weakening"


class Lang(str, Enum):
    en = "en"
    hi = "hi"
    or_ = "or"  # Python keyword workaround; the value is still "or"
    bn = "bn"


IST = timezone(timedelta(hours=5, minutes=30))


def imd_category(wind_kmh: float) -> str:
    """IMD classification by 3-minute sustained wind speed (km/h)."""
    if wind_kmh < 31:
        return "Low Pressure Area"
    if wind_kmh < 50:
        return "Depression"
    if wind_kmh < 62:
        return "Deep Depression"
    if wind_kmh < 89:
        return "Cyclonic Storm"
    if wind_kmh < 118:
        return "Severe Cyclonic Storm"
    if wind_kmh < 166:
        return "Very Severe Cyclonic Storm"
    if wind_kmh < 222:
        return "Extremely Severe Cyclonic Storm"
    return "Super Cyclonic Storm"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial compass bearing from point 1 to point 2 (0 = north, clockwise)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlmb = math.radians(lon2 - lon1)
    y = math.sin(dlmb) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlmb)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


_COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def compass(deg: float) -> str:
    return _COMPASS[int((deg + 11.25) // 22.5) % 16]


def _latin1(text: str) -> str:
    """fpdf2's built-in fonts only support Latin-1; make text safe."""
    return text.encode("latin-1", "replace").decode("latin-1")


# =============================================================================
# Storm dataset (simplified, approximate best-track style data)
# =============================================================================
# Each point: (hours_since_start, lat, lon, wind_kmh, pressure_hpa, stage)
# The LAST 4 points of every storm are the 6-hourly 24-hour FORECAST segment;
# everything before them is "historical" (observed analysis).

FORECAST_STEPS = 4

STORM_DEFS: dict[str, dict[str, Any]] = {
    "Amphan": {
        "basin": "Bay of Bengal",
        "year": 2020,
        "start_utc": datetime(2020, 5, 16, 12, tzinfo=timezone.utc),
        "landfall": {"place": "Sundarbans coast, West Bengal", "lat": 21.7, "lon": 88.3,
                     "time_utc": "2020-05-20T10:30:00Z"},
        "summary": "Rapidly intensified into a Super Cyclonic Storm in the Bay of Bengal "
                   "before striking the West Bengal-Bangladesh coast.",
        "points": [
            (0, 10.6, 87.6, 55, 1002, "Formation"),
            (12, 11.5, 87.4, 75, 996, "Developing"),
            (24, 12.6, 87.3, 120, 980, "Intensifying"),
            (36, 13.8, 87.4, 200, 940, "Intensifying"),
            (48, 15.2, 87.6, 240, 912, "Mature"),
            (60, 17.1, 87.7, 230, 920, "Mature"),
            (72, 19.2, 87.8, 190, 940, "Mature"),
            # ---- 24-hour forecast ----
            (78, 19.9, 88.0, 175, 948, "Mature"),
            (84, 20.7, 88.1, 165, 953, "Mature"),
            (90, 21.2, 88.2, 155, 957, "Weakening"),
            (96, 21.7, 88.3, 150, 960, "Weakening"),
        ],
    },
    "Biparjoy": {
        "basin": "Arabian Sea",
        "year": 2023,
        "start_utc": datetime(2023, 6, 6, 12, tzinfo=timezone.utc),
        "landfall": {"place": "Jakhau, Kutch, Gujarat", "lat": 23.2, "lon": 68.7,
                     "time_utc": "2023-06-15T15:00:00Z"},
        "summary": "A long-lived Arabian Sea cyclone that stayed intense for days "
                   "before landfall near the Kutch coast.",
        "points": [
            (0, 11.8, 66.2, 55, 1002, "Formation"),
            (24, 12.8, 66.6, 85, 992, "Developing"),
            (48, 14.2, 66.9, 150, 968, "Intensifying"),
            (72, 15.7, 67.2, 170, 958, "Intensifying"),
            (96, 17.3, 67.3, 195, 940, "Mature"),
            (120, 19.2, 67.0, 190, 945, "Mature"),
            (144, 20.7, 66.8, 185, 950, "Mature"),
            (168, 21.6, 67.0, 170, 955, "Mature"),
            (192, 22.2, 67.6, 150, 965, "Mature"),
            # ---- 24-hour forecast ----
            (198, 22.5, 67.9, 145, 967, "Mature"),
            (204, 22.8, 68.2, 140, 970, "Weakening"),
            (210, 23.0, 68.5, 135, 972, "Weakening"),
            (216, 23.2, 68.7, 125, 975, "Weakening"),
        ],
    },
    "Fani": {
        "basin": "Bay of Bengal",
        "year": 2019,
        "start_utc": datetime(2019, 4, 26, 12, tzinfo=timezone.utc),
        "landfall": {"place": "Puri, Odisha", "lat": 19.8, "lon": 85.8,
                     "time_utc": "2019-05-03T02:30:00Z"},
        "summary": "Extremely Severe Cyclonic Storm that made landfall near Puri, Odisha, "
                   "triggering one of India's largest cyclone evacuations.",
        "points": [
            (0, 5.5, 87.5, 45, 1004, "Formation"),
            (24, 7.0, 87.0, 65, 998, "Developing"),
            (48, 9.0, 85.8, 85, 990, "Developing"),
            (72, 11.0, 85.2, 110, 980, "Intensifying"),
            (96, 13.2, 84.5, 150, 965, "Intensifying"),
            (108, 14.5, 84.2, 175, 950, "Intensifying"),
            (120, 16.0, 84.4, 195, 940, "Mature"),
            (132, 17.5, 84.9, 215, 930, "Mature"),
            # ---- 24-hour forecast ----
            (138, 18.1, 85.2, 215, 933, "Mature"),
            (144, 18.8, 85.4, 210, 938, "Mature"),
            (150, 19.4, 85.6, 200, 945, "Mature"),
            (156, 19.8, 85.8, 185, 950, "Mature"),
        ],
    },
    "Mocha": {
        "basin": "Bay of Bengal",
        "year": 2023,
        "start_utc": datetime(2023, 5, 9, 12, tzinfo=timezone.utc),
        "landfall": {"place": "Sittwe, Rakhine coast, Myanmar", "lat": 20.2, "lon": 93.0,
                     "time_utc": "2023-05-14T09:30:00Z"},
        "summary": "Intense Bay of Bengal cyclone that tracked east-north-east and "
                   "hit Myanmar, sparing the Indian coast.",
        "points": [
            (0, 10.3, 87.6, 45, 1004, "Formation"),
            (12, 11.0, 87.8, 55, 1000, "Developing"),
            (24, 11.8, 88.0, 65, 998, "Developing"),
            (48, 13.5, 88.4, 85, 992, "Developing"),
            (72, 15.0, 88.9, 130, 975, "Intensifying"),
            (84, 16.2, 89.7, 165, 960, "Intensifying"),
            (96, 17.3, 90.4, 195, 945, "Mature"),
            (108, 18.4, 91.3, 210, 937, "Mature"),
            # ---- 24-hour forecast ----
            (114, 19.0, 91.9, 205, 940, "Mature"),
            (120, 19.8, 92.7, 195, 945, "Mature"),
            (126, 20.4, 93.2, 150, 960, "Weakening"),
            (132, 21.2, 93.7, 100, 980, "Weakening"),
        ],
    },
    "Tauktae": {
        "basin": "Arabian Sea",
        "year": 2021,
        "start_utc": datetime(2021, 5, 14, 12, tzinfo=timezone.utc),
        "landfall": {"place": "Near Diu / Una, Gujarat", "lat": 20.8, "lon": 71.0,
                     "time_utc": "2021-05-17T17:00:00Z"},
        "summary": "Extremely Severe Cyclonic Storm that ran up India's west coast "
                   "and struck Gujarat's Saurashtra coast.",
        "points": [
            (0, 11.0, 71.9, 50, 1002, "Formation"),
            (12, 12.0, 72.2, 65, 996, "Developing"),
            (24, 13.5, 72.2, 85, 990, "Developing"),
            (36, 14.8, 72.2, 110, 980, "Intensifying"),
            (48, 16.1, 71.9, 140, 968, "Intensifying"),
            (60, 17.6, 71.5, 175, 955, "Intensifying"),
            # ---- 24-hour forecast ----
            (66, 18.3, 71.3, 185, 950, "Mature"),
            (72, 19.5, 71.0, 200, 945, "Mature"),
            (78, 20.5, 71.0, 185, 950, "Mature"),
            (84, 21.5, 71.4, 150, 965, "Weakening"),
        ],
    },
}

_STAGE_RI_BONUS = {"Formation": 0.0, "Developing": 0.6, "Intensifying": 1.2,
                   "Mature": -1.5, "Weakening": -3.0}
_STAGE_BASE_SHEAR = {"Formation": 8, "Developing": 7, "Intensifying": 5,
                     "Mature": 8, "Weakening": 16}


def _environment(rng: random.Random, lat: float, stage: str) -> tuple[float, float]:
    """Heuristic SST (deg C) and vertical wind shear (knots) for a track point."""
    sst = 30.4 - 0.10 * max(0.0, lat - 12.0) + rng.uniform(-0.3, 0.3)
    shear = _STAGE_BASE_SHEAR[stage] + rng.uniform(-2.0, 2.0)
    if stage == "Weakening":
        sst -= 1.5  # cooler water / land interaction
    return round(sst, 1), round(max(shear, 1.0), 1)


def _ri_probability(wind: float, sst: float, shear: float, stage: str) -> float:
    """
    Toy Rapid-Intensification (RI) probability: logistic of SST, shear and stage.
    Warm water and low shear favour RI; very strong or weakening storms rarely RI.
    """
    score = (-1.5
             + 0.8 * (sst - 28.5)
             - 0.12 * (shear - 10.0)
             + _STAGE_RI_BONUS[stage]
             - 0.01 * max(0.0, wind - 150.0))
    return round(1.0 / (1.0 + math.exp(-score)), 3)


def _build_storms() -> dict[str, dict[str, Any]]:
    storms: dict[str, dict[str, Any]] = {}
    for name, d in STORM_DEFS.items():
        rng = random.Random(zlib.crc32(name.encode()))  # deterministic per storm
        pts = d["points"]
        first_fc = len(pts) - FORECAST_STEPS
        last_obs_hour = pts[first_fc - 1][0]
        trajectory = []
        for i, (hour, lat, lon, wind, pres, stage) in enumerate(pts):
            is_fc = i >= first_fc
            lead = hour - last_obs_hour if is_fc else 0
            sst, shear = _environment(rng, lat, stage)
            # Position uncertainty: ~15 km for analyses; cone grows with lead time.
            unc = 40_000 + 4_000 * lead if is_fc else 15_000 + rng.randint(-3000, 3000)
            trajectory.append({
                "step_index": i,
                "timestamp": (d["start_utc"] + timedelta(hours=hour)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "hours_since_start": hour,
                "is_forecast": is_fc,
                "lat": lat,
                "lon": lon,
                "wind_speed_kmh": wind,
                "pressure_hpa": pres,
                "stage": stage,
                "imd_category": imd_category(wind),
                "uncertainty_radius_m": int(unc),
                "ri_probability": _ri_probability(wind, sst, shear, stage),
                "sst_c": sst,
                "wind_shear_kt": shear,
            })
        storms[name] = {
            "name": name,
            "basin": d["basin"],
            "year": d["year"],
            "summary": d["summary"],
            "landfall": d["landfall"],
            "peak_wind_kmh": max(p["wind_speed_kmh"] for p in trajectory),
            "min_pressure_hpa": min(p["pressure_hpa"] for p in trajectory),
            "forecast_start_index": first_fc,
            "analysis_time_utc": trajectory[first_fc - 1]["timestamp"],
            "trajectory": trajectory,
        }
    return storms


STORMS = _build_storms()
_STORM_LOOKUP = {k.lower(): k for k in STORMS}


def get_storm(name: str) -> dict[str, Any]:
    key = _STORM_LOOKUP.get(name.strip().lower())
    if key is None:
        raise HTTPException(404, f"Unknown storm '{name}'. Available: {', '.join(STORMS)}")
    return STORMS[key]


def get_step(storm: dict[str, Any], step_index: int) -> dict[str, Any]:
    traj = storm["trajectory"]
    if not 0 <= step_index < len(traj):
        raise HTTPException(404, f"step_index must be between 0 and {len(traj) - 1} "
                                 f"for {storm['name']}.")
    return traj[step_index]


# =============================================================================
# 1) GET /storms
# =============================================================================

@app.get("/", tags=["meta"])
def root():
    return {"service": "CycloneAI", "docs": "/docs", "storms": list(STORMS)}


@app.get("/storms", tags=["storms"])
def list_storms():
    """Metadata + historical track + 24-hour forecast for all five cyclones."""
    return {"count": len(STORMS), "storms": list(STORMS.values())}


# =============================================================================
# 2) POST /predict-image  (placeholder model, easy to swap for PyTorch)
# =============================================================================
# ---- HOW TO PLUG IN YOUR OWN PyTorch MODEL LATER ----------------------------
# 1. pip install torch torchvision
# 2. Replace the body of `run_model_inference` with something like:
#
#       import torch, torchvision.transforms as T
#       _MODEL = torch.load("models/cyclone_resnet.pth", map_location="cpu")
#       _MODEL.eval()
#       _TF = T.Compose([T.Resize((224, 224)), T.ToTensor(),
#                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#
#       def run_model_inference(image_bytes):
#           img = Image.open(BytesIO(image_bytes)).convert("RGB")
#           x = _TF(img).unsqueeze(0)
#           with torch.no_grad():
#               out = _MODEL(x)                       # e.g. [detect_logit, 5 stage logits, wind]
#           ...map `out` to the dict returned below...
#
# The endpoint only relies on the returned dict keys, so nothing else changes.
# -----------------------------------------------------------------------------

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

_STAGE_BASE_WIND = {"Formation": 45, "Developing": 70, "Intensifying": 120,
                    "Mature": 190, "Weakening": 90}


def run_model_inference(image_bytes: bytes) -> dict[str, Any]:
    """
    PLACEHOLDER inference. Replace with a real model (see notes above).

    Contract (keep these keys when swapping the model):
        cyclone_detected : bool
        stage            : one of Formation/Developing/Intensifying/Mature/Weakening, or None
        confidence       : float in [0, 1]
        estimated_wind_speed_kmh : int
        model            : str (name/version of the model that produced the result)

    The heuristic below treats the picture like an infrared satellite image,
    where bright pixels = cold, high cloud tops. A larger cold-cloud fraction and
    more texture is read as a stronger, more organised storm. It is NOT science.
    """
    try:
        img = Image.open(BytesIO(image_bytes)).convert("L")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("File is not a readable image.") from exc

    img.thumbnail((256, 256))
    hist = img.histogram()
    total = max(sum(hist), 1)
    cold_ratio = sum(hist[200:]) / total          # fraction of very bright pixels
    std = ImageStat.Stat(img).stddev[0]            # contrast / texture

    detected = cold_ratio >= 0.05 and std >= 20

    if not detected:
        return {"cyclone_detected": False, "stage": None,
                "confidence": round(min(0.95, 0.6 + (0.05 - min(cold_ratio, 0.05)) * 6), 2),
                "estimated_wind_speed_kmh": 0, "model": "placeholder-heuristic-v0"}

    if cold_ratio >= 0.20 and std < 35:
        stage = "Weakening"        # large but diffuse cloud shield
    elif cold_ratio < 0.10:
        stage = "Formation"
    elif cold_ratio < 0.20:
        stage = "Developing"
    elif cold_ratio < 0.32:
        stage = "Intensifying"
    else:
        stage = "Mature"

    wind = _STAGE_BASE_WIND[stage] + int((cold_ratio * 100) % 10)  # small deterministic nudge
    confidence = round(min(0.97, 0.55 + min(std, 60) / 60 * 0.4), 2)
    return {"cyclone_detected": True, "stage": stage, "confidence": confidence,
            "estimated_wind_speed_kmh": wind, "model": "placeholder-heuristic-v0"}


@app.post("/predict-image", tags=["inference"])
async def predict_image(file: UploadFile = File(..., description="Satellite image (JPG/PNG)")):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(415, "Please upload an image file (image/jpeg, image/png, ...).")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image too large (max 10 MB).")
    try:
        result = run_model_inference(data)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    result["imd_category"] = imd_category(result["estimated_wind_speed_kmh"])
    result["filename"] = file.filename
    return result


# =============================================================================
# 3) GET /vulnerability/{storm_name}/{step_index}
# =============================================================================

DISTRICTS = [
    {"name": "Visakhapatnam", "state": "Andhra Pradesh", "lat": 17.6868, "lon": 83.2185},
    {"name": "Puri", "state": "Odisha", "lat": 19.8135, "lon": 85.8312},
    {"name": "Paradip", "state": "Odisha", "lat": 20.3166, "lon": 86.6085},
    {"name": "Balasore", "state": "Odisha", "lat": 21.4934, "lon": 86.9135},
    {"name": "Digha", "state": "West Bengal", "lat": 21.6275, "lon": 87.5093},
    {"name": "Mumbai", "state": "Maharashtra", "lat": 19.0760, "lon": 72.8777},
    {"name": "Diu", "state": "Daman & Diu", "lat": 20.7144, "lon": 70.9874},
    {"name": "Veraval", "state": "Gujarat", "lat": 20.9077, "lon": 70.3679},
    {"name": "Porbandar", "state": "Gujarat", "lat": 21.6417, "lon": 69.6293},
    {"name": "Dwarka", "state": "Gujarat", "lat": 22.2394, "lon": 68.9678},
    {"name": "Mandvi", "state": "Gujarat", "lat": 22.8333, "lon": 69.3500},
]

_RISK_RANK = {"Low Risk": 0, "Moderate Watch": 1, "High Risk": 2}
_RISK_ADVICE = {
    "High Risk": "Evacuate low-lying areas, move to cyclone shelters, suspend sea activity.",
    "Moderate Watch": "Prepare emergency kits, secure property, follow district alerts.",
    "Low Risk": "No immediate threat. Continue to monitor official updates.",
}


def _translational_speed_kmh(traj: list[dict[str, Any]], idx: int) -> float:
    """Storm forward speed estimated from the neighbouring track segment."""
    j = idx + 1 if idx < len(traj) - 1 else idx - 1
    a, b = traj[idx], traj[j]
    dist = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
    hours = abs(b["hours_since_start"] - a["hours_since_start"]) or 1
    return max(dist / hours, 8.0)  # floor avoids absurd ETAs for stalled storms


def assess_vulnerability(storm: dict[str, Any], step_index: int) -> dict[str, Any]:
    """
    Risk rules (simple and explainable - good for a viva):
      gale_radius_km = 60 + 1.3 * wind_kmh   (rough extent of damaging winds)
      High Risk       : district is inside the gale radius
      Moderate Watch  : storm approaching, within 2.5 x gale radius, ETA <= 48 h
      Low Risk        : otherwise
    """
    traj = storm["trajectory"]
    cur = get_step(storm, step_index)
    speed = _translational_speed_kmh(traj, step_index)
    gale_radius = 60 + 1.3 * cur["wind_speed_kmh"]
    ts = datetime.strptime(cur["timestamp"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

    results = []
    for d in DISTRICTS:
        dist = haversine_km(cur["lat"], cur["lon"], d["lat"], d["lon"])
        # Is the storm getting closer? Compare with the adjacent track point.
        if step_index < len(traj) - 1:
            nxt = traj[step_index + 1]
            approaching = haversine_km(nxt["lat"], nxt["lon"], d["lat"], d["lon"]) < dist
        else:
            prv = traj[step_index - 1]
            approaching = haversine_km(prv["lat"], prv["lon"], d["lat"], d["lon"]) > dist

        eta_h = dist / speed if approaching else None

        if dist <= gale_radius:
            risk = "High Risk"
        elif approaching and dist <= 2.5 * gale_radius and eta_h is not None and eta_h <= 48:
            risk = "Moderate Watch"
        else:
            risk = "Low Risk"

        brg = bearing_deg(d["lat"], d["lon"], cur["lat"], cur["lon"])  # district -> storm
        results.append({
            "district": d["name"],
            "state": d["state"],
            "distance_km": round(dist, 1),
            "storm_direction_from_district": compass(brg),
            "approaching": approaching,
            "eta_hours": round(eta_h, 1) if eta_h is not None else None,
            "eta_utc": (ts + timedelta(hours=eta_h)).strftime("%Y-%m-%dT%H:%MZ") if eta_h is not None else None,
            "gale_onset_hours": round(max(0.0, (dist - gale_radius) / speed), 1) if approaching else None,
            "risk": risk,
            "advice": _RISK_ADVICE[risk],
        })

    results.sort(key=lambda r: r["distance_km"])
    overall = max((r["risk"] for r in results), key=lambda r: _RISK_RANK[r])
    return {
        "storm": storm["name"],
        "step_index": step_index,
        "timestamp": cur["timestamp"],
        "is_forecast": cur["is_forecast"],
        "storm_position": {"lat": cur["lat"], "lon": cur["lon"]},
        "wind_speed_kmh": cur["wind_speed_kmh"],
        "stage": cur["stage"],
        "translational_speed_kmh": round(speed, 1),
        "gale_radius_km": round(gale_radius, 1),
        "overall_risk": overall,
        "districts": results,
    }


@app.get("/vulnerability/{storm_name}/{step_index}", tags=["risk"])
def vulnerability(storm_name: str, step_index: int):
    return assess_vulnerability(get_storm(storm_name), step_index)


# =============================================================================
# 4) GET /alert-translate
# =============================================================================

STAGE_LABELS = {
    "en": {"Formation": "Formation", "Developing": "Developing", "Intensifying": "Intensifying",
           "Mature": "Mature", "Weakening": "Weakening"},
    "hi": {"Formation": "निर्माण चरण", "Developing": "विकसित हो रहा", "Intensifying": "तेज़ी से मजबूत हो रहा",
           "Mature": "पूर्ण विकसित", "Weakening": "कमज़ोर पड़ रहा"},
    "or": {"Formation": "ଗଠନ ପର୍ଯ୍ୟାୟ", "Developing": "ବିକାଶ ହେଉଛି", "Intensifying": "ଶକ୍ତିଶାଳୀ ହେଉଛି",
           "Mature": "ପୂର୍ଣ୍ଣ ବିକଶିତ", "Weakening": "ଦୁର୍ବଳ ହେଉଛି"},
    "bn": {"Formation": "গঠনের পর্যায়", "Developing": "বিকশিত হচ্ছে", "Intensifying": "শক্তিশালী হচ্ছে",
           "Mature": "পূর্ণ বিকশিত", "Weakening": "দুর্বল হচ্ছে"},
}

# status line is the same for every severity tier; headline + action change.
ALERT_TEXT = {
    "en": {
        "status": "Cyclone status: {stage}. Wind speed: about {wind} km/h.",
        "advisory": ("CYCLONE ADVISORY:", "Stay tuned to IMD updates and keep emergency supplies ready."),
        "warning": ("CYCLONE WARNING:", "Fishermen must not venture into the sea. Stay away from weak "
                                        "structures and keep an emergency kit ready."),
        "danger": ("SEVERE CYCLONE DANGER:", "People in coastal areas must follow evacuation orders "
                                             "immediately and move to the nearest cyclone shelter."),
    },
    "hi": {
        "status": "चक्रवात की स्थिति: {stage}। हवा की गति: लगभग {wind} किमी/घंटा।",
        "advisory": ("चक्रवात सूचना:", "कृपया मौसम विभाग के ताज़ा अपडेट पर नज़र रखें और तैयार रहें।"),
        "warning": ("चक्रवात चेतावनी:", "मछुआरे समुद्र में न जाएँ, कमज़ोर घरों से दूर रहें और "
                                      "आपातकालीन सामान तैयार रखें।"),
        "danger": ("गंभीर चक्रवात खतरा:", "तटीय क्षेत्रों के लोग तुरंत प्रशासन के निकासी आदेश का पालन करें "
                                         "और निकटतम चक्रवात आश्रय में पहुँचें।"),
    },
    "or": {
        "status": "ବାତ୍ୟାର ସ୍ଥିତି: {stage}। ପବନର ବେଗ: ପ୍ରାୟ {wind} କିମି/ଘଣ୍ଟା।",
        "advisory": ("ବାତ୍ୟା ସୂଚନା:", "ଦୟାକରି ପାଣିପାଗ ବିଭାଗର ସର୍ବଶେଷ ସୂଚନା ଉପରେ ନଜର ରଖନ୍ତୁ ଏବଂ ପ୍ରସ୍ତୁତ ରୁହନ୍ତୁ।"),
        "warning": ("ବାତ୍ୟା ସତର୍କତା:", "ମତ୍ସ୍ୟଜୀବୀମାନେ ସମୁଦ୍ରକୁ ଯିବେ ନାହିଁ, ଦୁର୍ବଳ ଘରଠାରୁ ଦୂରରେ ରୁହନ୍ତୁ "
                                      "ଏବଂ ଜରୁରୀକାଳୀନ ସାମଗ୍ରୀ ପ୍ରସ୍ତୁତ ରଖନ୍ତୁ।"),
        "danger": ("ଗୁରୁତର ବାତ୍ୟା ବିପଦ:", "ଉପକୂଳବର୍ତ୍ତୀ ଅଞ୍ଚଳର ଲୋକମାନେ ତୁରନ୍ତ ପ୍ରଶାସନର ସ୍ଥାନାନ୍ତର ନିର୍ଦ୍ଦେଶ ପାଳନ "
                                     "କରନ୍ତୁ ଏବଂ ନିକଟତମ ବାତ୍ୟା ଆଶ୍ରୟସ୍ଥଳକୁ ଯାଆନ୍ତୁ।"),
    },
    "bn": {
        "status": "ঘূর্ণিঝড়ের অবস্থা: {stage}। বাতাসের গতি: প্রায় {wind} কিমি/ঘণ্টা।",
        "advisory": ("ঘূর্ণিঝড় বিজ্ঞপ্তি:", "অনুগ্রহ করে আবহাওয়া দপ্তরের সর্বশেষ আপডেটের দিকে নজর রাখুন "
                                            "এবং প্রস্তুত থাকুন।"),
        "warning": ("ঘূর্ণিঝড় সতর্কতা:", "মৎস্যজীবীরা সমুদ্রে যাবেন না, দুর্বল ঘর থেকে দূরে থাকুন "
                                        "এবং জরুরি সামগ্রী প্রস্তুত রাখুন।"),
        "danger": ("গুরুতর ঘূর্ণিঝড়ের বিপদ:", "উপকূলীয় এলাকার মানুষ অবিলম্বে প্রশাসনের সরিয়ে নেওয়ার "
                                              "নির্দেশ মেনে চলুন এবং নিকটতম ঘূর্ণিঝড় আশ্রয়কেন্দ্রে যান।"),
    },
}


def build_alert(stage: str, wind_kmh: float, lang: str) -> dict[str, str]:
    severity = "danger" if wind_kmh >= 118 else "warning" if wind_kmh >= 62 else "advisory"
    t = ALERT_TEXT[lang]
    headline, action = t[severity]
    stage_local = STAGE_LABELS[lang][stage]
    status = t["status"].format(stage=stage_local, wind=round(wind_kmh))
    return {"language": lang, "severity": severity, "stage_localized": stage_local,
            "message": f"{headline} {status} {action}"}


@app.get("/alert-translate", tags=["alerts"])
def alert_translate(
    stage: Stage = Query(..., description="Storm stage"),
    wind_speed_kmh: float = Query(..., ge=0, le=400, description="Sustained wind speed in km/h"),
    lang: Lang = Query(Lang.en, description="en | hi | or | bn"),
):
    return build_alert(stage.value, wind_speed_kmh, lang.value)


# =============================================================================
# 5) GET /generate-bulletin/{storm_name}/{step_index}
# =============================================================================

class IMDBulletinPDF(FPDF):
    """PDF with an IMD-style letterhead and page footer."""

    def header(self):
        self.set_font("Helvetica", "B", 13)
        self.cell(0, 6, "INDIA METEOROLOGICAL DEPARTMENT", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", "", 9)
        self.cell(0, 5, "Ministry of Earth Sciences, Government of India", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.cell(0, 5, "Cyclone Warning Division - Tropical Cyclone Bulletin", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(0, 51, 102)
        self.set_line_width(0.6)
        self.line(10, self.get_y() + 1, 200, self.get_y() + 1)
        self.ln(4)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(90, 90, 90)
        self.multi_cell(0, 3.5, "EDUCATIONAL SIMULATION generated by CycloneAI. Not an official IMD product. "
                                "Do not use for real-world decisions.   Page "
                                f"{self.page_no()}/{{nb}}", align="C")
        self.set_text_color(0, 0, 0)


def _section(pdf: FPDF, title: str):
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(230, 236, 245)
    pdf.cell(0, 6, _latin1(title), fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.ln(1)


def _para(pdf: FPDF, text: str):
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, _latin1(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _table(pdf: FPDF, headers: list[str], rows: list[list[Any]], widths: tuple[float, ...]):
    pdf.set_font("Helvetica", "", 8)
    with pdf.table(col_widths=widths, width=sum(widths), text_align="CENTER", line_height=5.5,
                   headings_style=FontFace(emphasis="BOLD", fill_color=(215, 222, 235))) as table:
        head = table.row()
        for h in headers:
            head.cell(_latin1(h))
        for r in rows:
            row = table.row()
            for c in r:
                row.cell(_latin1(str(c)))


def build_bulletin_pdf(storm: dict[str, Any], step_index: int) -> bytes:
    cur = get_step(storm, step_index)
    vuln = assess_vulnerability(storm, step_index)
    traj = storm["trajectory"]
    name = storm["name"].upper()
    category = imd_category(cur["wind_speed_kmh"])
    nearest = vuln["districts"][0]
    overall = vuln["overall_risk"]

    ts_utc = datetime.strptime(cur["timestamp"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    ts_ist = ts_utc.astimezone(IST)

    colour, rgb = {
        "High Risk": ("RED", (200, 30, 30)),
        "Moderate Watch": ("ORANGE", (230, 120, 0)),
    }.get(overall, ("YELLOW", (210, 170, 0)) if cur["wind_speed_kmh"] >= 62 else ("GREEN", (40, 140, 60)))

    pdf = IMDBulletinPDF(format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # --- Bulletin meta ---
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(95, 6, f"BULLETIN NO.: {storm['name'][:3].upper()}-{storm['year']}-{step_index + 1:02d}")
    pdf.cell(0, 6, f"Basin: {storm['basin']}", align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, f"Issued (IST): {ts_ist:%d %b %Y, %H%M} hrs    |    Valid (UTC): {ts_utc:%d %b %Y, %H%M}",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 5, f"Type: {'24-hour FORECAST position' if cur['is_forecast'] else 'Observed ANALYSIS position'}",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    # --- Colour-coded subject banner ---
    pdf.set_fill_color(*rgb)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 10)
    pdf.multi_cell(0, 7, _latin1(f" SUBJECT: {category} '{name}' over {storm['basin']} - {colour} MESSAGE"),
                   fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)

    # --- Synoptic situation ---
    _section(pdf, "1. CURRENT STATUS")
    gust = round(cur["wind_speed_kmh"] * 1.25)
    _para(pdf, (f"The {category} '{name}' ({cur['stage']} stage) lay centred at {ts_ist:%H%M} hrs IST of "
                f"{ts_ist:%d %b %Y} near Lat. {cur['lat']:.1f}N / Long. {cur['lon']:.1f}E, about "
                f"{nearest['distance_km']:.0f} km {nearest['storm_direction_from_district']} of {nearest['district']} "
                f"({nearest['state']}). Maximum sustained wind speed is {cur['wind_speed_kmh']} km/h, gusting to "
                f"about {gust} km/h. Estimated central pressure is {cur['pressure_hpa']} hPa. The system is moving at "
                f"about {vuln['translational_speed_kmh']:.0f} km/h. Position uncertainty radius: "
                f"{cur['uncertainty_radius_m'] / 1000:.0f} km."))

    # --- Environment ---
    _section(pdf, "2. ENVIRONMENTAL CONDITIONS & INTENSIFICATION OUTLOOK")
    _table(pdf, ["Sea Surface Temp (C)", "Wind Shear (kt)", "Rapid-Intensification Prob.", "Stage"],
           [[cur["sst_c"], cur["wind_shear_kt"], f"{cur['ri_probability'] * 100:.0f}%", cur["stage"]]],
           (50, 40, 60, 40))

    # --- Forecast table ---
    _section(pdf, "3. FORECAST TRACK AND INTENSITY")
    future = traj[step_index + 1: step_index + 5]
    if future:
        rows = []
        for p in future:
            t = datetime.strptime(p["timestamp"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            rows.append([f"{t:%d %b %H%M}", f"{p['lat']:.1f}N / {p['lon']:.1f}E",
                         p["wind_speed_kmh"], p["pressure_hpa"], imd_category(p["wind_speed_kmh"])])
        _table(pdf, ["Date/Time (UTC)", "Position", "Max Wind (km/h)", "Pressure (hPa)", "Category"],
               rows, (32, 38, 32, 30, 58))
    else:
        _para(pdf, "No further forecast positions available in this simulation (end of track).")

    # --- District risk ---
    _section(pdf, "4. COASTAL DISTRICT VULNERABILITY (nearest 6)")
    rows = [[d["district"], d["state"], f"{d['distance_km']:.0f}",
             f"{d['eta_hours']:.0f}" if d["eta_hours"] is not None else "-", d["risk"]]
            for d in vuln["districts"][:6]]
    _table(pdf, ["District", "State", "Distance (km)", "ETA (h)", "Risk Level"],
           rows, (36, 42, 34, 28, 50))

    # --- Warnings ---
    _section(pdf, "5. WARNINGS (illustrative)")
    w = cur["wind_speed_kmh"]
    if w >= 118:
        sea = "Phenomenal to very high seas; storm surge likely in low-lying coastal areas."
    elif w >= 62:
        sea = "Very rough to high seas near the system centre."
    else:
        sea = "Rough seas near the system centre."
    _para(pdf, f"Sea condition: {sea}")
    _para(pdf, "Fishermen: Total suspension of fishing operations. Boats at sea should return to safe harbour.")
    _para(pdf, f"Overall regional risk level: {overall.upper()}.")

    # --- Action ---
    _section(pdf, "6. ACTION SUGGESTED")
    _para(pdf, _RISK_ADVICE[overall])
    _para(pdf, "Disaster management authorities should keep NDRF/SDRF teams, relief material and "
               "communication systems on standby. Citizens should follow official instructions only.")

    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 5, "Next bulletin: next forecast step of the CycloneAI timeline.",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    return bytes(pdf.output())


@app.get("/generate-bulletin/{storm_name}/{step_index}", tags=["reports"])
def generate_bulletin(storm_name: str, step_index: int):
    storm = get_storm(storm_name)
    pdf_bytes = build_bulletin_pdf(storm, step_index)
    filename = f"IMD_Bulletin_{storm['name']}_step{step_index}.pdf"
    return Response(content=pdf_bytes, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


# =============================================================================
# 6) GET /generate-audio
# =============================================================================

def _broadcast_text(storm: dict[str, Any], step_index: int, lang: str) -> str:
    cur = get_step(storm, step_index)
    alert = build_alert(cur["stage"], cur["wind_speed_kmh"], lang)
    if lang != "en":
        return alert["message"]  # already fully localized
    vuln = assess_vulnerability(storm, step_index)
    n = vuln["districts"][0]
    risky = [d["district"] for d in vuln["districts"] if d["risk"] == "High Risk"]
    risk_line = (f"High risk districts are: {', '.join(risky)}. " if risky
                 else "No district is currently at high risk. ")
    return (f"This is a CycloneAI emergency broadcast. {imd_category(cur['wind_speed_kmh'])} {storm['name']} "
            f"is currently in the {cur['stage']} stage, with winds of {round(cur['wind_speed_kmh'])} kilometres per hour. "
            f"It is located {round(n['distance_km'])} kilometres {n['storm_direction_from_district']} of "
            f"{n['district']}. {risk_line}{alert['message']}")


@app.get("/generate-audio", tags=["reports"])
def generate_audio(
    storm_name: str = Query(..., description="Amphan, Biparjoy, Fani, Mocha or Tauktae"),
    step_index: int | None = Query(None, description="Track step; defaults to latest observed step"),
    lang: Lang = Query(Lang.en, description="en | hi | bn  (Odia falls back to English)"),
):
    if gTTS is None:
        raise HTTPException(503, "gTTS is not installed. Run: pip install gTTS")

    storm = get_storm(storm_name)
    idx = storm["forecast_start_index"] - 1 if step_index is None else step_index
    get_step(storm, idx)  # validates index

    tts_lang = lang.value
    fallback = False
    try:
        if tts_lang not in tts_langs():  # gTTS has no Odia voice
            tts_lang, fallback = "en", True
    except Exception:  # offline language list lookup failed; assume supported
        pass

    text = _broadcast_text(storm, idx, tts_lang)
    try:
        buf = BytesIO()
        gTTS(text=text, lang=tts_lang).write_to_fp(buf)
    except Exception as exc:  # gTTS raises generic errors on network problems
        raise HTTPException(503, f"Text-to-speech service unavailable (internet needed): {exc}") from exc

    headers = {"Content-Disposition": f'inline; filename="{storm["name"]}_alert_{tts_lang}.mp3"'}
    if fallback:
        headers["X-TTS-Language-Fallback"] = "en"
    return Response(content=buf.getvalue(), media_type="audio/mpeg", headers=headers)


# =============================================================================
# Local entry point:  python main.py
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)