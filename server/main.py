import os
import json
import math
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from joblib import load
import uvicorn

# =====================================================
# Config
# =====================================================
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "artifacts"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")

API_BASE_URL = os.getenv("API_BASE_URL", "http://0.0.0.0")

# Artifacts: ONLY the finish-position regressor (XGBoost pipeline)
REGRESSOR_PATH = MODEL_DIR / "finish_regressor_xgb.pkl"

# Helper metadata files
CIRCUIT_LAPS_PATH = HERE / "helper" / "circuit_laps.json"
OVERTAKE_INDEX_PATH = HERE / "helper" / "overtake_index.json"

# If helper's "overtakeIndex" is 'ease', flip to 'difficulty' expected by the model
OVERTAKE_HIGHER_IS_EASIER = True

# Debug + postprocessing knobs
DEBUG_PRED = os.getenv("DEBUG_PRED", "0") == "1"
SQUASH_SCALE = float(os.getenv("SQUASH_SCALE", "3.0"))  # higher = softer mapping
SQUASH_BIAS = float(os.getenv("SQUASH_BIAS", "0.0"))    # shifts the sigmoid left/right
INTERVAL_WIDTH = float(os.getenv("INTERVAL_WIDTH", "4.0"))
ROUND_DEFAULT = int(os.getenv("ROUND_DEFAULT", "1"))

# =====================================================
# Schemas (I/O)
# =====================================================
class PitStop(BaseModel):
    lap: int = Field(..., ge=1, description="Lap number of the stop")
    durationMs: int = Field(..., ge=1000, le=100000, description="Pit stop duration in milliseconds")

class PredictRequest(BaseModel):
    circuitId: Union[int, str]
    gridPosition: int = Field(..., ge=1, le=20)
    pitPlan: List[PitStop]
    driverId: Optional[str] = None

    @validator("pitPlan")
    def sort_pits(cls, v: List[PitStop]) -> List[PitStop]:
        return sorted(v, key=lambda p: p.lap)

class FeatureImpact(BaseModel):
    name: str
    impact: float
    direction: Optional[str] = None

# add this new model:
class Top3Out(BaseModel):
    probability: float
    source: Optional[str] = None  # e.g., "distribution"

# then change PredictResponse to use it:
class PredictResponse(BaseModel):
    prediction: Dict[str, float]
    top3: Top3Out
    positionProbs: Optional[Dict[str, float]] = None
    perPitEffects: Optional[List[Dict[str, float]]] = None
    explanation: Optional[Dict[str, List[FeatureImpact]]] = None
    modelVersion: str


class CompareScenario(BaseModel):
    id: str
    circuitId: Union[int, str]
    gridPosition: int = Field(..., ge=1, le=20)
    pitPlan: List[PitStop]

class CompareRequest(BaseModel):
    scenarios: List[CompareScenario]

class CompareResult(BaseModel):
    scenarioId: str
    finishP50: float
    intervalWidth: float
    top3Probability: float
    robustnessScore: float

class CompareResponse(BaseModel):
    results: List[CompareResult]
    recommendedScenarioId: str
    modelVersion: str

# =====================================================
# App init
# =====================================================
app = FastAPI(title="F1 Strategy Prediction API", version=MODEL_VERSION)

# --- CORS: allow all ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# Load artifacts + metadata at startup
# =====================================================
@dataclass
class Artifacts:
    regressor: Any = None
    circuit_laps: list = None
    overtake_difficulty: dict = None

ART = Artifacts()

def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)

def _require_int_circuit_id(raw) -> int:
    try:
        return int(raw)
    except Exception:
        s = str(raw).strip()
        if s.isdigit():
            return int(s)
        raise ValueError(f"Non-numeric circuitId: {raw}")

def _build_overtake_difficulty_map(overtake_json: Optional[list]) -> Dict[int, float]:
    if not overtake_json:
        return {}
    vals = [row.get("overtakeIndex") for row in overtake_json if "overtakeIndex" in row]
    vmin, vmax = (min(vals), max(vals)) if vals else (0.0, 1.0)
    rng = (vmax - vmin) or 1.0
    out: Dict[int, float] = {}
    for row in overtake_json:
        try:
            cid = _require_int_circuit_id(row.get("circuitId"))
        except ValueError:
            continue
        val = float(row.get("overtakeIndex", 0.5))
        norm = (val - vmin) / rng
        diff = 1.0 - norm if OVERTAKE_HIGHER_IS_EASIER else norm
        out[cid] = float(diff)
    return out

def _build_circuit_maps(circuit_laps_json: Optional[list]):
    circuit_meta: Dict[int, Dict[str, Any]] = {}
    name_to_id: Dict[str, int] = {}
    if not circuit_laps_json:
        return circuit_meta, name_to_id
    for row in circuit_laps_json:
        try:
            cid = _require_int_circuit_id(row.get("circuitId"))
        except ValueError:
            continue
        name = row.get("name_circuit") or row.get("name") or f"circuit_{cid}"
        country = row.get("country", "Unknown")
        avg_laps = row.get("avgLaps")
        if avg_laps is not None:
            try:
                avg_laps = float(avg_laps)
            except Exception:
                avg_laps = None
        circuit_meta[cid] = {"name": name, "country": country, "avgLaps": avg_laps}
        name_to_id[name.strip().lower()] = cid
    return circuit_meta, name_to_id

@app.on_event("startup")
def _startup():
    if not REGRESSOR_PATH.exists():
        raise RuntimeError(f"Regressor artifact not found: {REGRESSOR_PATH}")
    ART.regressor = load(REGRESSOR_PATH)

    circuit_laps_json = _load_json(CIRCUIT_LAPS_PATH) or []
    overtake_json = _load_json(OVERTAKE_INDEX_PATH) or []
    ART.circuit_laps = circuit_laps_json
    ART.overtake_difficulty = _build_overtake_difficulty_map(overtake_json)

    global CIRCUIT_META, NAME_TO_ID
    CIRCUIT_META, NAME_TO_ID = _build_circuit_maps(circuit_laps_json)

    # --- Schema self-check (fail fast if columns drift) ---
    try:
        dummy = pd.DataFrame([{
            "grid": 10,
            "pit_count": 0,
            "pit_total_duration": 0,
            "pit_avg_duration": 0,
            "first_pit_lap": 0,
            "last_pit_lap": 0,
            "circuit_overtake_difficulty": float(np.mean(list((ART.overtake_difficulty or {}).values()) or [0.5])),
            "round": ROUND_DEFAULT,
            "circuitId": 1,
            "country": "Unknown"
        }])
        expected = getattr(ART.regressor, "feature_names_in_", None)
        if expected is not None:
            missing = [c for c in expected if c not in dummy.columns]
            if missing:
                raise RuntimeError(f"Regressor schema mismatch. Missing columns at serve time: {missing}")
    except Exception as e:
        raise RuntimeError(f"Startup schema check failed: {e}")

NUMERIC_DEFAULTS = {
    "pit_count": 0,
    "pit_total_duration": 0,
    "pit_avg_duration": 0,
    "first_pit_lap": 0,
    "last_pit_lap": 0,
    "round": ROUND_DEFAULT,
}

def _resolve_circuit_id(circuit_id: Union[int, str]) -> int:
    try:
        return int(circuit_id)
    except Exception:
        key = str(circuit_id).strip().lower()
        if key in NAME_TO_ID:
            return NAME_TO_ID[key]
        if key.isdigit():
            return int(key)
        raise HTTPException(status_code=422, detail=f"Invalid circuitId: {circuit_id}. Use a known name or numeric id.")

def _scenario_to_features(circuit_id: Union[int, str], grid: int, pit_plan: List[PitStop]) -> pd.DataFrame:
    cid = _resolve_circuit_id(circuit_id)
    cmeta = CIRCUIT_META.get(cid, {"name": f"circuit_{cid}", "country": "Unknown", "avgLaps": None})
    country = cmeta["country"]

    pit_count = len(pit_plan)
    durations = [p.durationMs for p in pit_plan] if pit_plan else []
    laps = [p.lap for p in pit_plan] if pit_plan else []

    total_ms = int(np.sum(durations)) if durations else 0
    avg_ms = int(np.mean(durations)) if durations else 0
    first_lap = int(min(laps)) if laps else 0
    last_lap = int(max(laps)) if laps else 0

    od_map = ART.overtake_difficulty or {}
    od_values = list(od_map.values()) or [0.5]
    od_default = float(np.mean(od_values))
    circuit_overtake_difficulty = float(od_map.get(cid, od_default))

    row = pd.DataFrame([{
        "grid": grid,
        "pit_count": pit_count,
        "pit_total_duration": total_ms,
        "pit_avg_duration": avg_ms,
        "first_pit_lap": first_lap,
        "last_pit_lap": last_lap,
        "circuit_overtake_difficulty": circuit_overtake_difficulty,
        "round": NUMERIC_DEFAULTS["round"],
        "circuitId": cid,
        "country": country
    }])
    return row

def _squash_to_1_20(x: float, scale: float = SQUASH_SCALE, bias: float = SQUASH_BIAS) -> float:
    """Smoothly map any real-valued prediction to (1, 20) using a sigmoid."""
    z = (x + bias) / max(1e-6, scale)
    return 1.0 + 19.0 * (1.0 / (1.0 + math.exp(-z)))

# ---- Distribution over positions 1..20 ----
_Z_90 = 1.2815515655446004  # Phi^{-1}(0.90)

def _infer_sigma_from_interval(p10: float, p90: float) -> float:
    band = max(1e-6, p90 - p10)
    return max(1e-3, band / (2.0 * _Z_90))

def _phi_cdf(x: float, mu: float, sigma: float) -> float:
    z = (x - mu) / (max(1e-9, sigma) * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))

def _position_distribution(p50: float, p10: float, p90: float) -> Dict[str, float]:
    """
    Build a discrete distribution over {1..20} by integrating a Normal(mu=p50, sigma from p10/p90)
    into integer buckets [k-0.5, k+0.5], clamped at [-inf, 1.5] for k=1 and [19.5, +inf] for k=20.
    Returns dict {"P1": ..., ..., "P20": ...} summing to 1.0.
    """
    mu = float(p50)
    sigma = _infer_sigma_from_interval(float(p10), float(p90))
    probs = []
    for k in range(1, 21):
        lo = -float("inf") if k == 1 else (k - 0.5)
        hi =  float("inf") if k == 20 else (k + 0.5)
        p = _phi_cdf(hi, mu, sigma) - _phi_cdf(lo, mu, sigma)
        probs.append(max(0.0, p))
    s = sum(probs) or 1.0
    probs = [p / s for p in probs]
    return {f"P{k}": probs[k-1] for k in range(1, 21)}

def _predict_finish_and_distribution(circuit_id, grid, pit_plan: List[PitStop]):
    X_row = _scenario_to_features(circuit_id, grid, pit_plan)

    if DEBUG_PRED:
        print("\n--- DEBUG: Inference row ---")
        print(X_row.dtypes)
        print(X_row.to_dict(orient="records"))

    # Predict raw finish (regression), then squash to [1, 20]
    try:
        finish_pred_raw = float(ART.regressor.predict(X_row)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Regression prediction failed: {e}")

    p50 = _squash_to_1_20(finish_pred_raw)
    p10 = max(1.0, p50 - INTERVAL_WIDTH / 2)
    p90 = min(20.0, p50 + INTERVAL_WIDTH / 2)

    # Build discrete position distribution and derive top-3 from it
    pos_probs = _position_distribution(p50, p10, p90)
    top3_prob = float(sum(pos_probs[f"P{i}"] for i in (1, 2, 3)))

    return {
    "prediction": {
        "finishP50": float(p50),
        "finishP10": float(p10),
        "finishP90": float(p90)
    },
    "top3": {"probability": float(sum(pos_probs[f"P{i}"] for i in (1, 2, 3))),
             "source": "distribution"},
    "positionProbs": pos_probs
}


def _robustness_score(p50: float, interval_width: float, top3_prob: float) -> float:
    iw_score = max(0.0, 1.0 - (interval_width / 10.0))
    top3_score = float(top3_prob)
    rank_score = max(0.0, 1.0 - (p50 - 1.0) / 19.0)
    return float(0.4 * iw_score + 0.4 * top3_score + 0.2 * rank_score)

@app.get("/healthz")
def healthz():
    ok = ART.regressor is not None
    return {"status": "ok" if ok else "error", "modelVersion": MODEL_VERSION}

@app.get("/metadata")
def metadata():
    circuits = []
    od_map = ART.overtake_difficulty or {}
    for row in (ART.circuit_laps or []):
        try:
            cid = _require_int_circuit_id(row.get("circuitId"))
        except ValueError:
            continue
        circuits.append({
            "circuitId": cid,
            "name": row.get("name_circuit") or row.get("name") or f"circuit_{cid}",
            "country": row.get("country", "Unknown"),
            "avgLaps": row.get("avgLaps"),
            "overtakeDifficulty": float(od_map.get(cid, 0.5))
        })
    return {"circuits": circuits, "modelVersion": MODEL_VERSION}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    out = _predict_finish_and_distribution(req.circuitId, req.gridPosition, req.pitPlan)
    return {
        "prediction": out["prediction"],
        "top3": out["top3"],  # from distribution
        "positionProbs": out.get("positionProbs"),
        "perPitEffects": None,
        "explanation": None,
        "modelVersion": MODEL_VERSION
    }

@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    results: List[CompareResult] = []
    for sc in req.scenarios:
        out = _predict_finish_and_distribution(sc.circuitId, sc.gridPosition, sc.pitPlan)
        p50 = out["prediction"]["finishP50"]
        p10 = out["prediction"]["finishP10"]
        p90 = out["prediction"]["finishP90"]
        interval_width = float(p90 - p10)
        top3_prob = float(out["top3"]["probability"])
        robust = _robustness_score(p50, interval_width, top3_prob)
        results.append(CompareResult(
            scenarioId=sc.id,
            finishP50=float(p50),
            intervalWidth=float(interval_width),
            top3Probability=float(top3_prob),
            robustnessScore=float(robust),
        ))
    best = max(results, key=lambda r: r.robustnessScore) if results else None
    return CompareResponse(
        results=results,
        recommendedScenarioId=best.scenarioId if best else "",
        modelVersion=MODEL_VERSION
    )

# --- Uvicorn Runner ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting FastAPI server on {API_BASE_URL} (Port: {port})")
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
