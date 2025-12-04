import json
from pathlib import Path
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from torch import nn

# =====================================================
# 1. Setup & Config
# =====================================================
_CWD = Path.cwd().resolve()
_candidates = [_CWD, _CWD.parent, _CWD.parent.parent]

def _find_project_root():
    for base in _candidates:
        if (base / "data" / "constructors.csv").exists():
            return base
    for base in _candidates:
        d = base / "data"
        if d.exists() and (d / "results.csv").exists() and (d / "races.csv").exists():
            return base
    raise FileNotFoundError("Could not locate project root with a ./data folder containing the F1 CSVs.")

PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Load Data Once
races = pd.read_csv(DATA_DIR / "races.csv")
drivers = pd.read_csv(DATA_DIR / "drivers.csv")
results = pd.read_csv(DATA_DIR / "results.csv")
laps_raw = pd.read_csv(DATA_DIR / "lap_times.csv")
laps_raw = laps_raw.rename(columns={"milliseconds": "lap_ms"})
pits_raw = pd.read_csv(DATA_DIR / "pit_stops.csv")
qualifying = pd.read_csv(DATA_DIR / "qualifying.csv") # New for CPI

# =====================================================
# 2. Helpers (Must match Training Logic)
# =====================================================
def calculate_car_performance(results, races, qualifying):
    """
    Derives Car Performance Index (CPI). Re-calculated here for runtime lookup.
    """
    def _to_ms(x):
        if pd.isna(x): return np.nan
        s = str(x).strip()
        try:
            if ":" in s:
                m, rest = s.split(":")
                return (int(m) * 60.0 + float(rest)) * 1000.0
            return float(s) * 1000.0
        except:
            return np.nan

    q_df = qualifying.copy()
    for col in ["q1", "q2", "q3"]:
        q_df[col + "_ms"] = q_df[col].map(_to_ms)
    q_df["bestQ_ms"] = q_df[["q1_ms", "q2_ms", "q3_ms"]].min(axis=1)

    drv_cons = results[["raceId", "driverId", "constructorId"]].drop_duplicates()
    q = q_df.merge(drv_cons, on=["raceId", "driverId"], how="left")
    q = q.merge(races[["raceId", "year"]], on="raceId", how="left")
    
    if "constructorId_x" in q.columns:
        q["constructorId"] = q["constructorId_x"].fillna(q["constructorId_y"])

    team_best = (q.dropna(subset=["bestQ_ms"])
                   .groupby(["raceId", "year", "constructorId"], as_index=False)["bestQ_ms"]
                   .min())

    cons_season = (team_best
                   .groupby(["year", "constructorId"], as_index=False)["bestQ_ms"]
                   .median()
                   .rename(columns={"bestQ_ms": "med_bestQ_ms"}))

    season_minmax = cons_season.groupby("year")["med_bestQ_ms"].agg(["min", "max"]).reset_index()
    cons_season = cons_season.merge(season_minmax, on="year", how="left")
    rng = (cons_season["max"] - cons_season["min"]).replace(0, 1.0)
    
    cons_season["carPerformanceIndex"] = 1.0 - ((cons_season["med_bestQ_ms"] - cons_season["min"]) / rng)
    
    cpi_map = {}
    for _, row in cons_season.iterrows():
        cpi_map[(int(row["year"]), int(row["constructorId"]))] = float(row["carPerformanceIndex"])
    return cpi_map

def get_team_pit_standards(pits, races, results):
    """
    Calculates median pit duration per Team per Year for Sanitization.
    """
    df = pits.copy()
    df["milliseconds"] = pd.to_numeric(df["milliseconds"], errors="coerce")
    df = df.merge(races[["raceId", "year"]], on="raceId", how="left")
    driver_team_map = results[["raceId", "driverId", "constructorId"]].drop_duplicates()
    df = df.merge(driver_team_map, on=["raceId", "driverId"], how="left")
    df = df[df["milliseconds"] < 50000]
    
    standards = (df.groupby(["year", "constructorId"])["milliseconds"]
                   .median()
                   .to_dict())
    return standards

# Pre-calculate these on startup
CPI_MAP = calculate_car_performance(results, races, qualifying)
TEAM_PIT_STANDARDS = get_team_pit_standards(pits_raw, races, results)

def build_lap_pace_features(laps):
    grp = laps.groupby(["raceId", "lap"])["lap_ms"]
    med = grp.transform("median")
    std = grp.transform("std").replace(0, np.nan)
    laps = laps.copy()
    laps["pace_delta_ms"] = laps["lap_ms"] - med
    laps["pace_z"] = laps["pace_delta_ms"] / std
    laps["pace_z"] = laps["pace_z"].fillna(0.0)
    return laps

def build_pit_flags(laps, pits):
    pit_flags = pits[["raceId", "driverId", "lap", "milliseconds"]].copy()
    pit_flags.rename(columns={"milliseconds": "pit_ms"}, inplace=True)
    laps = laps.merge(pit_flags, on=["raceId", "driverId", "lap"], how="left")
    laps["on_pit_lap"] = laps["pit_ms"].notna().astype(np.float32)
    laps["pit_ms"] = laps["pit_ms"].fillna(0.0)
    laps = laps.sort_values(["raceId", "driverId", "lap"])

    def add_stint_features(df):
        stint_index = []
        laps_since = []
        current_stint = 0
        current_since = 0
        for pit_flag in df["on_pit_lap"].values:
            stint_index.append(current_stint)
            laps_since.append(current_since)
            current_since += 1
            if pit_flag > 0:
                current_stint += 1
                current_since = 0
        df["stint_index"] = np.asarray(stint_index, dtype=np.float32)
        df["laps_since_last_pit"] = np.asarray(laps_since, dtype=np.float32)
        return df

    laps = laps.groupby(["raceId", "driverId"], group_keys=False).apply(add_stint_features)
    return laps

# =====================================================
# 3. Request/Response Models
# =====================================================
class HistoryLap(BaseModel):
    lap: int
    lapTimeMs: int

class FuturePit(BaseModel):
    lap: int
    durationMs: int

class Scenario(BaseModel):
    name: str
    futurePits: List[FuturePit] = Field(default_factory=list)

class StrategyRequest(BaseModel):
    season: int | None = None
    grandPrix: str
    driverCode: str
    cutLap: int
    history: List[HistoryLap] = Field(default_factory=list)
    scenarios: List[Scenario]

class ScenarioResult(BaseModel):
    name: str
    predictedPosition: float # CHANGED: Now returns position (e.g. 3.4)
    futurePits: List[FuturePit]

class StrategyResponse(BaseModel):
    bestScenario: ScenarioResult
    allScenarios: List[ScenarioResult]

# =====================================================
# 4. Model Architecture (Must Match Training)
# =====================================================
class StrategyLSTM(nn.Module):
    def __init__(self, seq_input_dim, static_dim, plan_dim, hidden_dim=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=seq_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.static_proj = nn.Linear(static_dim, hidden_dim)
        self.plan_proj = nn.Linear(plan_dim, hidden_dim)
        
        # Regression Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 1),
        )

    def forward(self, seq, seq_lens, static, plan):
        packed = nn.utils.rnn.pack_padded_sequence(seq, seq_lens.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, (h_n, c_n) = self.lstm(packed)
        h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        static_h = torch.relu(self.static_proj(static))
        plan_h = torch.relu(self.plan_proj(plan))
        concat = torch.cat([h_last, static_h, plan_h], dim=-1)
        out = self.head(concat).squeeze(-1)
        return out

# Load Model
with open(ARTIFACTS_DIR / "strategy_lstm_config.json", "r") as f:
    cfg = json.load(f)

seq_input_dim = int(cfg["seq_input_dim"])
static_dim = int(cfg["static_dim"]) # Should be 3 now
plan_dim = int(cfg["plan_dim"])
hidden_dim = int(cfg.get("hidden_dim", 64))
num_layers = int(cfg.get("num_layers", 1))

model = StrategyLSTM(seq_input_dim, static_dim, plan_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=0.1)
state = torch.load(ARTIFACTS_DIR / "strategy_lstm.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()

device = torch.device("cpu")
model.to(device)

# =====================================================
# 5. FastAPI App
# =====================================================
app = FastAPI()

origins = [
    "http://localhost:9002",
    "http://127.0.0.1:9002",
    "https://captivating-emotion-production.up.railway.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,    
    allow_credentials=False,
    allow_methods=["*"],      
    allow_headers=["*"],
)

def _resolve_race_and_driver(req: StrategyRequest):
    # Find Driver
    drv_rows = drivers[drivers["code"] == req.driverCode]
    if drv_rows.empty:
        raise HTTPException(status_code=404, detail="Driver code not found")
    driver_id = int(drv_rows.iloc[0]["driverId"])

    # Find Race Candidates
    cand = races[races["name"] == req.grandPrix]
    if cand.empty:
        raise HTTPException(status_code=404, detail="Race not found for grandPrix")
    cand_ids = cand["raceId"].tolist()

    # Find Results
    res_mask = (results["driverId"] == driver_id) & (results["raceId"].isin(cand_ids))
    res_join = results[res_mask].merge(cand[["raceId", "year"]], on="raceId", how="left")
    res_join = res_join.merge(results[["raceId", "driverId", "constructorId"]], on=["raceId", "driverId"], how="left")

    if res_join.empty:
        raise HTTPException(status_code=404, detail="Driver did not race in matching grandPrix")

    # Pick Best Race (Most recent with data)
    res_join["laps"] = pd.to_numeric(res_join["laps"], errors='coerce').fillna(0)
    
    if req.season is not None:
        res_join = res_join[res_join["year"] == req.season]

    viable = res_join[res_join["laps"] > req.cutLap]
    if not viable.empty:
        chosen = viable.sort_values("year", ascending=False).iloc[0]
    else:
        chosen = res_join.sort_values("year", ascending=False).iloc[0]

    race_id = int(chosen["raceId"])
    year = int(chosen["year"])
    constructor_id = int(chosen["constructorId_x"]) if "constructorId_x" in chosen else int(chosen["constructorId"])
    
    # Get Grid & Total Laps
    grid = float(chosen["grid"])
    total_laps = int(chosen["laps"])
    
    return race_id, driver_id, year, constructor_id, grid, total_laps

def build_sequence(race_id, driver_id, cut_lap, history: List[HistoryLap]):
    race_laps = laps_raw[laps_raw["raceId"] == race_id].copy()
    if race_laps.empty:
        raise HTTPException(status_code=404, detail="No lap data for this race")

    # Apply Overrides (Real History)
    if history:
        overrides = {h.lap: float(h.lapTimeMs) for h in history}
        for lap, ms in overrides.items():
            mask = (race_laps["driverId"] == driver_id) & (race_laps["lap"] == lap)
            if mask.any():
                race_laps.loc[mask, "lap_ms"] = ms

    race_laps = build_lap_pace_features(race_laps)
    race_pits = pits_raw[pits_raw["raceId"] == race_id]
    race_laps = build_pit_flags(race_laps, race_pits)

    df = race_laps[(race_laps["driverId"] == driver_id) & (race_laps["lap"] <= cut_lap)]
    df = df.sort_values("lap")
    
    total_laps_f = float(df["lap"].max()) # Fallback
    
    seq_feats = np.stack([
        df["lap"].values / total_laps_f, # Approx normalization if total_laps not passed
        df["lap_ms"].values.astype(np.float32),
        df["pace_delta_ms"].values.astype(np.float32),
        df["pace_z"].values.astype(np.float32),
        df["position"].values.astype(np.float32),
        df["on_pit_lap"].values.astype(np.float32),
        df["pit_ms"].values.astype(np.float32),
        df["stint_index"].values.astype(np.float32),
        df["laps_since_last_pit"].values.astype(np.float32),
    ], axis=-1).astype(np.float32)

    return seq_feats, len(seq_feats)

def build_plan_features(cut_lap, total_laps_f, scenario: Scenario, year, constructor_id):
    # Retrieve Sanitized Standard for this Team
    std_pit_ms = TEAM_PIT_STANDARDS.get((year, constructor_id), 24000.0)
    
    remaining_pits = [p.lap for p in scenario.futurePits if p.lap > cut_lap]
    remaining_stops = len(remaining_pits)
    
    if remaining_stops > 0:
        next_pit = remaining_pits[0] / total_laps_f
        last_pit = remaining_pits[-1] / total_laps_f
        segments = [cut_lap] + remaining_pits + [int(total_laps_f)]
        gaps = [(segments[i+1] - segments[i]) / total_laps_f for i in range(len(segments)-1)]
        mean_stint = float(np.mean(gaps))
    else:
        next_pit = 1.0
        last_pit = cut_lap / total_laps_f
        mean_stint = (total_laps_f - cut_lap) / total_laps_f
        
    max_stops = 4
    pit_lap_vec = np.ones(max_stops, dtype=np.float32)
    pit_dur_vec = np.zeros(max_stops, dtype=np.float32)
    
    for i in range(max_stops):
        if i < remaining_stops:
            pit_lap_vec[i] = remaining_pits[i] / total_laps_f
            # SANITIZED DURATION (Team Standard)
            pit_dur_vec[i] = std_pit_ms / 100000.0
            
    plan_feat = np.concatenate([
        np.array([remaining_stops, next_pit, last_pit, mean_stint], dtype=np.float32),
        pit_lap_vec,
        pit_dur_vec,
    ], axis=0)
    
    if plan_feat.shape[0] != plan_dim:
        raise HTTPException(status_code=500, detail=f"Plan dim mismatch: Got {plan_feat.shape[0]}, expected {plan_dim}")
    return plan_feat

@app.post("/score", response_model=StrategyResponse)
def score_strategy(req: StrategyRequest):
    # 1. Resolve Context
    race_id, driver_id, year, constructor_id, grid, total_laps = _resolve_race_and_driver(req)
    total_laps_f = float(total_laps)

    # 2. Build Sequence (History - Dirty/Real)
    # Note: passing cut_lap to determine sequence length
    seq_feats, seq_len = build_sequence(race_id, driver_id, req.cutLap, req.history)
    
    # Fix Sequence Normalization (uses calculated total_laps_f from resolve, not just df max)
    seq_feats[:, 0] = seq_feats[:, 0] * (float(len(seq_feats)) / total_laps_f) # Re-scale approx
    
    seq_tensor = torch.from_numpy(seq_feats).float().unsqueeze(0).to(device)
    seq_lens_tensor = torch.tensor([seq_len], dtype=torch.long).to(device)

    # 3. Build Static (Grid, Laps, CPI)
    cpi = CPI_MAP.get((year, constructor_id), 0.5)
    static_feat = np.array([grid, total_laps_f, cpi], dtype=np.float32)
    static_tensor = torch.from_numpy(static_feat).float().unsqueeze(0).to(device)

    results_list = []
    
    # 4. Score Scenarios
    for scen in req.scenarios:
        # Build Plan (Future - Sanitized/Standard)
        plan_feat = build_plan_features(req.cutLap, total_laps_f, scen, year, constructor_id)
        plan_tensor = torch.from_numpy(plan_feat).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(seq_tensor, seq_lens_tensor, static_tensor, plan_tensor)
            # Regression Output: Position (1.0 to 20.0)
            pred_pos = float(torch.clamp(pred, 1.0, 20.0).item())

        results_list.append(
            ScenarioResult(
                name=scen.name,
                predictedPosition=pred_pos,
                futurePits=scen.futurePits,
            )
        )

    if not results_list:
        raise HTTPException(status_code=400, detail="No scenarios provided")

    # Lower position is better
    best = min(results_list, key=lambda r: r.predictedPosition)
    return StrategyResponse(bestScenario=best, allScenarios=results_list)