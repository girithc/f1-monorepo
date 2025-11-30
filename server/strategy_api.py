import json
from pathlib import Path
from typing import List
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from torch import nn

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

races = pd.read_csv(DATA_DIR / "races.csv")
drivers = pd.read_csv(DATA_DIR / "drivers.csv")
results = pd.read_csv(DATA_DIR / "results.csv")
laps_raw = pd.read_csv(DATA_DIR / "lap_times.csv")
laps_raw = laps_raw.rename(columns={"milliseconds": "lap_ms"})
pits_raw = pd.read_csv(DATA_DIR / "pit_stops.csv")

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

laps_feat = build_lap_pace_features(laps_raw)
laps_feat = build_pit_flags(laps_feat, pits_raw)

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
    successProb: float
    futurePits: List[FuturePit]
    relativeScore: float | None = None   # 0.0–1.0 within this request
    rating: int | None = None 

class StrategyResponse(BaseModel):
    bestScenario: ScenarioResult
    allScenarios: List[ScenarioResult]

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

with open(ARTIFACTS_DIR / "finish_regressor_lstm_config.json", "r") as f:
    cfg = json.load(f)

seq_input_dim = int(cfg["seq_input_dim"])
static_dim = int(cfg["static_dim"])
plan_dim = int(cfg["plan_dim"])
hidden_dim = int(cfg.get("hidden_dim", 64))
num_layers = int(cfg.get("num_layers", 1))

model = StrategyLSTM(seq_input_dim, static_dim, plan_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=0.1)
state = torch.load(ARTIFACTS_DIR / "finish_regressor_lstm.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()

device = torch.device("cpu")
model.to(device)

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


# ---------- NEW: history-based boost helper ----------

def _compute_history_boost(req: StrategyRequest, race_id: int, driver_id: int) -> float:
    """
    Compare user's history lapTimeMs to this driver's typical pace at this GP,
    and return a logit boost:

      - Positive -> user laps are faster than typical
      - Negative -> user laps are slower than typical
    """
    if not req.history:
        return 0.0

    ref_df = laps_raw[
        (laps_raw["raceId"] == race_id)
        & (laps_raw["driverId"] == driver_id)
        & (laps_raw["lap"] <= req.cutLap)
    ]
    if ref_df.empty:
        return 0.0

    # Reference "normal" pace: median lap time in this race up to cutLap
    ref_med = float(ref_df["lap_ms"].median())

    hist_ms = np.array([h.lapTimeMs for h in req.history], dtype=np.float32)
    hist_avg = float(hist_ms.mean())

    # Lower lap time = faster = better
    rel_diff = (ref_med - hist_avg) / max(ref_med, 1.0)
    # Clamp to ±20% so outliers don’t explode
    rel_diff = max(-0.2, min(0.2, rel_diff))

    # Map ±20% difference → about ±1.5 logit shift
    max_boost_logit = 1.5
    boost = (rel_diff / 0.2) * max_boost_logit  # -0.2 -> -1.5, +0.2 -> +1.5

    return float(boost)

# -----------------------------------------------------

def _resolve_race_and_driver(req: StrategyRequest):
    """
    Returns (race_id, driver_id, year, driver_laps).

    Logic:
      - find driverId from driverCode
      - find all races with this grandPrix name
      - keep only races where this driver actually has a result
      - compute how many laps the driver has data for (results.laps, fall back to lap_times)
      - if possible, choose the most recent race where driver_laps > cutLap
        otherwise, choose the most recent race overall and let caller validate cutLap
    """
    # 1) Driver
    drv_rows = drivers[drivers["code"] == req.driverCode]
    if drv_rows.empty:
        raise HTTPException(status_code=404, detail="Driver code not found")
    driver_id = int(drv_rows.iloc[0]["driverId"])

    # 2) Candidate races by GP name
    cand = races[races["name"] == req.grandPrix]
    if cand.empty:
        raise HTTPException(status_code=404, detail="Race not found for grandPrix")
    cand_ids = cand["raceId"].tolist()

    # 3) Results for this driver in those races
    res_mask = (results["driverId"] == driver_id) & (results["raceId"].isin(cand_ids))
    res_join = results[res_mask].merge(cand[["raceId", "year"]], on="raceId", how="left")

    if res_join.empty:
        raise HTTPException(
            status_code=404,
            detail="Driver did not race in any matching grandPrix"
        )

    # 4) Compute driver_laps for each race (use results.laps, fallback to lap_times if needed)
    def _compute_driver_laps(row):
        laps_val = row.get("laps", None)
        if laps_val is None or not np.isfinite(laps_val) or laps_val <= 0:
            mask = (laps_raw["raceId"] == row["raceId"]) & (laps_raw["driverId"] == driver_id)
            max_lap = laps_raw.loc[mask, "lap"].max()
            return float(max_lap) if np.isfinite(max_lap) else 0.0
        return float(laps_val)

    res_join["driver_laps"] = res_join.apply(_compute_driver_laps, axis=1)

    # 5) If season is provided, prefer that year, but still respect laps
    if req.season is not None:
        res_season = res_join[res_join["year"] == req.season]
        if not res_season.empty:
            res_join = res_season

    # 6) Prefer races where driver_laps > cutLap
    viable = res_join[res_join["driver_laps"] > req.cutLap]
    if not viable.empty:
        # most recent among viable
        chosen = viable.sort_values("year", ascending=False).iloc[0]
    else:
        # fallback: most recent overall
        chosen = res_join.sort_values("year", ascending=False).iloc[0]

    race_id = int(chosen["raceId"])
    year = int(chosen["year"])
    driver_laps = int(chosen["driver_laps"])
    return race_id, driver_id, year, driver_laps

def build_sequence(race_id, driver_id, cut_lap, history: List[HistoryLap]):
    # 1) Start from *raw* laps for this race (all drivers)
    race_laps = laps_raw[laps_raw["raceId"] == race_id].copy()
    if race_laps.empty:
        raise HTTPException(status_code=404, detail="No lap data for this race")

    # 2) Apply user-provided history overrides for THIS driver
    if history:
        overrides = {h.lap: float(h.lapTimeMs) for h in history}
        for lap, ms in overrides.items():
            mask = (race_laps["driverId"] == driver_id) & (race_laps["lap"] == lap)
            if mask.any():
                race_laps.loc[mask, "lap_ms"] = ms

    # 3) Rebuild pace & pit features for this race only
    race_laps = build_lap_pace_features(race_laps)
    race_pits = pits_raw[pits_raw["raceId"] == race_id]
    race_laps = build_pit_flags(race_laps, race_pits)

    # 4) Now filter down to this driver and laps <= cut_lap
    df = race_laps[(race_laps["driverId"] == driver_id) & (race_laps["lap"] <= cut_lap)]
    df = df.sort_values("lap")
    if df.empty:
        raise HTTPException(status_code=400, detail="cutLap before first recorded lap")

    # 5) Work out total laps for this race/driver
    total_laps = df["lap"].max()
    res_row = results[(results["raceId"] == race_id) & (results["driverId"] == driver_id)]
    if not res_row.empty and np.isfinite(res_row["laps"].iloc[0]) and res_row["laps"].iloc[0] > 0:
        total_laps = int(res_row["laps"].iloc[0])
    total_laps_f = float(total_laps)

    # 6) Build the per-lap feature sequence
    seq_feats = np.stack(
        [
            df["lap"].values.astype(np.float32) / total_laps_f,
            df["lap_ms"].values.astype(np.float32),
            df["pace_delta_ms"].values.astype(np.float32),
            df["pace_z"].values.astype(np.float32),
            df["position"].values.astype(np.float32),
            df["on_pit_lap"].values.astype(np.float32),
            df["pit_ms"].values.astype(np.float32),
            df["stint_index"].values.astype(np.float32),
            df["laps_since_last_pit"].values.astype(np.float32),
        ],
        axis=-1,
    ).astype(np.float32)

    seq_len = seq_feats.shape[0]
    if seq_input_dim != seq_feats.shape[1]:
        raise HTTPException(status_code=500, detail="seq_input_dim mismatch")

    # 7) Static features (grid, total laps)
    grid = 0.0
    if not res_row.empty:
        grid = float(res_row["grid"].iloc[0])
    static_feat = np.array([grid, total_laps_f], dtype=np.float32)

    return seq_feats, seq_len, static_feat, total_laps_f

def build_plan_features(cut_lap, total_laps_f, scenario: Scenario):
    remaining_pits = [p.lap for p in scenario.futurePits if p.lap > cut_lap]
    remaining_durs = [float(p.durationMs) for p in scenario.futurePits if p.lap > cut_lap]
    remaining_stops = len(remaining_pits)
    if remaining_stops > 0:
        next_pit = remaining_pits[0] / total_laps_f
        last_pit = remaining_pits[-1] / total_laps_f
        segments = [cut_lap] + remaining_pits + [total_laps_f]
        gaps = []
        for i in range(len(segments) - 1):
            gaps.append((segments[i + 1] - segments[i]) / total_laps_f)
        mean_stint = float(np.mean(gaps)) if gaps else 0.0
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
            pit_dur_vec[i] = remaining_durs[i] / 100000.0
    plan_feat = np.concatenate(
        [
            np.array([remaining_stops, next_pit, last_pit, mean_stint], dtype=np.float32),
            pit_lap_vec,
            pit_dur_vec,
        ],
        axis=0,
    )
    if plan_feat.shape[0] != plan_dim:
        raise HTTPException(status_code=500, detail="plan_dim mismatch")
    return plan_feat

@app.post("/score", response_model=StrategyResponse)
def score_strategy(req: StrategyRequest):
    # Resolve which historical race to use for laps, based on GP + driver (+ optional season + cutLap)
    race_id, driver_id, year, driver_laps = _resolve_race_and_driver(req)

    if driver_laps <= 0:
        raise HTTPException(
            status_code=404,
            detail="No lap data available for this race/driver"
        )

    total_laps_val = driver_laps

    if req.cutLap < 1 or req.cutLap >= total_laps_val:
        raise HTTPException(
            status_code=400,
            detail=f"cutLap must be between 1 and {total_laps_val - 1} for this race/driver (max laps = {total_laps_val})"
        )

    # NEW: compute history-based logit shift once per request
    history_boost = _compute_history_boost(req, race_id, driver_id)

    seq_feats, seq_len, static_feat, total_laps_f = build_sequence(
        race_id,
        driver_id,
        req.cutLap,
        req.history,
    )
    seq_tensor = torch.from_numpy(seq_feats).float().unsqueeze(0).to(device)
    seq_lens_tensor = torch.tensor([seq_len], dtype=torch.long).to(device)
    static_tensor = torch.from_numpy(static_feat).float().unsqueeze(0).to(device)

    results_list = []
    for scen in req.scenarios:
        plan_feat = build_plan_features(req.cutLap, total_laps_f, scen)
        plan_tensor = torch.from_numpy(plan_feat).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(seq_tensor, seq_lens_tensor, static_tensor, plan_tensor)

            # apply history boost on top of model logits
            if history_boost != 0.0:
                logits = logits + history_boost

            prob = torch.sigmoid(logits).cpu().numpy()[0].item()
        results_list.append(
            ScenarioResult(
                name=scen.name,
                successProb=float(prob),
                futurePits=scen.futurePits,
            )
        )

    if not results_list:
        raise HTTPException(status_code=400, detail="No scenarios provided")

    # ---- normalize probabilities to a 0–1 range per request ----
    probs = [r.successProb for r in results_list]
    p_min = min(probs)
    p_max = max(probs)

    if p_max > p_min:
        # ---- map absolute probability → rating 0–100 ----
        for r in results_list:
            # clamp just in case
            p = max(0.0, min(1.0, r.successProb))
            r.relativeScore = float(p)              # 0.0 – 1.0, same as prob
            r.rating = int(round(p * 100.0))        # 0 – 100

    else:
        # All scenarios basically identical → mark as 50/100
        for r in results_list:
            r.relativeScore = 0.5
            r.rating = 50

    best = max(results_list, key=lambda r: r.successProb)
    return StrategyResponse(bestScenario=best, allScenarios=results_list)
