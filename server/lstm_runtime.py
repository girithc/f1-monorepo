import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_CWD = Path(__file__).resolve().parent
_candidates = [_CWD, _CWD.parent, _CWD.parent.parent]

def _find_project_root():
    for base in _candidates:
        if (base / "data" / "constructors.csv").exists():
            return base
    for base in _candidates:
        d = base / "data"
        if d.exists() and (d / "results.csv").exists() and (d / "races.csv").exists():
            return base
    raise FileNotFoundError("Could not locate project root with ./data folder containing F1 CSVs")

PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

for fname in ["lap_times.csv", "pit_stops.csv"]:
    if not (DATA_DIR / fname).exists():
        raise FileNotFoundError(f"Missing {fname} in {DATA_DIR}")

LAP_TIMES = pd.read_csv(DATA_DIR / "lap_times.csv")
PIT_STOPS = pd.read_csv(DATA_DIR / "pit_stops.csv")

race_laps = LAP_TIMES.groupby("raceId", as_index=False)["lap"].max().rename(columns={"lap": "total_laps"})
LAP_TIMES = LAP_TIMES.merge(race_laps, on="raceId", how="left")

pit_flags = PIT_STOPS.groupby(["raceId", "driverId", "lap"], as_index=False)["stop"].count().rename(columns={"stop": "pit_flag"})
pit_flags["pit_flag"] = 1
LAP_TIMES = LAP_TIMES.merge(pit_flags, on=["raceId", "driverId", "lap"], how="left")
LAP_TIMES["pit_flag"] = LAP_TIMES["pit_flag"].fillna(0)

LAP_TIMES = LAP_TIMES.sort_values(["raceId", "driverId", "lap"])

class F1LstmRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h_last = h_n[-1]
        out = self.fc(h_last)
        return out.squeeze(-1)

def load_lstm_model(device=None):
    if device is None:
        device = torch.device("cpu")
    cfg_path = ARTIFACTS_DIR / "finish_regressor_lstm_config.json"
    weights_path = ARTIFACTS_DIR / "finish_regressor_lstm.pt"
    if not cfg_path.exists() or not weights_path.exists():
        raise FileNotFoundError("LSTM artifacts not found, run lstm_train.py first")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    model = F1LstmRegressor(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
    )
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device

def build_lap_sequence(race_id, driver_id, current_lap=None):
    df = LAP_TIMES[(LAP_TIMES["raceId"] == race_id) & (LAP_TIMES["driverId"] == driver_id)].copy()
    if df.empty:
        raise ValueError(f"No lap data for raceId={race_id}, driverId={driver_id}")
    df = df.sort_values("lap")
    if current_lap is not None:
        df = df[df["lap"] <= int(current_lap)]
        if df.empty:
            raise ValueError(f"No laps <= {current_lap} for raceId={race_id}, driverId={driver_id}")
    df["total_laps"] = df["total_laps"].clip(lower=1)
    df["lap_norm"] = df["lap"] / df["total_laps"]
    df["lap_ms"] = pd.to_numeric(df["milliseconds"], errors="coerce")
    if df["lap_ms"].isna().all():
        raise ValueError("lap_ms is all NaN for this sequence")
    
    df["lap_ms"] = (
        df["lap_ms"]
        .ffill()
        .bfill()
        .fillna(df["lap_ms"].median())
    )    
    df["gap_prev_lap_ms"] = df["lap_ms"].diff().fillna(0)
    if "position" not in df.columns:
        raise ValueError("lap_times.csv has no position column")
    
    df["position"] = (
        pd.to_numeric(df["position"], errors="coerce")
        .ffill()
        .bfill()
    )    
    if df["position"].isna().all():
        raise ValueError("position is all NaN for this sequence")
    pit_flag = df["pit_flag"].values
    stint = np.zeros_like(pit_flag, dtype=np.int64)
    current = 0
    for i in range(len(stint)):
        stint[i] = current
        if pit_flag[i] == 1:
            current += 1
    df["stint_idx"] = stint
    x = df[["lap_norm", "position", "pit_flag", "stint_idx", "lap_ms", "gap_prev_lap_ms"]].values.astype("float32")
    if x.shape[0] < 1:
        raise ValueError("Sequence has no laps")
    return x

def predict_finish_from_history(model, device, race_id, driver_id, current_lap=None):
    seq = build_lap_sequence(race_id, driver_id, current_lap=current_lap)
    length = seq.shape[0]
    x = torch.from_numpy(seq).unsqueeze(0).to(device)
    lengths = torch.tensor([length], dtype=torch.long).to(device)
    with torch.no_grad():
        y = model(x, lengths)
        y = torch.clamp(y, 1.0, 20.0)
    return float(y.item())
