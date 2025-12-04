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

RESULTS = pd.read_csv(DATA_DIR / "results.csv")
LAP_TIMES = pd.read_csv(DATA_DIR / "lap_times.csv")
PIT_STOPS = pd.read_csv(DATA_DIR / "pit_stops.csv")

LAP_TIMES = LAP_TIMES.rename(columns={"milliseconds": "lap_ms"})

race_laps = LAP_TIMES.groupby("raceId", as_index=False)["lap"].max().rename(columns={"lap": "total_laps"})
LAP_TIMES = LAP_TIMES.merge(race_laps, on="raceId", how="left")

pit_flags = PIT_STOPS.groupby(["raceId", "driverId", "lap"], as_index=False)["stop"].count().rename(columns={"stop": "pit_flag"})
pit_flags["pit_flag"] = 1
LAP_TIMES = LAP_TIMES.merge(pit_flags, on=["raceId", "driverId", "lap"], how="left")
LAP_TIMES["pit_flag"] = LAP_TIMES["pit_flag"].fillna(0.0)

LAP_TIMES = LAP_TIMES.sort_values(["raceId", "driverId", "lap"])

class StrategyLSTM(nn.Module):
    def __init__(self, seq_input_dim, static_dim, plan_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=seq_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
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
        packed = nn.utils.rnn.pack_padded_sequence(seq, seq_lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        static_h = torch.relu(self.static_proj(static))
        plan_h = torch.relu(self.plan_proj(plan))
        concat = torch.cat([h_last, static_h, plan_h], dim=-1)
        out = self.head(concat).squeeze(-1)
        return out

def load_strategy_lstm_model(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_path = ARTIFACTS_DIR / "strategy_lstm_config.json"
    weights_path = ARTIFACTS_DIR / "strategy_lstm.pt"
    if not cfg_path.exists() or not weights_path.exists():
        raise FileNotFoundError("strategy_lstm artifacts not found, run strategy_lstm.py first")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    model = StrategyLSTM(
        seq_input_dim=cfg["seq_input_dim"],
        static_dim=cfg["static_dim"],
        plan_dim=cfg["plan_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
    )
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device, cfg

def _build_sequence_and_static(race_id, driver_id, current_lap):
    df = LAP_TIMES[(LAP_TIMES["raceId"] == race_id) & (LAP_TIMES["driverId"] == driver_id)].copy()
    if df.empty:
        raise ValueError(f"No lap data for raceId={race_id}, driverId={driver_id}")
    df = df.sort_values("lap")
    if current_lap is not None:
        df = df[df["lap"] <= int(current_lap)]
        if df.empty:
            raise ValueError(f"No laps <= {current_lap} for raceId={race_id}, driverId={driver_id}")

    res_row = RESULTS[(RESULTS["raceId"] == race_id) & (RESULTS["driverId"] == driver_id)]
    if res_row.empty:
        raise ValueError(f"No result row for raceId={race_id}, driverId={driver_id}")
    res_row = res_row.iloc[0]
    grid = float(res_row["grid"])
    total_laps = float(res_row["laps"])
    if not np.isfinite(total_laps) or total_laps <= 0:
        total_laps = float(df["total_laps"].max())
    if total_laps <= 0:
        total_laps = float(df["lap"].max())

    df["total_laps"] = df["total_laps"].clip(lower=1)
    df["lap_norm"] = df["lap"] / df["total_laps"]
    df["lap_ms"] = pd.to_numeric(df["lap_ms"], errors="coerce")
    if df["lap_ms"].isna().all():
        raise ValueError("lap_ms is all NaN for this sequence")
    df["lap_ms"] = df["lap_ms"].ffill().bfill().fillna(df["lap_ms"].median())

    grp = df.groupby(["raceId", "lap"])["lap_ms"]
    med = grp.transform("median")
    std = grp.transform("std").replace(0, np.nan)
    df["pace_delta_ms"] = df["lap_ms"] - med
    df["pace_z"] = df["pace_delta_ms"] / std
    df["pace_z"] = df["pace_z"].fillna(0.0)

    df["gap_prev_lap_ms"] = df["lap_ms"].diff().fillna(0.0)

    df["position"] = pd.to_numeric(df["position"], errors="coerce").ffill().bfill()
    if df["position"].isna().all():
        raise ValueError("position is all NaN for this sequence")

    pit_flag = df["pit_flag"].values
    stint = np.zeros_like(pit_flag, dtype=np.float32)
    laps_since = np.zeros_like(pit_flag, dtype=np.float32)
    current_stint = 0.0
    since = 0.0
    for i, flag in enumerate(pit_flag):
        stint[i] = current_stint
        laps_since[i] = since
        since += 1.0
        if flag == 1.0:
            current_stint += 1.0
            since = 0.0
    df["stint_index"] = stint
    df["laps_since_last_pit"] = laps_since

    x = df[[
        "lap_norm",
        "lap_ms",
        "pace_delta_ms",
        "pace_z",
        "position",
        "pit_flag",
        "lap_ms",
        "stint_index",
        "laps_since_last_pit",
    ]].copy()

    x_vals = np.stack([
        x["lap_norm"].values.astype(np.float32),
        x["lap_ms"].values.astype(np.float32),
        x["pace_delta_ms"].values.astype(np.float32),
        x["pace_z"].values.astype(np.float32),
        x["position"].values.astype(np.float32),
        x["pit_flag"].values.astype(np.float32),
        x["lap_ms"].values.astype(np.float32),
        x["stint_index"].values.astype(np.float32),
        x["laps_since_last_pit"].values.astype(np.float32),
    ], axis=-1)

    static_feat = np.array([grid, float(total_laps)], dtype=np.float32)
    return x_vals, static_feat, int(total_laps)

def _build_plan_features(total_laps, current_lap, planned_pits, max_stops=4):
    total_laps_f = float(total_laps)
    cp = int(current_lap)
    pits_after = sorted(int(p) for p in planned_pits if int(p) > cp and int(p) <= total_laps)
    remaining_stops = len(pits_after)

    if remaining_stops > 0:
        next_pit = pits_after[0] / total_laps_f
        last_pit = pits_after[-1] / total_laps_f
        segments = [cp] + pits_after + [total_laps]
        gaps = []
        for i in range(len(segments) - 1):
            gaps.append((segments[i + 1] - segments[i]) / total_laps_f)
        mean_stint = float(np.mean(gaps)) if gaps else 0.0
    else:
        next_pit = 1.0
        last_pit = cp / total_laps_f
        mean_stint = (total_laps - cp) / total_laps_f

    pit_vec = np.ones(max_stops, dtype=np.float32)
    for i in range(max_stops):
        if i < remaining_stops:
            pit_vec[i] = pits_after[i] / total_laps_f

    plan_feat = np.concatenate([
        np.array([remaining_stops, next_pit, last_pit, mean_stint], dtype=np.float32),
        pit_vec,
    ], axis=0)
    return plan_feat

def predict_success_for_strategy(model, device, race_id, driver_id, current_lap, planned_pits):
    seq_np, static_np, total_laps = _build_sequence_and_static(race_id, driver_id, current_lap)
    plan_np = _build_plan_features(total_laps, current_lap, planned_pits)

    length = seq_np.shape[0]
    seq = torch.from_numpy(seq_np).unsqueeze(0).to(device)
    seq_lens = torch.tensor([length], dtype=torch.long).to(device)
    static = torch.from_numpy(static_np).unsqueeze(0).to(device)
    plan = torch.from_numpy(plan_np).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(seq, seq_lens, static, plan)
        prob = torch.sigmoid(logits)
    return float(prob.item())
