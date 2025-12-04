import os
from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

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
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# 2. Data Loading & Preprocessing Helpers
# =====================================================
def load_core():
    results = pd.read_csv(DATA_DIR / "results.csv")
    laps = pd.read_csv(DATA_DIR / "lap_times.csv")
    laps = laps.rename(columns={"milliseconds": "lap_ms"})
    pits = pd.read_csv(DATA_DIR / "pit_stops.csv")
    races = pd.read_csv(DATA_DIR / "races.csv")
    qualifying = pd.read_csv(DATA_DIR / "qualifying.csv") # Needed for Car Perf
    return results, laps, pits, races, qualifying

def build_base_results(results, races):
    # Merge race info
    res = results.merge(races[["raceId", "year", "round"]], on="raceId", how="left")
    
    # Filter valid finishes
    res = res[(res["positionOrder"].notna()) & (res["positionOrder"] > 0)]
    res = res[res["grid"] > 0]
    
    # Filter chaotic/small races
    finishers = (res.assign(classified=(res["positionOrder"] > 0).astype(int))
                    .groupby("raceId", as_index=False)["classified"].sum()
                    .rename(columns={"classified": "n_finishers"}))
    res = res.merge(finishers, on="raceId", how="left")
    res = res[res["n_finishers"] >= 16]
    res = res[res["year"] >= 2014] # Hybrid era only
    return res

def calculate_car_performance(results, races, qualifying):
    """
    Derives Car Performance Index (CPI) from Qualifying Pace.
    1.0 = Fastest car of the year, 0.0 = Slowest.
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

    qualifying = qualifying.copy()
    for col in ["q1", "q2", "q3"]:
        qualifying[col + "_ms"] = qualifying[col].map(_to_ms)
    qualifying["bestQ_ms"] = qualifying[["q1_ms", "q2_ms", "q3_ms"]].min(axis=1)

    # Link Constructor
    drv_cons = results[["raceId", "driverId", "constructorId"]].drop_duplicates()
    q = qualifying.merge(drv_cons, on=["raceId", "driverId"], how="left")
    q = q.merge(races[["raceId", "year"]], on="raceId", how="left")
    
    # Fill missing constructor if possible
    if "constructorId_x" in q.columns:
        q["constructorId"] = q["constructorId_x"].fillna(q["constructorId_y"])

    # Best qualifying time per Team per Race
    team_best = (q.dropna(subset=["bestQ_ms"])
                   .groupby(["raceId", "year", "constructorId"], as_index=False)["bestQ_ms"]
                   .min())

    # Median gap per Year
    cons_season = (team_best
                   .groupby(["year", "constructorId"], as_index=False)["bestQ_ms"]
                   .median()
                   .rename(columns={"bestQ_ms": "med_bestQ_ms"}))

    # Normalize 0..1 per season
    season_minmax = cons_season.groupby("year")["med_bestQ_ms"].agg(["min", "max"]).reset_index()
    cons_season = cons_season.merge(season_minmax, on="year", how="left")
    rng = (cons_season["max"] - cons_season["min"]).replace(0, 1.0)
    
    # 1.0 = Fastest (Lowest Time)
    cons_season["carPerformanceIndex"] = 1.0 - ((cons_season["med_bestQ_ms"] - cons_season["min"]) / rng)
    
    # Return lookup dict: {(year, constructorId): cpi}
    cpi_map = {}
    for _, row in cons_season.iterrows():
        cpi_map[(int(row["year"]), int(row["constructorId"]))] = float(row["carPerformanceIndex"])
    return cpi_map

def get_team_pit_standards(pits, races, results):
    """
    Calculates median pit duration per Team per Year.
    Used to SANITIZE future pit stops in the plan.
    """
    df = pits.copy()
    df["milliseconds"] = pd.to_numeric(df["milliseconds"], errors="coerce")
    df = df.merge(races[["raceId", "year"]], on="raceId", how="left")
    
    driver_team_map = results[["raceId", "driverId", "constructorId"]].drop_duplicates()
    df = df.merge(driver_team_map, on=["raceId", "driverId"], how="left")
    
    # Filter out repairs (>50s)
    df = df[df["milliseconds"] < 50000]
    
    standards = (df.groupby(["year", "constructorId"])["milliseconds"]
                   .median()
                   .to_dict())
    return standards

def build_lap_pace_features(laps):
    # Calculate Pace Z-Score per Race
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
    
    # Calculate Stint Index and Laps Since Pit
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
# 3. Example Generation (The Core Logic)
# =====================================================
def make_examples(res, laps, pits, cpi_map, team_pit_standards, min_cut_lap=5):
    laps = build_lap_pace_features(laps)
    laps = build_pit_flags(laps, pits)
    
    # Merge targets and static info
    # IMPORTANT: Include 'constructorId' and 'year' for lookups
    laps = laps.merge(res[["raceId", "driverId", "grid", "positionOrder", "laps", "constructorId", "year"]], 
                      on=["raceId", "driverId"], how="inner")
    
    laps = laps.sort_values(["raceId", "driverId", "lap"])
    
    pits = pits.copy()
    pits["milliseconds"] = pd.to_numeric(pits["milliseconds"], errors="coerce").fillna(0.0)
    
    groups = laps.groupby(["raceId", "driverId"])
    examples = []
    race_ids = []
    
    for (race_id, driver_id), df in groups:
        df = df.reset_index(drop=True)
        total_laps = df["laps"].iloc[0]
        if not np.isfinite(total_laps) or total_laps <= 0:
            total_laps = df["lap"].max()
        if total_laps <= 0: continue
            
        constructor_id = int(df["constructorId"].iloc[0])
        year = int(df["year"].iloc[0])
        
        # 1. Static Features: [Grid, TotalLaps, CarPerfIndex]
        grid = float(df["grid"].iloc[0])
        cpi = cpi_map.get((year, constructor_id), 0.5) # Default to mid-field if unknown
        static_feat = np.array([grid, float(total_laps), cpi], dtype=np.float32)
        
        # Target: Final Position (Regression)
        final_pos = float(df["positionOrder"].iloc[0])
        target = np.array([final_pos], dtype=np.float32)

        # Get Standard Pit Time for this team (for Future Plan)
        std_pit_ms = team_pit_standards.get((year, constructor_id), 24000.0)
        
        # Pit Stops for this driver
        pit_df = pits[(pits["raceId"] == race_id) & (pits["driverId"] == driver_id)].sort_values("lap")
        actual_pit_laps = pit_df["lap"].tolist()
        
        # Create examples at different cut points (25%, 50%, 75% of race)
        cut_points = set()
        for frac in [0.25, 0.5, 0.75]:
            cp = int(total_laps * frac)
            if cp >= min_cut_lap and cp < total_laps:
                cut_points.add(cp)
        
        for cp in sorted(cut_points):
            # --- A. Sequence Input (HISTORY) ---
            # KEEP IT DIRTY: Use actual pace, actual pit durations
            seq_df = df[df["lap"] <= cp]
            if len(seq_df) < min_cut_lap: continue
            
            seq_feats = np.stack([
                seq_df["lap"].values / float(total_laps), # Norm Lap
                seq_df["lap_ms"].values.astype(np.float32), # Raw Ms (Model learns pace)
                seq_df["pace_delta_ms"].values.astype(np.float32),
                seq_df["pace_z"].values.astype(np.float32), # Normalized Pace
                seq_df["position"].values.astype(np.float32),
                seq_df["on_pit_lap"].values.astype(np.float32),
                seq_df["pit_ms"].values.astype(np.float32), # ACTUAL pit time (History)
                seq_df["stint_index"].values.astype(np.float32),
                seq_df["laps_since_last_pit"].values.astype(np.float32),
            ], axis=-1)
            
            # --- B. Plan Input (FUTURE) ---
            # SANITIZE IT: Use team standard pit time
            future_pits = [p for p in actual_pit_laps if p > cp]
            remaining_stops = len(future_pits)
            total_laps_f = float(total_laps)
            
            if remaining_stops > 0:
                next_pit = future_pits[0] / total_laps_f
                last_pit = future_pits[-1] / total_laps_f
                segments = [cp] + future_pits + [total_laps]
                gaps = [(segments[i+1] - segments[i]) / total_laps_f for i in range(len(segments)-1)]
                mean_stint = float(np.mean(gaps))
            else:
                next_pit = 1.0
                last_pit = cp / total_laps_f
                mean_stint = (total_laps - cp) / total_laps_f
            
            max_stops = 4
            pit_lap_vec = np.ones(max_stops, dtype=np.float32)
            pit_dur_vec = np.zeros(max_stops, dtype=np.float32)
            
            for i in range(max_stops):
                if i < remaining_stops:
                    pit_lap_vec[i] = future_pits[i] / total_laps_f
                    # SANITIZED DURATION
                    pit_dur_vec[i] = std_pit_ms / 100000.0 
            
            plan_feat = np.concatenate([
                np.array([remaining_stops, next_pit, last_pit, mean_stint], dtype=np.float32),
                pit_lap_vec,
                pit_dur_vec
            ], axis=0)
            
            meta = (race_id, driver_id, cp, final_pos)
            examples.append((seq_feats, static_feat, plan_feat, target, meta))
            race_ids.append(race_id)
            
    return examples, race_ids

# =====================================================
# 4. Dataset & Model
# =====================================================
class StrategyDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        seq, static, plan, target, meta = self.examples[idx]
        return {
            "seq": torch.from_numpy(seq),
            "seq_len": torch.tensor(seq.shape[0], dtype=torch.long),
            "static": torch.from_numpy(static),
            "plan": torch.from_numpy(plan),
            "target": torch.from_numpy(target),
            "meta": meta,
        }

def collate_batch(batch):
    batch = sorted(batch, key=lambda x: x["seq_len"], reverse=True)
    seq_lens = torch.stack([b["seq_len"] for b in batch])
    max_len = int(seq_lens.max().item())
    feat_dim = batch[0]["seq"].shape[1]
    
    seq_tensor = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    for i, b in enumerate(batch):
        l = int(b["seq_len"].item())
        seq_tensor[i, :l] = b["seq"]
        
    static = torch.stack([b["static"] for b in batch])
    plan = torch.stack([b["plan"] for b in batch])
    target = torch.stack([b["target"] for b in batch]).view(-1) # Shape (Batch,)
    metas = [b["meta"] for b in batch]
    return seq_tensor, seq_lens, static, plan, target, metas

class StrategyLSTM(nn.Module):
    def __init__(self, seq_input_dim, static_dim, plan_dim, hidden_dim=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=seq_input_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0.0, bidirectional=True)
        
        self.static_proj = nn.Linear(static_dim, hidden_dim)
        self.plan_proj = nn.Linear(plan_dim, hidden_dim)
        
        # Regression Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 1) # Output 1 scalar (Position)
        )
        
    def forward(self, seq, seq_lens, static, plan):
        packed = nn.utils.rnn.pack_padded_sequence(seq, seq_lens.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, (h_n, c_n) = self.lstm(packed)
        
        # Bidirectional: Concat last hidden states
        h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        
        static_h = torch.relu(self.static_proj(static))
        plan_h = torch.relu(self.plan_proj(plan))
        
        concat = torch.cat([h_last, static_h, plan_h], dim=-1)
        out = self.head(concat).squeeze(-1)
        return out

# =====================================================
# 5. Training Loop
# =====================================================
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model():
    set_seeds(42)
    print("Loading data...")
    results, laps, pits, races, qualifying = load_core()
    
    print("Building features...")
    res = build_base_results(results, races)
    cpi_map = calculate_car_performance(results, races, qualifying)
    team_standards = get_team_pit_standards(pits, races, results)
    
    print("Generating examples...")
    examples, race_ids = make_examples(res, laps, pits, cpi_map, team_standards)
    
    if not examples:
        raise RuntimeError("No training examples built; check data filters.")
    
    # Split by Race
    unique_races = sorted(set(race_ids))
    random.shuffle(unique_races)
    split = int(len(unique_races) * 0.8)
    train_races = set(unique_races[:split])
    
    train_examples = [ex for ex, r in zip(examples, race_ids) if r in train_races]
    valid_examples = [ex for ex, r in zip(examples, race_ids) if r not in train_races]
    
    print(f"Train samples: {len(train_examples)} | Val samples: {len(valid_examples)}")
    
    train_ds = StrategyDataset(train_examples)
    valid_ds = StrategyDataset(valid_examples)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_ds, batch_size=128, shuffle=False, collate_fn=collate_batch)
    
    # Dimensions
    seq_input_dim = train_examples[0][0].shape[1]
    static_dim = train_examples[0][1].shape[0] # Should be 3 (Grid, Laps, CPI)
    plan_dim = train_examples[0][2].shape[0]
    
    model = StrategyLSTM(seq_input_dim, static_dim, plan_dim, hidden_dim=64, num_layers=1, dropout=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.L1Loss() # MAE Loss for Regression
    
    best_val_mae = float("inf")
    best_state = None
    
    print("Starting training...")
    for epoch in range(20):
        model.train()
        total_loss = 0.0
        total_count = 0
        
        for seq, seq_lens, static, plan, target, metas in train_loader:
            seq, seq_lens = seq.to(device), seq_lens.to(device)
            static, plan = static.to(device), plan.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            preds = model(seq, seq_lens, static, plan)
            loss = criterion(preds, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += float(loss.item()) * target.size(0)
            total_count += target.size(0)
            
        train_mae = total_loss / max(total_count, 1)
        
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for seq, seq_lens, static, plan, target, metas in valid_loader:
                seq, seq_lens = seq.to(device), seq_lens.to(device)
                static, plan = static.to(device), plan.to(device)
                target = target.to(device)
                
                preds = model(seq, seq_lens, static, plan)
                # Clamp prediction for sanity (1-20)
                preds = torch.clamp(preds, 1.0, 20.0)
                loss = criterion(preds, target)
                
                val_loss += float(loss.item()) * target.size(0)
                val_count += target.size(0)
        
        val_mae = val_loss / max(val_count, 1)
        print(f"Epoch {epoch+1:02d}: Train MAE={train_mae:.3f} | Val MAE={val_mae:.3f}")
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
    # Save
    if best_state is None:
        best_state = model.state_dict()
        
    torch.save(best_state, ARTIFACTS_DIR / "strategy_lstm.pt")
    
    cfg = {
        "seq_input_dim": int(seq_input_dim),
        "static_dim": int(static_dim),
        "plan_dim": int(plan_dim),
        "hidden_dim": 64,
        "num_layers": 1,
        "features": ["grid", "total_laps", "cpi"]
    }
    with open(ARTIFACTS_DIR / "strategy_lstm_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
        
    print(f"Saved model. Best Val MAE: {best_val_mae:.3f}")

if __name__ == "__main__":
    train_model()