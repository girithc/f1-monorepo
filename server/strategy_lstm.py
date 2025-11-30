import os
from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

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

def load_core():
    results = pd.read_csv(DATA_DIR / "results.csv")
    laps = pd.read_csv(DATA_DIR / "lap_times.csv")
    laps = laps.rename(columns={"milliseconds": "lap_ms"})
    pits = pd.read_csv(DATA_DIR / "pit_stops.csv")
    races = pd.read_csv(DATA_DIR / "races.csv")
    return results, laps, pits, races

def build_base_results(results, races):
    res = results.merge(races[["raceId", "year", "round"]], on="raceId", how="left")
    res = res[(res["positionOrder"].notna()) & (res["positionOrder"] > 0)]
    res = res[res["grid"] > 0]
    finishers = (res.assign(classified=(res["positionOrder"] > 0).astype(int))
                    .groupby("raceId", as_index=False)["classified"].sum()
                    .rename(columns={"classified": "n_finishers"}))
    res = res.merge(finishers, on="raceId", how="left")
    res = res[res["n_finishers"] >= 16]
    res = res[res["year"] >= 2014]
    return res

def add_success_label(res, top_k=10):
    res = res.copy()
    res["success"] = (res["positionOrder"] <= top_k).astype(np.float32)
    return res

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

def make_examples(res, laps, pits, min_cut_lap=5):
    laps = build_lap_pace_features(laps)
    laps = build_pit_flags(laps, pits)
    laps = laps.merge(res[["raceId", "driverId", "grid", "positionOrder", "success", "laps"]], on=["raceId", "driverId"], how="inner")
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
        if total_laps <= 0:
            continue
        cut_points = set()
        for frac in [0.25, 0.5, 0.75]:
            cp = int(total_laps * frac)
            if cp >= min_cut_lap and cp < total_laps:
                cut_points.add(cp)
        if not cut_points:
            continue
        pit_df = pits[(pits["raceId"] == race_id) & (pits["driverId"] == driver_id)].sort_values("lap")
        actual_pit_laps = pit_df["lap"].tolist()
        actual_pit_durs = pit_df["milliseconds"].astype(np.float32).tolist()
        success = float(df["success"].iloc[0])
        grid = float(df["grid"].iloc[0])
        final_pos = float(df["positionOrder"].iloc[0])
        for cp in sorted(cut_points):
            seq_df = df[df["lap"] <= cp]
            seq_len = len(seq_df)
            if seq_len < min_cut_lap:
                continue
            seq_feats = np.stack([
                seq_df["lap"].values / float(total_laps),
                seq_df["lap_ms"].values.astype(np.float32),
                seq_df["pace_delta_ms"].values.astype(np.float32),
                seq_df["pace_z"].values.astype(np.float32),
                seq_df["position"].values.astype(np.float32),
                seq_df["on_pit_lap"].values.astype(np.float32),
                seq_df["pit_ms"].values.astype(np.float32),
                seq_df["stint_index"].values.astype(np.float32),
                seq_df["laps_since_last_pit"].values.astype(np.float32),
            ], axis=-1)
            idxs = [i for i, lap in enumerate(actual_pit_laps) if lap > cp]
            remaining_pits = [actual_pit_laps[i] for i in idxs]
            remaining_durs = [actual_pit_durs[i] for i in idxs]
            remaining_stops = len(remaining_pits)
            total_laps_f = float(total_laps)
            if remaining_stops > 0:
                next_pit = remaining_pits[0] / total_laps_f
                last_pit = remaining_pits[-1] / total_laps_f
                segments = [cp] + remaining_pits + [total_laps]
                gaps = []
                for i in range(len(segments) - 1):
                    gaps.append((segments[i + 1] - segments[i]) / total_laps_f)
                mean_stint = float(np.mean(gaps)) if gaps else 0.0
            else:
                next_pit = 1.0
                last_pit = cp / total_laps_f
                mean_stint = (total_laps - cp) / total_laps_f
            max_stops = 4
            pit_lap_vec = np.ones(max_stops, dtype=np.float32)
            pit_dur_vec = np.zeros(max_stops, dtype=np.float32)
            for i in range(max_stops):
                if i < remaining_stops:
                    pit_lap_vec[i] = remaining_pits[i] / total_laps_f
                    dur_norm = remaining_durs[i] / 100000.0
                    pit_dur_vec[i] = dur_norm
            plan_feat = np.concatenate([
                np.array([remaining_stops, next_pit, last_pit, mean_stint], dtype=np.float32),
                pit_lap_vec,
                pit_dur_vec
            ], axis=0)
            static_feat = np.array([grid, total_laps_f], dtype=np.float32)
            target = np.array([success], dtype=np.float32)
            meta = (race_id, driver_id, cp, final_pos)
            examples.append((seq_feats, static_feat, plan_feat, target, meta))
            race_ids.append(race_id)
    return examples, race_ids

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
    target = torch.stack([b["target"] for b in batch]).view(-1)
    metas = [b["meta"] for b in batch]
    return seq_tensor, seq_lens, static, plan, target, metas

class StrategyLSTM(nn.Module):
    def __init__(self, seq_input_dim, static_dim, plan_dim, hidden_dim=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=seq_input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0, bidirectional=True)
        self.static_proj = nn.Linear(static_dim, hidden_dim)
        self.plan_proj = nn.Linear(plan_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 1)
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

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model():
    set_seeds(42)
    results, laps, pits, races = load_core()
    res = build_base_results(results, races)
    res = add_success_label(res, top_k=10)
    examples, race_ids = make_examples(res, laps, pits)
    if not examples:
        raise RuntimeError("No training examples built; check data filters.")
    unique_races = sorted(set(race_ids))
    random.shuffle(unique_races)
    split = int(len(unique_races) * 0.8)
    train_races = set(unique_races[:split])
    train_examples = [ex for ex, r in zip(examples, race_ids) if r in train_races]
    valid_examples = [ex for ex, r in zip(examples, race_ids) if r not in train_races]
    train_ds = StrategyDataset(train_examples)
    valid_ds = StrategyDataset(valid_examples)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_ds, batch_size=128, shuffle=False, collate_fn=collate_batch)
    seq_input_dim = train_examples[0][0].shape[1]
    static_dim = train_examples[0][1].shape[0]
    plan_dim = train_examples[0][2].shape[0]
    model = StrategyLSTM(seq_input_dim, static_dim, plan_dim, hidden_dim=64, num_layers=1, dropout=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    best_val_loss = float("inf")
    best_state = None
    for epoch in range(20):
        model.train()
        total_loss = 0.0
        total_count = 0
        for seq, seq_lens, static, plan, target, metas in train_loader:
            seq = seq.to(device)
            seq_lens = seq_lens.to(device)
            static = static.to(device)
            plan = plan.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            logits = model(seq, seq_lens, static, plan)
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * target.size(0)
            total_count += target.size(0)
        train_loss = total_loss / max(total_count, 1)
        model.eval()
        val_loss = 0.0
        val_count = 0
        correct = 0
        with torch.no_grad():
            for seq, seq_lens, static, plan, target, metas in valid_loader:
                seq = seq.to(device)
                seq_lens = seq_lens.to(device)
                static = static.to(device)
                plan = plan.to(device)
                target = target.to(device)
                logits = model(seq, seq_lens, static, plan)
                loss = criterion(logits, target)
                val_loss += float(loss.item()) * target.size(0)
                val_count += target.size(0)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                correct += (preds == target).sum().item()
        val_loss = val_loss / max(val_count, 1)
        val_acc = correct / max(val_count, 1)
        print(f"epoch {epoch+1:02d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    if best_state is None:
        best_state = model.state_dict()
    torch.save(best_state, ARTIFACTS_DIR / "finish_regressor_lstm.pt")
    cfg = {
        "seq_input_dim": int(seq_input_dim),
        "static_dim": int(static_dim),
        "plan_dim": int(plan_dim),
        "hidden_dim": 64,
        "num_layers": 1,
        "success_top_k": 10
    }
    with open(ARTIFACTS_DIR / "finish_regressor_lstm_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print("saved artifacts/finish_regressor_lstm.pt and finish_regressor_lstm_config.json")

if __name__ == "__main__":
    train_model()
