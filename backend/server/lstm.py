import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
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
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

for fname in ["lap_times.csv", "pit_stops.csv", "results.csv"]:
    if not (DATA_DIR / fname).exists():
        raise FileNotFoundError(f"Missing {fname} in {DATA_DIR}")

lap_times = pd.read_csv(DATA_DIR / "lap_times.csv")
pit_stops = pd.read_csv(DATA_DIR / "pit_stops.csv")
results = pd.read_csv(DATA_DIR / "results.csv")

results = results[(results["positionOrder"].notna()) & (results["positionOrder"] > 0)]
results = results[results["grid"] > 0]
finish_pos = results[["raceId", "driverId", "positionOrder"]].rename(columns={"positionOrder": "finish_pos"})

race_laps = lap_times.groupby("raceId", as_index=False)["lap"].max().rename(columns={"lap": "total_laps"})
lap_times = lap_times.merge(race_laps, on="raceId", how="left")

pit_flags = pit_stops.groupby(["raceId", "driverId", "lap"], as_index=False)["stop"].count().rename(columns={"stop": "pit_flag"})
pit_flags["pit_flag"] = 1
lap_times = lap_times.merge(pit_flags, on=["raceId", "driverId", "lap"], how="left")
lap_times["pit_flag"] = lap_times["pit_flag"].fillna(0)

lap_times = lap_times.sort_values(["raceId", "driverId", "lap"])

def _build_sequences():
    seqs = []
    groups = lap_times.groupby(["raceId", "driverId"])
    for (race_id, driver_id), df in groups:
        df = df.sort_values("lap")
        y_row = finish_pos[(finish_pos["raceId"] == race_id) & (finish_pos["driverId"] == driver_id)]
        if y_row.empty:
            continue
        y = float(y_row["finish_pos"].iloc[0])
        y = max(1.0, min(20.0, y))
        df["total_laps"] = df["total_laps"].clip(lower=1)
        df["lap_norm"] = df["lap"] / df["total_laps"]
        df["lap_ms"] = pd.to_numeric(df["milliseconds"], errors="coerce")
        if df["lap_ms"].isna().all():
            continue
        df["lap_ms"] = df["lap_ms"].fillna(method="ffill").fillna(method="bfill").fillna(df["lap_ms"].median())
        df["gap_prev_lap_ms"] = df["lap_ms"].diff().fillna(0)
        pit_flag = df["pit_flag"].values
        stint = np.zeros_like(pit_flag, dtype=np.int64)
        current = 0
        for i in range(len(stint)):
            stint[i] = current
            if pit_flag[i] == 1:
                current += 1
        df["stint_idx"] = stint
        if "position" not in df.columns:
            continue
        df["position"] = pd.to_numeric(df["position"], errors="coerce").ffill().bfill()
        if df["position"].isna().all():
            continue
        x = df[["lap_norm", "position", "pit_flag", "stint_idx", "lap_ms", "gap_prev_lap_ms"]].values.astype("float32")
        if x.shape[0] < 3:
            continue
        seqs.append((race_id, driver_id, x, y))
    if not seqs:
        raise RuntimeError("No valid sequences built from lap_times and pit_stops")
    return seqs

class F1LstmDataset(Dataset):
    def __init__(self, seqs, max_prefixes_per_seq=5, min_lap=3, seed=42):
        self.samples = []
        self.rng = np.random.default_rng(seed)
        for race_id, driver_id, x, y in seqs:
            T = x.shape[0]
            if T <= min_lap + 1:
                continue
            cut_candidates = list(range(min_lap, T))
            if len(cut_candidates) > max_prefixes_per_seq:
                cut_candidates = list(self.rng.choice(cut_candidates, size=max_prefixes_per_seq, replace=False))
            for L in cut_candidates:
                x_prefix = x[:L]
                self.samples.append((race_id, x_prefix, y))
        if not self.samples:
            raise RuntimeError("No samples generated from sequences")
        self.max_len = max(s[1].shape[0] for s in self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        race_id, x, y = self.samples[idx]
        T = x.shape[0]
        pad_len = self.max_len - T
        if pad_len > 0:
            pad = np.zeros((pad_len, x.shape[1]), dtype=np.float32)
            x_padded = np.concatenate([x, pad], axis=0)
        else:
            x_padded = x
        length = T
        return torch.from_numpy(x_padded), torch.tensor(length, dtype=torch.long), torch.tensor(y, dtype=torch.float32), torch.tensor(race_id, dtype=torch.long)

def collate_fn(batch):
    xs, lengths, ys, race_ids = zip(*batch)
    xs = torch.stack(xs, dim=0)
    lengths = torch.stack(lengths, dim=0)
    ys = torch.stack(ys, dim=0)
    race_ids = torch.stack(race_ids, dim=0)
    return xs, lengths, ys, race_ids

class F1LstmRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h_last = h_n[-1]
        out = self.fc(h_last)
        return out.squeeze(-1)

def _train_val_split(seqs, val_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    race_ids = sorted({s[0] for s in seqs})
    rng.shuffle(race_ids)
    n_val = max(1, int(len(race_ids) * val_frac))
    val_ids = set(race_ids[:n_val])
    train_seqs = [s for s in seqs if s[0] not in val_ids]
    val_seqs = [s for s in seqs if s[0] in val_ids]
    if not train_seqs or not val_seqs:
        raise RuntimeError("Train/val split failed, not enough races")
    return train_seqs, val_seqs

def main():
    seqs = _build_sequences()
    train_seqs, val_seqs = _train_val_split(seqs)
    train_ds = F1LstmDataset(train_seqs)
    val_ds = F1LstmDataset(val_seqs, max_prefixes_per_seq=5, min_lap=3, seed=123)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = train_ds[0][0].shape[1]
    hidden_dim = 128
    num_layers = 2
    model = F1LstmRegressor(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()
    best_val = float("inf")
    patience = 8
    patience_left = patience
    max_epochs = 80
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        n_samples = 0
        for x, lengths, y, race_ids in train_loader:
            x = x.to(device)
            lengths = lengths.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred_raw = model(x, lengths)
            loss = criterion(pred_raw, y)
            loss.backward()
            optimizer.step()

            pred_clamped = torch.clamp(pred_raw.detach(), 1.0, 20.0)
            mae_batch = torch.mean(torch.abs(pred_clamped - y))

            total_loss += mae_batch.item() * x.size(0)
            n_samples += x.size(0)

        train_mae = total_loss / max(1, n_samples)

        model.eval()
        val_total = 0.0
        val_n = 0
        with torch.no_grad():
            for x, lengths, y, race_ids in val_loader:
                x = x.to(device)
                lengths = lengths.to(device)
                y = y.to(device)

                pred_raw = model(x, lengths)
                pred_clamped = torch.clamp(pred_raw, 1.0, 20.0)
                mae_batch = torch.mean(torch.abs(pred_clamped - y))

                val_total += mae_batch.item() * x.size(0)
                val_n += x.size(0)

        val_mae = val_total / max(1, val_n)
        print(f"epoch={epoch} train_mae={train_mae:.4f} val_mae={val_mae:.4f}")

        if val_mae + 1e-4 < best_val:
            best_val = val_mae
            patience_left = patience
            torch.save(model.state_dict(), ARTIFACTS_DIR / "finish_regressor_lstm.pt")
            cfg = {"input_dim": input_dim, "hidden_dim": hidden_dim, "num_layers": num_layers}
            with open(ARTIFACTS_DIR / "finish_regressor_lstm_config.json", "w") as f:
                json.dump(cfg, f)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping")
                break

if __name__ == "__main__":
    main()
