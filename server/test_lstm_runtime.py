from pathlib import Path
import pandas as pd

from lstm_runtime import load_lstm_model, predict_finish_from_history

DATA_DIR = Path("../data")

def pick_examples(n=5):
    results = pd.read_csv(DATA_DIR / "results.csv")
    laps = pd.read_csv(DATA_DIR / "lap_times.csv")[["raceId", "lap"]]
    race_laps = laps.groupby("raceId", as_index=False)["lap"].max().rename(columns={"lap": "total_laps"})
    df = results.merge(race_laps, on="raceId", how="left")
    df = df[(df["positionOrder"] > 0) & (df["grid"] > 0)]
    df = df[df["total_laps"] >= 30]
    df = df[df["year"] >= 2014] if "year" in df.columns else df
    df = df.sample(n=min(n, len(df)), random_state=42)
    return df[["raceId", "driverId", "grid", "positionOrder", "total_laps"]].reset_index(drop=True)

def main():
    model, device = load_lstm_model()
    print("Loaded LSTM model on device:", device)
    examples = pick_examples(5)

    rows = []
    for _, row in examples.iterrows():
        race_id = int(row["raceId"])
        driver_id = int(row["driverId"])
        grid = int(row["grid"])
        finish = int(row["positionOrder"])
        total_laps = int(row["total_laps"])

        for lap in [5, 10, int(total_laps * 0.5), total_laps - 5]:
            if lap <= 0 or lap > total_laps:
                continue
            try:
                pred = predict_finish_from_history(model, device, race_id=race_id, driver_id=driver_id, current_lap=lap)
                rows.append({
                    "raceId": race_id,
                    "driverId": driver_id,
                    "grid": grid,
                    "actual_finish": finish,
                    "lap": lap,
                    "pred_finish": pred,
                })
            except Exception as e:
                rows.append({
                    "raceId": race_id,
                    "driverId": driver_id,
                    "grid": grid,
                    "actual_finish": finish,
                    "lap": lap,
                    "pred_finish": f"ERROR: {e}",
                })

    out = pd.DataFrame(rows)
    print()
    print("LSTM demo: predictions vs actual")
    print(out.to_string(index=False))

    numeric = out[out["pred_finish"].apply(lambda x: isinstance(x, (int, float)))]
    if not numeric.empty:
        numeric = numeric.copy()
        numeric["abs_err"] = (numeric["pred_finish"] - numeric["actual_finish"]).abs()
        mae = numeric["abs_err"].mean()
        print()
        print(f"Mean absolute error over these prefixes: {mae:.3f}")

if __name__ == "__main__":
    main()
