from pathlib import Path
import argparse
import pandas as pd

from lstm_runtime import load_lstm_model, predict_finish_from_history

DATA_DIR = Path("../data")

def get_driver_info(race_id: int, driver_id: int):
    results = pd.read_csv(DATA_DIR / "results.csv")
    row = results[(results["raceId"] == race_id) & (results["driverId"] == driver_id)]
    if row.empty:
        raise ValueError(f"No result found for raceId={race_id}, driverId={driver_id}")
    row = row.iloc[0]
    grid = int(row["grid"])
    finish = int(row["positionOrder"])
    return grid, finish

def get_total_laps(race_id: int) -> int:
    laps = pd.read_csv(DATA_DIR / "lap_times.csv")[["raceId", "lap"]]
    race_rows = laps[laps["raceId"] == race_id]
    if race_rows.empty:
        raise ValueError(f"No lap data for raceId={race_id}")
    return int(race_rows["lap"].max())

def choose_laps(total_laps: int):
    candidates = {5, 10, int(total_laps * 0.25), int(total_laps * 0.5), int(total_laps * 0.75), total_laps - 2}
    laps = sorted({l for l in candidates if 1 < l <= total_laps})
    return laps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raceId", type=int, default=1034)
    parser.add_argument("--driverId", type=int, default=844)
    parser.add_argument(
        "--laps",
        type=str,
        default="",
        help="Comma-separated lap numbers to evaluate, e.g. '5,10,20,40'. If empty, use auto selection.",
    )
    args = parser.parse_args()


    race_id = args.raceId
    driver_id = args.driverId

    model, device = load_lstm_model()
    print(f"Loaded LSTM model on device: {device}")

    grid, finish = get_driver_info(race_id, driver_id)
    total_laps = get_total_laps(race_id)

    if args.laps.strip():
        raw = [x.strip() for x in args.laps.split(",") if x.strip()]
        laps_to_eval = sorted({int(x) for x in raw})
    else:
        laps_to_eval = choose_laps(total_laps)

    laps_to_eval = [lap for lap in laps_to_eval if 1 < lap <= total_laps]


    print()
    print(f"Race {race_id}, driver {driver_id}")
    print(f"Grid position:     {grid}")
    print(f"Actual finish:     {finish}")
    print(f"Total race laps:   {total_laps}")
    print()

    rows = []
    for lap in laps_to_eval:
        pred = predict_finish_from_history(model, device, race_id=race_id, driver_id=driver_id, current_lap=lap)
        abs_err = abs(pred - finish)
        baseline = grid
        baseline_err = abs(baseline - finish)
        rows.append({
            "lap": lap,
            "pred_finish": float(pred),
            "actual_finish": finish,
            "abs_error": float(abs_err),
            "baseline_grid": baseline,
            "baseline_error": float(baseline_err),
        })

    df = pd.DataFrame(rows)
    print("Lap-by-lap prediction vs actual")
    print(df.to_string(index=False))

    mae = df["abs_error"].mean()
    print()
    print(f"Average model error over these laps:   {mae:.3f}")
    print(f"Baseline (finish = grid) error:        {df['baseline_error'].iloc[0]:.3f}")

if __name__ == "__main__":
    main()
