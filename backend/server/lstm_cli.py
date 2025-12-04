import argparse
import json

from lstm_runtime import load_lstm_model, predict_finish_from_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raceId", type=int, required=True)
    parser.add_argument("--driverId", type=int, required=True)
    parser.add_argument("--currentLap", type=int, required=True)
    args = parser.parse_args()

    model, device = load_lstm_model()
    pred = predict_finish_from_history(
        model,
        device,
        race_id=args.raceId,
        driver_id=args.driverId,
        current_lap=args.currentLap,
    )

    print(json.dumps({"finishP50": float(pred)}))


if __name__ == "__main__":
    main()

