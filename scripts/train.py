#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from air_quality_ml.data import load_dataset, train_test_split
from air_quality_ml.model import evaluate_model, save_model, train_linear_regression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an AQI regression model")
    parser.add_argument("--data", required=True, type=str)
    parser.add_argument("--model-out", default="models/air_quality_model.json", type=str)
    parser.add_argument("--epochs", default=2500, type=int)
    parser.add_argument("--learning-rate", default=0.02, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X, y = load_dataset(args.data)
    split = train_test_split(X, y)

    model = train_linear_regression(
        split.X_train, split.y_train, epochs=args.epochs, lr=args.learning_rate
    )
    metrics = evaluate_model(model, split.X_test, split.y_test)

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, str(model_path))

    print(f"Model saved to: {model_path}")
    print(f"MAE : {metrics.mae:.4f}")
    print(f"RMSE: {metrics.rmse:.4f}")
    print(f"R2  : {metrics.r2:.4f}")


if __name__ == "__main__":
    main()
