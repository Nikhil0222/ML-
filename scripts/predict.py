#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from air_quality_ml.data import load_feature_rows
from air_quality_ml.inference import predict_from_feature_dicts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict AQI from CSV feature rows")
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--data", required=True, type=str)
    parser.add_argument("--head", default=10, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_feature_rows(args.data)
    preds = predict_from_feature_dicts(args.model, rows)

    for idx, (row, pred) in enumerate(zip(rows, preds)):
        if idx >= args.head:
            break
        print(f"row={idx + 1:03d} predicted_aqi={pred:.4f} features={row}")


if __name__ == "__main__":
    main()
