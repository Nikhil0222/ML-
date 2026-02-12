#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from air_quality_ml.data import FEATURE_COLUMNS


def generate_air_quality_rows(rows: int, seed: int = 42):
    rng = random.Random(seed)
    for _ in range(rows):
        pm2_5 = max(2.0, min(180.0, rng.gauss(35, 12)))
        pm10 = max(5.0, min(300.0, pm2_5 * 1.5 + rng.gauss(8, 6)))
        no2 = max(3.0, min(120.0, rng.gauss(30, 10)))
        so2 = max(1.0, min(60.0, rng.gauss(12, 5)))
        co = max(0.1, min(4.5, rng.gauss(0.9, 0.3)))
        o3 = max(2.0, min(200.0, rng.gauss(45, 14)))
        temperature = max(-5.0, min(48.0, rng.gauss(24, 7)))
        humidity = max(10.0, min(100.0, rng.gauss(58, 16)))
        wind_speed = max(0.1, min(14.0, rng.gauss(3.2, 1.3)))

        noise = rng.gauss(0, 5)
        aqi = max(
            10.0,
            min(
                500.0,
                0.75 * pm2_5
                + 0.2 * pm10
                + 0.35 * no2
                + 0.25 * so2
                + 9.0 * co
                + 0.12 * o3
                - 0.18 * wind_speed
                + 0.06 * humidity
                + noise,
            ),
        )

        yield {
            "pm2_5": round(pm2_5, 4),
            "pm10": round(pm10, 4),
            "no2": round(no2, 4),
            "so2": round(so2, 4),
            "co": round(co, 4),
            "o3": round(o3, 4),
            "temperature": round(temperature, 4),
            "humidity": round(humidity, 4),
            "wind_speed": round(wind_speed, 4),
            "aqi": round(aqi, 4),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic air quality data")
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--output", type=str, default="data/air_quality_sample.csv")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[*FEATURE_COLUMNS, "aqi"])
        writer.writeheader()
        for row in generate_air_quality_rows(args.rows, args.seed):
            writer.writerow(row)

    print(f"Saved dataset to {output} ({args.rows} rows)")


if __name__ == "__main__":
    main()
