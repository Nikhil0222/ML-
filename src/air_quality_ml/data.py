from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple

TARGET_COLUMN = "aqi"
FEATURE_COLUMNS = [
    "pm2_5",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
    "temperature",
    "humidity",
    "wind_speed",
]


@dataclass
class DatasetSplit:
    X_train: List[List[float]]
    X_test: List[List[float]]
    y_train: List[float]
    y_test: List[float]


def load_dataset(path: str) -> Tuple[List[List[float]], List[float]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

        X: List[List[float]] = []
        y: List[float] = []
        for row in reader:
            X.append([float(row[c]) for c in FEATURE_COLUMNS])
            y.append(float(row[TARGET_COLUMN]))
    return X, y


def train_test_split(
    X: List[List[float]], y: List[float], test_ratio: float = 0.2
) -> DatasetSplit:
    split_idx = int(len(X) * (1.0 - test_ratio))
    return DatasetSplit(
        X_train=X[:split_idx],
        X_test=X[split_idx:],
        y_train=y[:split_idx],
        y_test=y[split_idx:],
    )


def load_feature_rows(path: str) -> List[Dict[str, float]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = set(FEATURE_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Dataset is missing feature columns: {sorted(missing)}")
        return [{col: float(row[col]) for col in FEATURE_COLUMNS} for row in reader]
