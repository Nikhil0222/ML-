from __future__ import annotations

from typing import Dict, List

from .data import FEATURE_COLUMNS
from .model import load_model, predict


def predict_from_feature_dicts(model_path: str, rows: List[Dict[str, float]]) -> List[float]:
    model = load_model(model_path)
    matrix = [[row[col] for col in FEATURE_COLUMNS] for row in rows]
    return predict(model, matrix)
