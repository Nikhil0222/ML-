from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import List


@dataclass
class TrainedModel:
    feature_means: List[float]
    feature_stds: List[float]
    weights: List[float]
    bias: float


@dataclass
class EvaluationResult:
    mae: float
    rmse: float
    r2: float


def _normalize_row(row: List[float], means: List[float], stds: List[float]) -> List[float]:
    return [(v - m) / s if s else 0.0 for v, m, s in zip(row, means, stds)]


def train_linear_regression(
    X_train: List[List[float]], y_train: List[float], epochs: int = 2000, lr: float = 0.02
) -> TrainedModel:
    feature_count = len(X_train[0])
    means = [sum(row[i] for row in X_train) / len(X_train) for i in range(feature_count)]
    stds = []
    for i in range(feature_count):
        var = sum((row[i] - means[i]) ** 2 for row in X_train) / len(X_train)
        stds.append(math.sqrt(var) if var > 0 else 1.0)

    Xn = [_normalize_row(row, means, stds) for row in X_train]
    weights = [0.0] * feature_count
    bias = 0.0

    n = float(len(Xn))
    for _ in range(epochs):
        grad_w = [0.0] * feature_count
        grad_b = 0.0
        for x, y in zip(Xn, y_train):
            pred = sum(w * xi for w, xi in zip(weights, x)) + bias
            err = pred - y
            for j in range(feature_count):
                grad_w[j] += err * x[j]
            grad_b += err
        for j in range(feature_count):
            weights[j] -= lr * (2.0 / n) * grad_w[j]
        bias -= lr * (2.0 / n) * grad_b

    return TrainedModel(feature_means=means, feature_stds=stds, weights=weights, bias=bias)


def predict(model: TrainedModel, X: List[List[float]]) -> List[float]:
    Xn = [_normalize_row(row, model.feature_means, model.feature_stds) for row in X]
    return [sum(w * xi for w, xi in zip(model.weights, x)) + model.bias for x in Xn]


def evaluate_model(model: TrainedModel, X_test: List[List[float]], y_test: List[float]) -> EvaluationResult:
    preds = predict(model, X_test)
    n = len(y_test)
    mae = sum(abs(p - y) for p, y in zip(preds, y_test)) / n
    mse = sum((p - y) ** 2 for p, y in zip(preds, y_test)) / n
    mean_y = sum(y_test) / n
    total_var = sum((y - mean_y) ** 2 for y in y_test)
    residual_var = sum((y - p) ** 2 for y, p in zip(y_test, preds))
    r2 = 1.0 - (residual_var / total_var if total_var else 0.0)
    return EvaluationResult(mae=mae, rmse=math.sqrt(mse), r2=r2)


def save_model(model: TrainedModel, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(model), f, indent=2)


def load_model(path: str) -> TrainedModel:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return TrainedModel(**payload)
