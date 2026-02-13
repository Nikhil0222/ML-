# Air Quality ML Project (Pure Python)

A complete starter machine learning project for **AQI prediction**, implemented in pure Python (no external dependencies).

## Highlights

- Synthetic air-quality data generation
- Linear regression model from scratch using gradient descent
- Train/test split and evaluation (MAE, RMSE, R²)
- Model persistence to JSON
- Prediction CLI for batch inference

## Project structure

```
.
├── data/
├── models/
├── scripts/
│   ├── generate_data.py
│   ├── train.py
│   └── predict.py
├── src/
│   └── air_quality_ml/
│       ├── __init__.py
│       ├── data.py
│       ├── model.py
│       └── inference.py
├── requirements.txt
└── README.md
```

## Run the project

```bash
# 1) Generate synthetic data
python scripts/generate_data.py --rows 1000 --output data/air_quality_sample.csv

# 2) Train the model
python scripts/train.py --data data/air_quality_sample.csv --model-out models/air_quality_model.json

# 3) Predict using the trained model
python scripts/predict.py --model models/air_quality_model.json --data data/air_quality_sample.csv --head 5
```

## Dataset columns

Input features:
- `pm2_5`, `pm10`, `no2`, `so2`, `co`, `o3`, `temperature`, `humidity`, `wind_speed`

Target:
- `aqi`
