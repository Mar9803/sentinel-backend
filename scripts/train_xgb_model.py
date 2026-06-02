"""
Addestra il Secondo Muro (XGBoost) su creditcard.csv e salva la pipeline in models/.

Uso:
    python scripts/train_xgb_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "creditcard.csv"
MODEL_PATH = ROOT / "models" / "xgb_sentinel_v1.pkl"

V_COLS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = V_COLS + ["Amount", "hour"]


def main() -> None:
    print("Addestramento XGBoost (Secondo Muro)...")
    df = pd.read_csv(DATA_PATH)
    df["hour"] = (df["Time"] // 3600) % 24

    X = df[FEATURE_COLUMNS]
    y = df["Class"]

    negatives = int((y == 0).sum())
    positives = int((y == 1).sum())
    scale_pos_weight = negatives / positives
    print(f"Classi — sane: {negatives}, frodi: {positives}, scale_pos_weight: {scale_pos_weight:.2f}")

    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), FEATURE_COLUMNS)],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    scale_pos_weight=scale_pos_weight,
                    max_depth=6,
                    n_estimators=120,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric="logloss",
                ),
            ),
        ]
    )

    pipeline.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_columns": FEATURE_COLUMNS,
            "model_type": "xgboost",
        },
        MODEL_PATH,
    )
    print(f"Pipeline salvata in: {MODEL_PATH}")


if __name__ == "__main__":
    main()
