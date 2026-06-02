"""
Secondo Muro — XGBoost supervisionato con pipeline Sklearn (ColumnTransformer).

Carica il modello da models/xgb_sentinel_v1.pkl e calcola P(frode) su singola transazione.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

DEFAULT_MODEL_PATH = "models/xgb_sentinel_v1.pkl"
V_FEATURE_COUNT = 28


class XGBoostScorer:
    """Wrapper inferenza per il classificatore XGBoost serializzato."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH) -> None:
        self._model_path = model_path
        self._pipeline: Pipeline | None = None
        self._feature_columns: list[str] = []
        self.is_loaded = False
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self._model_path):
            print(f"[XGBoostScorer] Modello non trovato: {self._model_path}")
            return

        artifact = joblib.load(self._model_path)
        if isinstance(artifact, dict):
            self._pipeline = artifact["pipeline"]
            self._feature_columns = artifact["feature_columns"]
        else:
            # Compatibilità: pipeline salvata direttamente
            self._pipeline = artifact
            self._feature_columns = _default_feature_columns()

        self.is_loaded = True
        print(f"[XGBoostScorer] Modello caricato da: {self._model_path}")

    def score(self, transaction: dict) -> float:
        if not self.is_loaded or self._pipeline is None:
            raise RuntimeError("Modello XGBoost non disponibile")

        row = self._build_feature_row(transaction)
        X = pd.DataFrame([row])[self._feature_columns]
        proba = self._pipeline.predict_proba(X)[0]
        fraud_class_index = _fraud_class_index(self._pipeline)
        return float(proba[fraud_class_index])

    def _build_feature_row(self, transaction: dict) -> dict[str, Any]:
        row: dict[str, Any] = {col: 0.0 for col in self._feature_columns}

        amount = float(transaction["amount"])
        timestamp = transaction["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        elif isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp, tz=timezone.utc)

        if "Amount" in row:
            row["Amount"] = amount
        if "amount" in row:
            row["amount"] = amount

        if "hour" in row:
            row["hour"] = float(timestamp.hour)
        if "day_of_week" in row:
            row["day_of_week"] = float(timestamp.weekday())

        for i in range(1, V_FEATURE_COUNT + 1):
            key = f"V{i}"
            if key in row and key in transaction:
                row[key] = float(transaction[key])

        for cat_key in ("country", "merchant_type", "card_type"):
            if cat_key in row:
                value = transaction.get(cat_key)
                row[cat_key] = str(value) if value is not None else "unknown"

        return row


def _default_feature_columns() -> list[str]:
    v_cols = [f"V{i}" for i in range(1, V_FEATURE_COUNT + 1)]
    return v_cols + ["Amount", "hour"]


def _fraud_class_index(pipeline: Pipeline) -> int:
    classifier = pipeline.named_steps.get("classifier") or pipeline.named_steps.get("xgb")
    if classifier is None:
        return 1
    classes = getattr(classifier, "classes_", None)
    if classes is None:
        return 1
    return int(np.where(classes == 1)[0][0])
