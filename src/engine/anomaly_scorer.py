"""
Terzo Muro — Anomaly Detection (Isolation Forest + Autoencoder mock).

Isolation Forest: modello legacy models/sentinel_v1.pkl
Autoencoder: mock funzionale basato su errore di ricostruzione MSE (fase PyTorch/Keras).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.engine.feature_builder import (
    LEGACY_IF_FEATURES,
    build_legacy_feature_row,
    extract_numeric_vector,
)

DEFAULT_IF_MODEL_PATH = "models/sentinel_v1.pkl"

# Soglie allineate alla logica batch in app/main.py (decision_function <= -0.05 → frode)
IF_DECISION_THRESHOLD = 0.05
IF_SCORE_SCALE = 0.25
IF_RISK_THRESHOLD = 0.7

# Mock Autoencoder: MSE normalizzato vs baseline "legittima" (docsAgent/2-Backend-ML.md)
AE_RECONSTRUCTION_TAU = 0.35
AE_RISK_THRESHOLD = 0.7


@dataclass
class AnomalyScores:
    isolation_forest: float
    autoencoder: float

    @property
    def max_score(self) -> float:
        return max(self.isolation_forest, self.autoencoder)

    @property
    def should_block(self) -> bool:
        return (
            self.isolation_forest >= IF_RISK_THRESHOLD
            or self.autoencoder >= AE_RISK_THRESHOLD
        )


class IsolationForestScorer:
    """Carica e valuta il modello Isolation Forest legacy."""

    def __init__(self, model_path: str = DEFAULT_IF_MODEL_PATH) -> None:
        self._model_path = model_path
        self._model: IsolationForest | None = None
        self._feature_columns: list[str] = LEGACY_IF_FEATURES
        self.is_loaded = False
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self._model_path):
            print(f"[IsolationForestScorer] Modello non trovato: {self._model_path}")
            return

        model = joblib.load(self._model_path)
        if not isinstance(model, IsolationForest):
            print(
                f"[IsolationForestScorer] File non valido (atteso IsolationForest): "
                f"{self._model_path}"
            )
            return

        self._model = model
        if hasattr(model, "feature_names_in_"):
            self._feature_columns = list(model.feature_names_in_)
        self.is_loaded = True
        print(f"[IsolationForestScorer] Modello caricato da: {self._model_path}")

    def score(self, transaction: dict) -> float:
        if not self.is_loaded or self._model is None:
            raise RuntimeError("Modello Isolation Forest non disponibile")

        row = build_legacy_feature_row(transaction)
        X = pd.DataFrame([row])[self._feature_columns]
        prediction = int(self._model.predict(X)[0])
        decision_value = float(self._model.decision_function(X)[0])
        return _normalize_if_score(prediction, decision_value)


class AutoencoderMockScorer:
    """
    Mock Autoencoder: simula l'errore di ricostruzione MSE su feature numeriche.

    In produzione verrà sostituito da una pipeline PyTorch/Keras con MinMaxScaler
    isolato e soglia tau calibrata su transazioni legittime.
    """

    is_loaded = True

    def score(self, transaction: dict) -> float:
        features = np.array(extract_numeric_vector(transaction), dtype=float)
        # Baseline "legittima": manifold compatto attorno a zero (transazioni sane)
        reconstructed = np.zeros_like(features)
        mse = float(np.mean((features - reconstructed) ** 2))
        return float(min(1.0, mse / AE_RECONSTRUCTION_TAU))


class AnomalyScorer:
    """Terzo Muro: ensemble IF + Autoencoder mock."""

    def __init__(self, if_model_path: str = DEFAULT_IF_MODEL_PATH) -> None:
        self._if = IsolationForestScorer(model_path=if_model_path)
        self._ae = AutoencoderMockScorer()

    @property
    def is_loaded(self) -> bool:
        return self._if.is_loaded

    def score(self, transaction: dict) -> AnomalyScores:
        if_score = self._if.score(transaction) if self._if.is_loaded else 0.0
        ae_score = self._ae.score(transaction)
        return AnomalyScores(isolation_forest=if_score, autoencoder=ae_score)


def _normalize_if_score(prediction: int, decision_value: float) -> float:
    """
    Converte decision_function/predict IF in score 0-1 (più alto = più anomalo).

    predict:  1 = normale, -1 = anomalia
    decision_function: valori bassi (<= -0.05) indicano frode nel batch legacy.
    """
    raw = (IF_DECISION_THRESHOLD - decision_value) / IF_SCORE_SCALE
    score = float(max(0.0, min(1.0, raw)))
    if prediction == -1:
        score = max(score, IF_RISK_THRESHOLD)
    return round(score, 4)
