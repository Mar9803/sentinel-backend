"""
Feature engineering condiviso per inferenza transazionale in tempo reale.

Il modello legacy Isolation Forest (sentinel_v1.pkl) è stato addestrato su
V1-V28 + Amount + feature grafo (pagerank, clustering, betweenness).
In serving online le feature grafo sono a 0.0 finché il Feature Store offline
non le arricchisce in batch.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

V_FEATURE_COUNT = 28
GRAPH_FEATURES = ("pagerank", "clustering", "betweenness")
LEGACY_IF_FEATURES = [f"V{i}" for i in range(1, V_FEATURE_COUNT + 1)] + [
    "Amount",
    *GRAPH_FEATURES,
]


def parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return _ensure_utc(value)
    if isinstance(value, str):
        return _ensure_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    raise ValueError(f"timestamp non valido: {value!r}")


def build_legacy_feature_row(transaction: dict) -> dict[str, float]:
    """Costruisce il vettore atteso da models/sentinel_v1.pkl (Isolation Forest)."""
    row: dict[str, float] = {col: 0.0 for col in LEGACY_IF_FEATURES}
    row["Amount"] = float(transaction["amount"])

    for i in range(1, V_FEATURE_COUNT + 1):
        key = f"V{i}"
        if key in transaction:
            row[key] = float(transaction[key])

    for graph_key in GRAPH_FEATURES:
        if graph_key in transaction:
            row[graph_key] = float(transaction[graph_key])

    return row


def extract_numeric_vector(transaction: dict) -> list[float]:
    """Feature numeriche normalizzate per il mock Autoencoder (0-1)."""
    amount = float(transaction["amount"])
    amount_norm = min(1.0, max(0.0, amount / 500.0))

    v_values = [float(transaction.get(f"V{i}", 0.0)) for i in range(1, V_FEATURE_COUNT + 1)]
    v_min, v_max = min(v_values), max(v_values)
    if v_max - v_min > 1e-9:
        v_norm = [(v - v_min) / (v_max - v_min) for v in v_values]
    else:
        v_norm = [0.0] * V_FEATURE_COUNT

    timestamp = parse_timestamp(transaction["timestamp"])
    hour_norm = timestamp.hour / 23.0

    return v_norm + [amount_norm, hour_norm]


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
