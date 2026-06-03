"""
Generatore di transazioni dinamiche per la simulazione a flusso continuo Sentinel.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "creditcard.csv"
COARSE_USER_ID = "user-coarse"


class StreamProfile(str, Enum):
    LEGITIMATE = "legitimate"
    SOPHISTICATED_FRAUD = "sophisticated_fraud"
    ZERO_DAY = "zero_day"
    COARSE_ATTACK = "coarse_attack"


@dataclass
class StreamConfig:
    interval_ms: int = 1000
    profiles: tuple[StreamProfile, ...] = (
        StreamProfile.LEGITIMATE,
        StreamProfile.LEGITIMATE,
        StreamProfile.SOPHISTICATED_FRAUD,
        StreamProfile.ZERO_DAY,
        StreamProfile.COARSE_ATTACK,
    )


def _load_fraud_v_features() -> dict[str, float]:
    try:
        df = pd.read_csv(DATA_PATH)
        row = df.loc[df["Class"] == 1].iloc[0]
        return {f"V{i}": float(row[f"V{i}"]) for i in range(1, 29)}
    except Exception:
        return {f"V{i}": float((i % 7) * 1.8 - 4.0) for i in range(1, 29)}


FRAUD_V_FEATURES = _load_fraud_v_features()
ZERO_DAY_V_FEATURES = {f"V{i}": float((i % 11) * 0.85 - 3.2) for i in range(1, 29)}


class TransactionStreamGenerator:
    def __init__(self, config: StreamConfig | None = None) -> None:
        self.config = config or StreamConfig()
        self._tick = 0
        self._clock = datetime.now(timezone.utc)
        self._running = False
        self._coarse_primed = False

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def reset(self) -> None:
        self._tick = 0
        self._clock = datetime.now(timezone.utc)
        self._running = False
        self._coarse_primed = False

    def next_transaction(self) -> dict[str, Any]:
        self._tick += 1
        self._clock += timedelta(seconds=self.config.interval_ms / 1000)
        profile = self._pick_profile()
        tx_id = f"stream-{uuid.uuid4().hex[:12]}"

        builders = {
            StreamProfile.LEGITIMATE: self._build_legitimate,
            StreamProfile.SOPHISTICATED_FRAUD: self._build_sophisticated_fraud,
            StreamProfile.ZERO_DAY: self._build_zero_day,
            StreamProfile.COARSE_ATTACK: self._build_coarse_attack,
        }
        payload = builders[profile](tx_id)
        payload["_stream_meta"] = {
            "tick": self._tick,
            "profile": profile.value,
            "label": profile.value.replace("_", " ").title(),
        }
        return payload

    def _pick_profile(self) -> StreamProfile:
        profiles = self.config.profiles
        return profiles[(self._tick - 1) % len(profiles)]

    def _build_legitimate(self, tx_id: str) -> dict[str, Any]:
        return {
            "transaction_id": tx_id,
            "user_id": f"user-legit-{self._tick % 3}",
            "amount": round(35.0 + (self._tick % 5) * 4.5, 2),
            "country": "IT",
            "timestamp": self._clock,
        }

    def _build_sophisticated_fraud(self, tx_id: str) -> dict[str, Any]:
        payload = {
            "transaction_id": tx_id,
            "user_id": "user-fraud-xgb",
            "amount": 80.0 + (self._tick % 3) * 5,
            "country": "IT",
            "timestamp": self._clock,
        }
        payload.update(FRAUD_V_FEATURES)
        return payload

    def _build_zero_day(self, tx_id: str) -> dict[str, Any]:
        payload = {
            "transaction_id": tx_id,
            "user_id": "user-zero-day",
            "amount": 42.0,
            "country": "DE",
            "timestamp": self._clock,
        }
        payload.update(ZERO_DAY_V_FEATURES)
        return payload

    def _build_coarse_attack(self, tx_id: str) -> dict[str, Any]:
        if not self._coarse_primed:
            self._coarse_primed = True
            return {
                "transaction_id": tx_id,
                "user_id": COARSE_USER_ID,
                "amount": 45.0,
                "country": "IT",
                "timestamp": self._clock,
            }
        return {
            "transaction_id": tx_id,
            "user_id": COARSE_USER_ID,
            "amount": 100.0 + self._tick * 2,
            "country": "SG",
            "timestamp": self._clock,
        }
