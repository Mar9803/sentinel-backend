"""
Sentinel Fraud Wrapper — interfaccia unificata per inferenza transazionale in tempo reale.

FastAPI parlerà esclusivamente con FraudWrapper.predict_all().
Il Primo Muro (regole statiche) usa uno store in-memory come baseline
prima dell'integrazione con PostgreSQL (Feature Store offline).
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# --- Costanti regole (allineate a docsAgent/2-Backend-ML.md) ---

VELOCITY_WINDOW_SECONDS = 60
VELOCITY_MAX_TRANSACTIONS = 3  # trigger se > 3 (i.e. la 4ª transazione)
AMOUNT_STD_THRESHOLD = 3.0
COMMERCIAL_JET_SPEED_KMH = 850.0
MIN_FLIGHT_BUFFER_MINUTES = 30.0

# Coordinate approssimative (capitale/centroide) per Impossible Travel
COUNTRY_COORDS: dict[str, tuple[float, float]] = {
    "IT": (41.9028, 12.4964),
    "US": (38.9072, -77.0369),
    "GB": (51.5074, -0.1278),
    "FR": (48.8566, 2.3522),
    "DE": (52.5200, 13.4050),
    "CN": (39.9042, 116.4074),
    "SG": (1.3521, 103.8198),
    "JP": (35.6762, 139.6503),
    "BR": (-15.7939, -47.8828),
    "AU": (-35.2809, 149.1300),
}


class Decision(str, Enum):
    PASS = "PASS"
    ALERT = "ALERT"
    BLOCK = "BLOCK"


@dataclass
class StoredTransaction:
    transaction_id: str
    user_id: str
    amount: float
    country: str
    timestamp: datetime


@dataclass
class RuleResult:
    rule_name: str
    triggered: bool
    critical: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


class InMemoryTransactionStore:
    """Store in-memory per storico transazioni e statistiche importo utente."""

    def __init__(self) -> None:
        self._transactions: list[StoredTransaction] = []
        self._amounts_by_user: dict[str, list[float]] = {}

    @classmethod
    def with_seed_data(cls) -> InMemoryTransactionStore:
        """Baseline fittizia per simulare scenari frontend (es. Scenario C)."""
        store = cls()
        base = datetime(2026, 5, 31, 12, 0, 0, tzinfo=timezone.utc)

        # Pagamento legittimo a Roma ~5 min prima dell'attacco (Scenario C)
        store.seed_transaction(
            transaction_id="seed-001",
            user_id="user_scenario_c",
            amount=45.0,
            country="IT",
            timestamp=base,
        )

        # Storico importi per High-Risk Amount (media ~50€, std ~10€)
        for amount in [42.0, 48.0, 55.0, 51.0, 47.0, 53.0, 49.0, 46.0, 52.0, 50.0]:
            store._amounts_by_user.setdefault("user_normal", []).append(amount)

        return store

    def seed_transaction(
        self,
        transaction_id: str,
        user_id: str,
        amount: float,
        country: str,
        timestamp: datetime,
    ) -> None:
        tx = StoredTransaction(
            transaction_id=transaction_id,
            user_id=user_id,
            amount=amount,
            country=country.upper(),
            timestamp=_ensure_utc(timestamp),
        )
        self._transactions.append(tx)
        self._amounts_by_user.setdefault(user_id, []).append(amount)

    def get_recent_by_user(
        self, user_id: str, since: datetime, before: datetime | None = None
    ) -> list[StoredTransaction]:
        since = _ensure_utc(since)
        before = _ensure_utc(before) if before else None
        result = []
        for tx in self._transactions:
            if tx.user_id != user_id:
                continue
            if tx.timestamp < since:
                continue
            if before and tx.timestamp >= before:
                continue
            result.append(tx)
        return sorted(result, key=lambda t: t.timestamp)

    def get_last_before(self, user_id: str, before: datetime) -> StoredTransaction | None:
        before = _ensure_utc(before)
        candidates = [
            tx for tx in self._transactions if tx.user_id == user_id and tx.timestamp < before
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda t: t.timestamp)

    def get_amount_stats(self, user_id: str) -> tuple[float, float, int]:
        amounts = self._amounts_by_user.get(user_id, [])
        if not amounts:
            return 0.0, 0.0, 0
        mean = sum(amounts) / len(amounts)
        if len(amounts) < 2:
            return mean, 0.0, len(amounts)
        variance = sum((a - mean) ** 2 for a in amounts) / len(amounts)
        return mean, math.sqrt(variance), len(amounts)

    def append(self, tx: StoredTransaction) -> None:
        self._transactions.append(tx)
        self._amounts_by_user.setdefault(tx.user_id, []).append(tx.amount)


class StaticRulesEngine:
    """Primo Muro: euristiche a costo zero prima dell'inferenza ML."""

    def __init__(self, store: InMemoryTransactionStore) -> None:
        self._store = store

    def evaluate(self, tx: StoredTransaction) -> list[RuleResult]:
        return [
            self._velocity_check(tx),
            self._impossible_travel(tx),
            self._high_risk_amount(tx),
        ]

    def _velocity_check(self, tx: StoredTransaction) -> RuleResult:
        window_start = tx.timestamp.timestamp() - VELOCITY_WINDOW_SECONDS
        since = datetime.fromtimestamp(window_start, tz=timezone.utc)
        recent = self._store.get_recent_by_user(tx.user_id, since=since, before=tx.timestamp)
        total_in_window = len(recent) + 1  # include la transazione corrente
        triggered = total_in_window > VELOCITY_MAX_TRANSACTIONS

        return RuleResult(
            rule_name="velocity_check",
            triggered=triggered,
            critical=True,
            message=(
                f"Velocity Check: {total_in_window} transazioni negli ultimi "
                f"{VELOCITY_WINDOW_SECONDS}s (soglia: >{VELOCITY_MAX_TRANSACTIONS})"
            ),
            details={
                "total_in_window": total_in_window,
                "prior_in_window": len(recent),
                "window_seconds": VELOCITY_WINDOW_SECONDS,
                "threshold": VELOCITY_MAX_TRANSACTIONS,
            },
        )

    def _impossible_travel(self, tx: StoredTransaction) -> RuleResult:
        previous = self._store.get_last_before(tx.user_id, tx.timestamp)

        if previous is None:
            return RuleResult(
                rule_name="impossible_travel",
                triggered=False,
                critical=True,
                message="Impossible Travel: nessuna transazione precedente per l'utente",
                details={"previous_country": None},
            )

        if previous.country == tx.country:
            return RuleResult(
                rule_name="impossible_travel",
                triggered=False,
                critical=True,
                message="Impossible Travel: stesso paese della transazione precedente",
                details={
                    "previous_country": previous.country,
                    "current_country": tx.country,
                },
            )

        elapsed_minutes = (tx.timestamp - previous.timestamp).total_seconds() / 60.0
        min_travel_minutes = _min_flight_time_minutes(previous.country, tx.country)
        triggered = elapsed_minutes < min_travel_minutes

        return RuleResult(
            rule_name="impossible_travel",
            triggered=triggered,
            critical=True,
            message=(
                f"Impossible Travel: {previous.country} → {tx.country} "
                f"in {elapsed_minutes:.1f} min (minimo aereo: {min_travel_minutes:.0f} min)"
            ),
            details={
                "previous_country": previous.country,
                "current_country": tx.country,
                "elapsed_minutes": round(elapsed_minutes, 2),
                "min_travel_minutes": round(min_travel_minutes, 2),
                "previous_transaction_id": previous.transaction_id,
            },
        )

    def _high_risk_amount(self, tx: StoredTransaction) -> RuleResult:
        mean, std, n = self._store.get_amount_stats(tx.user_id)

        if n < 2 or std == 0.0:
            return RuleResult(
                rule_name="high_risk_amount",
                triggered=False,
                critical=True,
                message="High-Risk Amount: storico insufficiente per calcolare la soglia",
                details={"history_size": n, "mean": mean, "std": std},
            )

        threshold = mean + AMOUNT_STD_THRESHOLD * std
        triggered = tx.amount > threshold

        return RuleResult(
            rule_name="high_risk_amount",
            triggered=triggered,
            critical=True,
            message=(
                f"High-Risk Amount: {tx.amount:.2f} "
                f"(soglia {AMOUNT_STD_THRESHOLD}σ = {threshold:.2f}, μ={mean:.2f}, σ={std:.2f})"
            ),
            details={
                "amount": tx.amount,
                "mean": round(mean, 2),
                "std": round(std, 2),
                "threshold": round(threshold, 2),
                "history_size": n,
            },
        )


class FraudWrapper:
    """
    Orchestratore unificato per inferenza transazionale singola.

    Flusso attuale (fase 1):
      1. Normalizza il JSON in ingresso
      2. Valuta le Regole Statiche (Primo Muro)
      3. Short-circuit su regola critica → BLOCK, ML bypassato
      4. (fase 2+) XGBoost, Isolation Forest, Autoencoder in parallelo
    """

    def __init__(self, store: InMemoryTransactionStore | None = None) -> None:
        self._store = store or InMemoryTransactionStore.with_seed_data()
        self._rules = StaticRulesEngine(self._store)

    def predict_all(self, transaction: dict) -> dict:
        tx = _parse_transaction(transaction)
        rule_results = self._rules.evaluate(tx)
        triggered_rules = [r for r in rule_results if r.triggered]
        critical_triggered = any(r.triggered and r.critical for r in rule_results)

        if critical_triggered:
            response = _build_response(
                tx=tx,
                decision=Decision.BLOCK,
                rule_results=rule_results,
                ml_bypassed=True,
                models=None,
            )
            self._store.append(tx)
            return response

        # Fase 2: qui verranno invocati XGBoost, IF e Autoencoder in parallelo
        response = _build_response(
            tx=tx,
            decision=Decision.PASS,
            rule_results=rule_results,
            ml_bypassed=False,
            models={
                "xgb": None,
                "isolation_forest": None,
                "autoencoder": None,
            },
        )
        self._store.append(tx)
        return response


# --- Helpers ---


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return _ensure_utc(value)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        return _ensure_utc(datetime.fromisoformat(normalized))
    raise ValueError(f"timestamp non valido: {value!r}")


def _parse_transaction(raw: dict) -> StoredTransaction:
    required = ("transaction_id", "user_id", "amount", "country", "timestamp")
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"Campi obbligatori mancanti: {missing}")

    return StoredTransaction(
        transaction_id=str(raw["transaction_id"]),
        user_id=str(raw["user_id"]),
        amount=float(raw["amount"]),
        country=str(raw["country"]).upper(),
        timestamp=_parse_timestamp(raw["timestamp"]),
    )


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _min_flight_time_minutes(country_a: str, country_b: str) -> float:
    coords_a = COUNTRY_COORDS.get(country_a)
    coords_b = COUNTRY_COORDS.get(country_b)

    if coords_a is None or coords_b is None:
        # Paese sconosciuto: soglia conservativa (es. Italia → USA ≈ 9h)
        return 480.0

    distance_km = _haversine_km(coords_a[0], coords_a[1], coords_b[0], coords_b[1])
    flight_hours = distance_km / COMMERCIAL_JET_SPEED_KMH
    return (flight_hours * 60.0) + MIN_FLIGHT_BUFFER_MINUTES


def _build_response(
    tx: StoredTransaction,
    decision: Decision,
    rule_results: list[RuleResult],
    ml_bypassed: bool,
    models: dict[str, Any] | None,
) -> dict:
    triggered = [r.rule_name for r in rule_results if r.triggered]
    rules_score = 1.0 if triggered else 0.0

    return {
        "transaction_id": tx.transaction_id,
        "user_id": tx.user_id,
        "decision": decision.value,
        "final_score": rules_score if ml_bypassed else 0.0,
        "rules": {
            "score": rules_score,
            "triggered": triggered,
            "ml_bypassed": ml_bypassed,
            "results": [
                {
                    "rule": r.rule_name,
                    "triggered": r.triggered,
                    "critical": r.critical,
                    "message": r.message,
                    "details": deepcopy(r.details),
                }
                for r in rule_results
            ],
        },
        "models": models,
    }
