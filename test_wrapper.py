"""
Verifica manuale del Primo Muro (Regole Statiche) via FraudWrapper.predict_all().

Scenari allineati a docsAgent/3-Frontend-UI.md:
  A — Ingegnere Finanziario  → regole NON devono scattare (PASS)
  B — Zero-Day Attack        → regole NON devono scattare (PASS)
  C — Frode Grossolana       → Impossible Travel (BLOCK + ml_bypassed)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.engine.wrapper import FraudWrapper, InMemoryTransactionStore

BASE = datetime(2026, 5, 31, 12, 0, 0, tzinfo=timezone.utc)


# --- Scenario A: "L'Attacco dell'Ingegnere Finanziario" ---
# Transazione che imita lo storico (importo leggermente diverso, stesso paese,
# nessuna velocity, nessun salto geografico). Il Primo Muro deve lasciar passare.

store_a = InMemoryTransactionStore()
for i, amount in enumerate([48.0, 51.0, 49.0, 52.0, 47.0]):
    store_a.seed_transaction(
        transaction_id=f"seed-a-{i}",
        user_id="user_scenario_a",
        amount=amount,
        country="IT",
        timestamp=BASE.replace(hour=8 + i),
    )

SCENARIO_A = {
    "transaction_id": "tx-scenario-a",
    "user_id": "user_scenario_a",
    "amount": 54.0,
    "country": "IT",
    "timestamp": BASE.replace(hour=14, minute=30),
}


# --- Scenario B: "L'Attacco Lampo (Zero-Day Attack)" ---
# Transazione superficialmente lecita: importo moderato, paese plausibile,
# nessuno storico sospetto. Le regole statiche non devono intervenire.

SCENARIO_B = {
    "transaction_id": "tx-scenario-b",
    "user_id": "user_scenario_b",
    "amount": 42.0,
    "country": "DE",
    "timestamp": BASE.replace(hour=15, minute=0),
}


# --- Scenario C: "La Frode Grossolana (Impossible Travel)" ---
# Il vero titolare ha pagato a Roma 5 minuti prima (seed in-memory su user_scenario_c).
# Prima transazione dall'Asia → Impossible Travel → BLOCK immediato.

SCENARIO_C = {
    "transaction_id": "tx-scenario-c",
    "user_id": "user_scenario_c",
    "amount": 22.0,
    "country": "SG",
    "timestamp": BASE.replace(minute=5),
}


def _print_result(label: str, expected: str, transaction: dict, result: dict) -> None:
    print("=" * 72)
    print(label)
    print(f"Atteso Primo Muro: {expected}")
    print("-" * 72)
    print("Input:")
    print(json.dumps(transaction, indent=2, default=str))
    print("-" * 72)
    print("Risposta predict_all():")
    print(json.dumps(result, indent=2, default=str))
    print()


def main() -> None:
    wrapper_a = FraudWrapper(store=store_a)
    wrapper_b = FraudWrapper()
    wrapper_c = FraudWrapper()  # include seed: pagamento IT per user_scenario_c

    scenarios = [
        (
            'SCENARIO A — "L\'Attacco dell\'Ingegnere Finanziario"',
            "PASS (nessuna regola triggerata, ML non bypassato)",
            wrapper_a,
            SCENARIO_A,
        ),
        (
            'SCENARIO B — "L\'Attacco Lampo (Zero-Day Attack)"',
            "PASS (nessuna regola triggerata, ML non bypassato)",
            wrapper_b,
            SCENARIO_B,
        ),
        (
            'SCENARIO C — "La Frode Grossolana (Impossible Travel)"',
            "BLOCK (impossible_travel, ml_bypassed=True)",
            wrapper_c,
            SCENARIO_C,
        ),
    ]

    print("Sentinel — Test Primo Muro (Regole Statiche)\n")

    for label, expected, wrapper, transaction in scenarios:
        result = wrapper.predict_all(transaction)
        _print_result(label, expected, transaction, result)


if __name__ == "__main__":
    main()
