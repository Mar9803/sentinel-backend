"""Metriche globali del flusso live (in-memory, sessione dashboard)."""

from __future__ import annotations

from dataclasses import dataclass

from src.web.stream_generator import StreamProfile


@dataclass
class StreamMetrics:
    transactions_analyzed: int = 0
    losses_avoided_eur: float = 0.0
    damages_eur: float = 0.0
    vanity_accuracy_pct: float = 100.0

    def reset(self) -> None:
        self.transactions_analyzed = 0
        self.losses_avoided_eur = 0.0
        self.damages_eur = 0.0
        self.vanity_accuracy_pct = 100.0

    def apply_tick(
        self,
        *,
        amount: float,
        decision: str,
        profile: str,
    ) -> None:
        self.transactions_analyzed += 1
        is_block = decision == "BLOCK"
        is_fraud_profile = profile in (
            StreamProfile.SOPHISTICATED_FRAUD.value,
            StreamProfile.COARSE_ATTACK.value,
            StreamProfile.ZERO_DAY.value,
        )

        if is_block and is_fraud_profile:
            self.losses_avoided_eur += amount
            self.vanity_accuracy_pct = min(100.0, self.vanity_accuracy_pct + 0.8)

        elif is_block and profile == StreamProfile.LEGITIMATE.value:
            self.vanity_accuracy_pct = max(72.0, self.vanity_accuracy_pct - 2.0)

        elif not is_block and profile == StreamProfile.ZERO_DAY.value:
            self.damages_eur += amount * 2.5
            self.vanity_accuracy_pct = max(12.0, self.vanity_accuracy_pct - 18.0)

        elif not is_block and is_fraud_profile:
            self.damages_eur += amount
            self.vanity_accuracy_pct = max(25.0, self.vanity_accuracy_pct - 10.0)

        elif not is_block and profile == StreamProfile.LEGITIMATE.value:
            self.vanity_accuracy_pct = min(100.0, self.vanity_accuracy_pct + 0.15)

        else:
            self.vanity_accuracy_pct = max(50.0, self.vanity_accuracy_pct - 0.5)


stream_metrics = StreamMetrics()
