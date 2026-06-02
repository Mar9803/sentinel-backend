from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class TransactionInput(BaseModel):
    """Schema di ingresso per inferenza transazionale singola (POST /api/v1/predict)."""

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    transaction_id: str = Field(..., min_length=1, examples=["tx-scenario-c"])
    user_id: str = Field(..., min_length=1, examples=["user_scenario_c"])
    amount: float = Field(..., gt=0, examples=[22.0])
    country: str = Field(
        ...,
        validation_alias=AliasChoices("country", "location"),
        min_length=2,
        max_length=2,
        description="Codice paese ISO 3166-1 alpha-2 (accetta anche il campo alias `location`).",
        examples=["SG"],
    )
    timestamp: datetime = Field(..., examples=["2026-05-31T12:05:00Z"])
    merchant_type: str | None = Field(
        default=None,
        description="Tipo merchant (opzionale, riservato ai modelli ML in fase 2).",
    )
    card_type: str | None = Field(
        default=None,
        description="Tipo carta (opzionale, riservato ai modelli ML in fase 2).",
    )

    @field_validator("country")
    @classmethod
    def normalize_country(cls, value: str) -> str:
        return value.upper()

    def to_wrapper_dict(self) -> dict:
        """Converte lo schema nel dizionario atteso da FraudWrapper.predict_all()."""
        payload = {
            "transaction_id": self.transaction_id,
            "user_id": self.user_id,
            "amount": self.amount,
            "country": self.country,
            "timestamp": self.timestamp,
        }
        if self.merchant_type is not None:
            payload["merchant_type"] = self.merchant_type
        if self.card_type is not None:
            payload["card_type"] = self.card_type
        return payload


class RuleEvaluation(BaseModel):
    rule: str
    triggered: bool
    critical: bool
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class RulesBlock(BaseModel):
    score: float
    triggered: list[str]
    ml_bypassed: bool
    results: list[RuleEvaluation]


class ModelScores(BaseModel):
    xgb: float | None = None
    isolation_forest: float | None = None
    autoencoder: float | None = None


class PredictionResponse(BaseModel):
    """Risposta completa del motore regole + (in futuro) modelli ML."""

    transaction_id: str
    user_id: str
    decision: Literal["PASS", "ALERT", "BLOCK"]
    final_score: float
    rules: RulesBlock
    models: ModelScores | None = None
