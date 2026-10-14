"""Request and response models.

Every field is constrained. `extra="forbid"` on request bodies means a typo in a
field name is a 422 rather than a silently ignored value, which matters when the
body carries the signals a score is computed from.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

EntityType = Literal["customer", "seller"]
DecisionValue = Literal["confirmed_fraud", "confirmed_good", "needs_info"]
# "Unknown" is a real band in mri.scoring: an entity no detector saw is not Low.
BandName = Literal["Low", "Medium", "High", "Critical", "Unknown"]

# Raw signals are produced on their own scales by upstream stages. ML, anomaly
# and temporal are already 0-1; graph is a normalised centrality blend. The
# ceiling is generous so a legitimately out-of-range value is reported by the
# calibration reference rather than rejected at the edge.
RawSignal = Annotated[float, Field(ge=0.0, le=1_000_000.0)]

EntityId = Annotated[
    str,
    Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_.:-]+$",
        description="Entity identifier. Alphanumerics plus _ . : - only.",
    ),
]


class SignalInput(BaseModel):
    """The four RAW signals, exactly as mri.scoring.score_entity expects them.

    `None` means "this detector did not see the entity". It is not zero: the
    signal is dropped from the weighted mean and `coverage` falls, instead of
    the entity being scored as low risk on evidence nobody gathered.
    """

    model_config = ConfigDict(extra="forbid")

    ml: RawSignal | None = Field(
        None, description="Supervised fraud probability in [0, 1] (mri.models.score)."
    )
    graph: RawSignal | None = Field(
        None, description="Raw graph_risk_score (mri.detection.graph)."
    )
    temporal: RawSignal | None = Field(
        None, description="Raw temporal_risk_score (mri.detection.temporal)."
    )
    anomaly: RawSignal | None = Field(
        None,
        description=(
            "RAW anomaly score: -IsolationForest.score_samples. Not the calibrated "
            "anomaly_score column, which the frozen reference has already transformed."
        ),
    )


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_type: EntityType
    entity_id: EntityId
    signals: SignalInput
    reason_codes: list[Annotated[str, Field(min_length=1, max_length=64)]] = Field(
        default_factory=list, max_length=20
    )

    @field_validator("entity_type", mode="before")
    @classmethod
    def _lower(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value


class SignalContribution(BaseModel):
    """One signal's share of the fused score. These sum to `fused_score`."""

    signal: str
    label: str
    weight: float = Field(ge=0.0, le=1.0, description="Published weight from mri.scoring.WEIGHTS.")
    effective_weight: float = Field(
        ge=0.0, le=1.0, description="Weight after renormalising over the observed signals."
    )
    raw: float | None
    calibrated: float | None = Field(
        None, ge=0.0, le=1.0, description="Raw value mapped through the frozen reference."
    )
    observed: bool
    contribution: float = Field(ge=0.0, le=1.0)
    share_of_score: float = Field(ge=0.0, le=1.0)


class CalibrationInfo(BaseModel):
    """Provenance of the frozen reference the score was computed against."""

    reference_path: str
    signals: dict[str, str] = Field(
        default_factory=dict, description="Signal -> calibration method (identity | quantile)."
    )
    frozen_on: dict[str, int] = Field(
        default_factory=dict, description="Signal -> population size the reference was fitted on."
    )


class ScoreResponse(BaseModel):
    entity_type: EntityType
    entity_id: str
    fused_score: float | None = Field(
        None, ge=0.0, le=1.0, description="Null when no signal was observed (coverage 0)."
    )
    band: BandName
    coverage: float = Field(
        ge=0.0, le=1.0, description="Fraction of the total weight mass actually observed."
    )
    signals_missing: list[str]
    contributions: list[SignalContribution]
    explanation: str
    model_version: str
    scoring_source: str
    calibration: CalibrationInfo
    request_id: str


class EntityResponse(BaseModel):
    entity_type: EntityType
    entity_id: str
    fused_score: float = Field(ge=0.0, le=1.0)
    band: BandName
    coverage: float = Field(ge=0.0, le=1.0)
    signals_missing: list[str]
    rank: int = Field(ge=1)
    population_size: int = Field(ge=0)
    contributions: list[SignalContribution]
    reason_codes: list[str]
    explanation: str
    evidence: dict[str, float]
    model_version: str
    decisions_recorded: int = Field(ge=0)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    model_version: str


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    detail: dict[str, Any]


class MetricsResponse(BaseModel):
    population_size: int = Field(ge=0)
    band_counts: dict[str, int]
    alert_rate: float = Field(ge=0.0, le=1.0)
    alert_bands: list[str]
    top_decile_threshold: float | None
    decisions: dict[str, int]
    labels: dict[str, int]
    evaluation: dict[str, Any] | None = Field(
        None, description="Contents of outputs/evaluation.json; null until mri.evaluation has run."
    )
    model_version: str
    scoring_source: str


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_type: EntityType
    entity_id: EntityId
    analyst: Annotated[str, Field(min_length=1, max_length=128)]
    decision: DecisionValue
    reason_code: Annotated[str, Field(min_length=1, max_length=64)]
    notes: Annotated[str, Field(max_length=4000)] | None = None
    fused_score_at_decision: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    model_version_at_decision: Annotated[str, Field(max_length=128)] | None = None

    @field_validator("entity_type", "decision", mode="before")
    @classmethod
    def _lower(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value


class DecisionResponse(BaseModel):
    id: int
    entity_type: str
    entity_id: str
    analyst: str
    decision: str
    reason_code: str
    notes: str | None
    fused_score_at_decision: float | None
    model_version_at_decision: str | None
    created_utc: str


class DecisionListResponse(BaseModel):
    count: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)
    decisions: list[DecisionResponse]


class ErrorResponse(BaseModel):
    """The only error shape this service emits. Never carries a traceback."""

    error: str
    detail: str
    status_code: int
    request_id: str
