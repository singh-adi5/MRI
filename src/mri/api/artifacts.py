"""Loads the batch artefacts the service serves, and reports what is missing.

Everything here is read-only and loaded once at startup into memory. The dataset
is a few thousand rows, so an index in a dict beats a query per request, and it
keeps the API free of any write path to the pipeline's own stores.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from mri.api.scoring_adapter import (
    SIGNAL_COLUMNS,
    SIGNAL_KEYS,
    CalibrationReference,
    ScoringDefinition,
    ScoringUnavailable,
    load_reference,
    load_scoring,
)
from mri.config import OUTPUT_DIR

log = logging.getLogger("mri.api.artifacts")

FUSED_PATH: Final[Path] = Path(OUTPUT_DIR) / "final_fused_risk_v4.csv"
EVALUATION_PATH: Final[Path] = Path(OUTPUT_DIR) / "evaluation.json"
LIFT_CURVE_PATH: Final[Path] = Path(OUTPUT_DIR) / "lift_curve.csv"
HEALTH_PATH: Final[Path] = Path(OUTPUT_DIR) / "v6_pipeline_health.json"

MODEL_VERSION: Final[str] = "mri-fusion-v4.1"

_EVIDENCE_COLUMNS: Final[tuple[str, ...]] = (
    "graph_degree",
    "degree_centrality",
    "component_size",
    "shared_device_links",
    "shared_region_links",
    "transaction_links",
    "order_velocity_per_hour",
    "order_burst_ratio",
    "refund_velocity_per_hour",
    "login_velocity_per_hour",
    "failed_attempt_density",
    "seller_order_velocity_per_hour",
    "seller_order_burst_ratio",
)


def entity_key(entity_type: str, entity_id: str) -> str:
    return f"{str(entity_type).strip().lower()}:{str(entity_id).strip()}"


@dataclass(frozen=True, slots=True)
class ScoredEntity:
    """One entity exactly as the batch run scored it."""

    entity_type: str
    entity_id: str
    fused_score: float
    band: str
    raw_signals: dict[str, float]
    normalised_signals: dict[str, float]
    contributions: dict[str, float]
    reason_codes: list[str]
    explanation: str
    evidence: dict[str, float]
    rank: int


class ArtifactStore:
    """Holds the batch artefacts. ``is_ready`` is what ``/readyz`` answers with."""

    def __init__(self) -> None:
        self.scoring: ScoringDefinition | None = None
        self.reference: CalibrationReference | None = None
        self.entities: dict[str, ScoredEntity] = {}
        self.evaluation: dict[str, Any] | None = None
        self.health: dict[str, Any] | None = None
        self.band_counts: dict[str, int] = {}
        self.errors: list[str] = []
        self.loaded = False

    # -- loading -----------------------------------------------------------
    def load(self) -> "ArtifactStore":
        self.errors = []
        try:
            self.scoring = load_scoring()
        except ScoringUnavailable as exc:
            self.errors.append(f"scoring definition: {exc}")
            self.scoring = None

        try:
            self.reference = load_reference()
        except ScoringUnavailable as exc:
            self.errors.append(f"calibration reference: {exc}")
            self.reference = None

        self._load_entities()
        self._load_evaluation()
        self._load_health()
        self.loaded = True

        if self.errors:
            log.warning("artefact load finished with gaps: %s", "; ".join(self.errors))
        else:
            log.info(
                "artefacts loaded: %d entities, scoring_source=%s",
                len(self.entities),
                self.scoring.source if self.scoring else "none",
            )
        return self

    def _load_entities(self) -> None:
        if not FUSED_PATH.exists():
            self.errors.append(f"batch scores missing: {FUSED_PATH}")
            return
        if self.scoring is None:
            return

        import pandas as pd

        frame = pd.read_csv(FUSED_PATH, dtype={"entity_id": str})
        frame = frame.sort_values("fused_risk_score", ascending=False).reset_index(drop=True)

        entities: dict[str, ScoredEntity] = {}
        band_counts: dict[str, int] = {band: 0 for band in self.scoring.bands}

        for rank, row in enumerate(frame.to_dict(orient="records"), start=1):
            raw = {
                key: _as_float(row.get(raw_col)) for key, (raw_col, _) in SIGNAL_COLUMNS.items()
            }
            normalised = {
                key: _as_float(row.get(norm_col)) for key, (_, norm_col) in SIGNAL_COLUMNS.items()
            }
            band = str(row.get("fused_risk_band", "Low"))
            entity = ScoredEntity(
                entity_type=str(row.get("entity_type", "")).lower(),
                entity_id=str(row.get("entity_id", "")),
                fused_score=_as_float(row.get("fused_risk_score")),
                band=band,
                raw_signals=raw,
                normalised_signals=normalised,
                contributions=self.scoring.contributions(normalised),
                reason_codes=_split_codes(row.get("top_reason_codes")),
                explanation=str(row.get("analyst_summary") or row.get("fusion_summary") or ""),
                evidence={
                    col: _as_float(row.get(col)) for col in _EVIDENCE_COLUMNS if col in row
                },
                rank=rank,
            )
            entities[entity_key(entity.entity_type, entity.entity_id)] = entity
            band_counts[band] = band_counts.get(band, 0) + 1

        self.entities = entities
        self.band_counts = band_counts

    def _load_evaluation(self) -> None:
        # Written by mri.evaluation, which is landing alongside this service.
        self.evaluation = _read_json(EVALUATION_PATH)
        if self.evaluation is None:
            log.info("%s not present yet; /metrics will report evaluation as null", EVALUATION_PATH)

    def _load_health(self) -> None:
        self.health = _read_json(HEALTH_PATH)

    # -- readiness ---------------------------------------------------------
    def is_ready(self) -> bool:
        return bool(self.loaded and self.scoring and self.reference and self.entities)

    def readiness_detail(self) -> dict[str, Any]:
        return {
            "scoring_definition": self.scoring is not None,
            "scoring_source": self.scoring.source if self.scoring else None,
            "calibration_reference": self.reference is not None,
            "batch_scores": bool(self.entities),
            "entities_loaded": len(self.entities),
            "evaluation_present": self.evaluation is not None,
            "lift_curve_present": LIFT_CURVE_PATH.exists(),
            "errors": list(self.errors),
        }

    # -- lookups -----------------------------------------------------------
    def get(self, entity_type: str, entity_id: str) -> ScoredEntity | None:
        return self.entities.get(entity_key(entity_type, entity_id))

    def alert_rate(self, alert_bands: tuple[str, ...] = ("High", "Critical")) -> float:
        total = len(self.entities)
        if not total:
            return 0.0
        alerts = sum(self.band_counts.get(band, 0) for band in alert_bands)
        return round(alerts / total, 6)

    def top_decile_threshold(self) -> float | None:
        """Fused score at the 90th percentile of the scored population."""
        if not self.entities:
            return None
        scores = sorted((e.fused_score for e in self.entities.values()), reverse=True)
        cut = max(0, int(len(scores) * 0.10) - 1)
        return float(scores[cut])


def _as_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if number != number else number  # NaN -> 0.0


def _split_codes(value: Any) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [part.strip() for part in value.split("|") if part.strip()]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning("could not parse %s", path)
        return None
    return payload if isinstance(payload, dict) else {"value": payload}


__all__ = [
    "EVALUATION_PATH",
    "FUSED_PATH",
    "LIFT_CURVE_PATH",
    "MODEL_VERSION",
    "SIGNAL_KEYS",
    "ArtifactStore",
    "ScoredEntity",
    "entity_key",
]
