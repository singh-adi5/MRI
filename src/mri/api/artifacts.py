"""Loads the batch artefacts the service serves, and reports what is missing.

Everything here is read-only and loaded once at startup into memory. The
population is a few thousand rows, so a dict index beats a query per request
and it keeps the API free of any write path into the pipeline's own stores.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from mri.api.scoring_adapter import (
    NORM_COLUMNS,
    RAW_COLUMNS,
    ScoringDefinition,
    ScoringUnavailable,
    ensure_reference,
    load_scoring,
    raw_signals_from_row,
    reference_path,
)
from mri.config import OUTPUT_DIR

log = logging.getLogger("mri.api.artifacts")

FUSED_PATH: Final[Path] = Path(OUTPUT_DIR) / "final_fused_risk_v4.csv"
EVALUATION_PATH: Final[Path] = Path(OUTPUT_DIR) / "evaluation.json"
LIFT_CURVE_PATH: Final[Path] = Path(OUTPUT_DIR) / "lift_curve.csv"
HEALTH_PATH: Final[Path] = Path(OUTPUT_DIR) / "v6_pipeline_health.json"

MODEL_VERSION: Final[str] = "mri-fusion-v4.1"

# Investigator-facing evidence carried through from the detection stages.
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
    "signal_coverage",
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
    coverage: float
    signals_missing: list[str]
    raw_signals: dict[str, float | None]
    calibrated_signals: dict[str, float]
    contributions: dict[str, float]
    reason_codes: list[str]
    explanation: str
    evidence: dict[str, float]
    rank: int


class ArtifactStore:
    """Holds the batch artefacts. ``is_ready`` is what ``/readyz`` answers with."""

    def __init__(self) -> None:
        self.scoring: ScoringDefinition | None = None
        self.reference: Any | None = None
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
            self.reference = ensure_reference()
        except ScoringUnavailable as exc:
            self.errors.append(f"calibration reference: {exc}")
            self.reference = None

        self._load_entities()
        self.evaluation = _read_json(EVALUATION_PATH)
        self.health = _read_json(HEALTH_PATH)
        self.loaded = True

        if self.errors:
            log.warning("artefacts loaded with gaps", extra={"gaps": self.errors})
        else:
            log.info(
                "artefacts loaded",
                extra={
                    "entities": len(self.entities),
                    "reference_path": str(reference_path()),
                    "evaluation_present": self.evaluation is not None,
                },
            )
        return self

    def _load_entities(self) -> None:
        if self.scoring is None:
            return
        if not FUSED_PATH.exists():
            self.errors.append(f"batch scores missing: {FUSED_PATH}")
            return

        import pandas as pd

        frame = pd.read_csv(FUSED_PATH, dtype={"entity_id": str})
        if "fused_risk_score" not in frame.columns:
            self.errors.append(f"{FUSED_PATH} has no fused_risk_score column")
            return
        frame = frame.sort_values("fused_risk_score", ascending=False).reset_index(drop=True)

        entities: dict[str, ScoredEntity] = {}
        band_counts: dict[str, int] = {band: 0 for band in self.scoring.bands}

        for rank, row in enumerate(frame.to_dict(orient="records"), start=1):
            calibrated = {
                key: _as_float(row.get(column)) for key, column in NORM_COLUMNS.items()
            }
            # The batch writes c_<signal>; recompute only if it did not.
            contributions = {
                key: _as_float(row.get(f"c_{key}")) for key in RAW_COLUMNS
            }
            if not any(contributions.values()):
                coverage_weight = sum(
                    self.scoring.weights[k] for k, v in calibrated.items() if v == v
                )
                if coverage_weight > 0:
                    contributions = {
                        key: (self.scoring.weights[key] / coverage_weight) * value
                        for key, value in calibrated.items()
                    }

            band = str(row.get("fused_risk_band") or "Unknown")
            entity = ScoredEntity(
                entity_type=str(row.get("entity_type", "")).lower(),
                entity_id=str(row.get("entity_id", "")),
                fused_score=_as_float(row.get("fused_risk_score")),
                band=band,
                coverage=_as_float(row.get("signal_coverage"), default=1.0),
                signals_missing=_split(row.get("signals_missing"), ","),
                raw_signals=raw_signals_from_row(row),
                calibrated_signals=calibrated,
                contributions=contributions,
                reason_codes=_split(row.get("top_reason_codes"), "|"),
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

    # -- readiness ---------------------------------------------------------
    def is_ready(self) -> bool:
        return bool(self.loaded and self.scoring and self.reference is not None and self.entities)

    def readiness_detail(self) -> dict[str, Any]:
        return {
            "scoring_definition": self.scoring is not None,
            "scoring_source": self.scoring.source if self.scoring else None,
            "calibration_reference": self.reference is not None,
            "calibration_reference_path": str(reference_path()),
            "batch_scores": bool(self.entities),
            "entities_loaded": len(self.entities),
            "evaluation_present": self.evaluation is not None,
            "lift_curve_present": LIFT_CURVE_PATH.exists(),
            "errors": list(self.errors),
        }

    # -- lookups -----------------------------------------------------------
    def get(self, entity_type: str, entity_id: str) -> ScoredEntity | None:
        return self.entities.get(entity_key(entity_type, entity_id))

    def alert_bands(self) -> tuple[str, ...]:
        """Bands at or above the shipped alert threshold, from mri.scoring."""
        if self.scoring is None:
            return ("Critical",)
        cut = self.scoring.alert_threshold
        return tuple(
            band
            for band in self.scoring.bands
            if self.scoring.band_thresholds.get(band, 0.0) >= cut
        ) or ("Critical",)

    def alert_rate(self) -> float:
        total = len(self.entities)
        if not total:
            return 0.0
        alerts = sum(self.band_counts.get(band, 0) for band in self.alert_bands())
        return round(alerts / total, 6)

    def top_decile_threshold(self) -> float | None:
        """Fused score at the 90th percentile of the scored population."""
        scores = sorted(
            (e.fused_score for e in self.entities.values() if e.fused_score == e.fused_score),
            reverse=True,
        )
        if not scores:
            return None
        cut = max(0, int(len(scores) * 0.10) - 1)
        return float(scores[cut])


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if number != number else number  # NaN -> default


def _split(value: Any, separator: str) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [part.strip() for part in value.split(separator) if part.strip()]


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
    "ArtifactStore",
    "ScoredEntity",
    "entity_key",
]
