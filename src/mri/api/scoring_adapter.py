"""The service's only entry point to the scoring definition.

`mri.scoring` is the project's single source of truth. Nothing here re-declares
a weight, a band cut-off or a calibration rule; this module only adapts that
contract to what an HTTP handler needs:

* `load_scoring()` - the weights, bands and `band_for` the service reports with.
* `load_reference()` - the frozen `ReferenceCalibration` persisted at
  `mri.scoring.REFERENCE_PATH`. Loaded once at startup, never per request.
* `score_one()` - a thin pass-through to `mri.scoring.score_entity`, which is
  what makes `POST /score` possible without a population.

The reference artefact is written by the pipeline, not by this service. If it
is missing the service stays live but reports itself not-ready and `/score`
answers 503, rather than quietly scoring against something it invented. The one
exception is an explicit operator opt-in (`MRI_API_FIT_REFERENCE=1`), used by
the tests and the self-contained container demo; see `ensure_reference`.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping

log = logging.getLogger("mri.api.scoring")

# Raw signal -> candidate columns in the batch output, most correct first.
#
# These are the values `mri.scoring.score_entity` expects, i.e. *pre*
# calibration. Two of them are easy to get wrong:
#
#   ml       `final_risk_score` is the supervised probability, because
#            ML_BLEND now puts zero weight on anomaly. `supervised_score` is
#            the same number and is accepted as a synonym.
#   anomaly  must be `anomaly_score_raw` (-IsolationForest.score_samples).
#            `anomaly_score` is *already* calibrated against the frozen
#            reference, so feeding it in would transform it a second time.
#            It is deliberately not a fallback: a missing raw column reads as
#            "not observed" and lowers coverage, which is visible, rather than
#            producing a quietly wrong score.
RAW_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "ml": ("final_risk_score", "supervised_score"),
    "graph": ("graph_risk_score",),
    "temporal": ("temporal_risk_score",),
    "anomaly": ("anomaly_score_raw",),
}

# The calibrated 0-1 columns the batch fusion actually fuses.
NORM_COLUMNS: Final[dict[str, str]] = {
    "ml": "ml_score_norm",
    "graph": "graph_score_norm",
    "temporal": "temporal_score_norm",
    "anomaly": "anomaly_score_norm",
}

SIGNAL_LABELS: Final[dict[str, str]] = {
    "ml": "ML risk",
    "graph": "Graph",
    "temporal": "Temporal",
    "anomaly": "Anomaly",
}

ENV_REFERENCE_PATH: Final[str] = "MRI_SCORE_REFERENCE"
ENV_FIT_REFERENCE: Final[str] = "MRI_API_FIT_REFERENCE"


class ScoringUnavailable(RuntimeError):
    """Raised when the scoring contract or its frozen reference cannot be loaded."""


@dataclass(frozen=True, slots=True)
class ScoringDefinition:
    """What the service needs from `mri.scoring`, resolved once at startup."""

    source: str
    weights: dict[str, float]
    signal_order: tuple[str, ...]
    bands: list[str]
    band_thresholds: dict[str, float]
    alert_threshold: float
    ml_blend: dict[str, float]
    _band_for: Any
    _score_entity: Any

    def band_for(self, score: float | None) -> str:
        return str(self._band_for(score))

    def score_entity(self, raw_signals: Mapping[str, float | None], reference: Any) -> dict:
        return dict(self._score_entity(dict(raw_signals), reference=reference))


def load_scoring() -> ScoringDefinition:
    """Resolve the scoring contract. Raises if a required symbol is absent."""
    try:
        import mri.scoring as contract
    except ImportError as exc:  # pragma: no cover - the contract is a hard dependency
        raise ScoringUnavailable(
            "mri.scoring is not importable; the API has no scoring definition to serve"
        ) from exc

    required = (
        "WEIGHTS",
        "BAND_THRESHOLDS",
        "BANDS",
        "band_for",
        "fuse_signals",
        "score_entity",
        "load_reference",
        "REFERENCE_PATH",
    )
    missing = [name for name in required if not hasattr(contract, name)]
    if missing:
        raise ScoringUnavailable(f"mri.scoring is missing contract symbols: {missing}")

    return ScoringDefinition(
        source="mri.scoring",
        weights={str(k): float(v) for k, v in dict(contract.WEIGHTS).items()},
        signal_order=tuple(getattr(contract, "SIGNAL_ORDER", tuple(contract.WEIGHTS))),
        bands=list(contract.BANDS),
        band_thresholds={str(k): float(v) for k, v in dict(contract.BAND_THRESHOLDS).items()},
        alert_threshold=float(
            getattr(contract, "ALERT_THRESHOLD", contract.BAND_THRESHOLDS["Critical"])
        ),
        ml_blend={str(k): float(v) for k, v in dict(getattr(contract, "ML_BLEND", {})).items()},
        _band_for=contract.band_for,
        _score_entity=contract.score_entity,
    )


def reference_path() -> Path:
    """Where the frozen reference lives. Overridable for tests and containers."""
    override = os.getenv(ENV_REFERENCE_PATH)
    if override:
        return Path(override)
    import mri.scoring as contract

    return Path(contract.REFERENCE_PATH)


def raw_signals_from_row(row: Mapping[str, Any]) -> dict[str, float | None]:
    """Pull the four raw signals out of one batch-output row.

    A column that is absent or non-numeric yields ``None`` - "not observed" -
    rather than 0.0, which is the fail-open `mri.scoring` exists to remove.
    """
    signals: dict[str, float | None] = {}
    for name, candidates in RAW_COLUMNS.items():
        value: float | None = None
        for column in candidates:
            if column in row:
                try:
                    candidate = float(row[column])
                except (TypeError, ValueError):
                    continue
                if candidate == candidate:  # not NaN
                    value = candidate
                    break
        signals[name] = value
    return signals


def load_reference(path: Path | None = None) -> Any:
    """Load the frozen `ReferenceCalibration`. Raises when it is not there."""
    import mri.scoring as contract

    target = path or reference_path()
    if not target.exists():
        raise ScoringUnavailable(
            f"frozen calibration reference not found at {target}. Run "
            "`python -m mri.pipeline --no-ui` to produce it, or set "
            f"{ENV_FIT_REFERENCE}=1 to let the service freeze one from the batch output."
        )
    try:
        return contract.load_reference(target)
    except Exception as exc:  # noqa: BLE001 - a bad artefact is an operational fault
        raise ScoringUnavailable(f"calibration reference at {target} is unusable: {exc}") from exc


def ensure_reference(path: Path | None = None, fused_path: Path | None = None) -> Any:
    """Load the reference; freeze one from the batch output only if told to.

    Fitting the reference is the pipeline's job. This opt-in exists so the
    container demo and the test suite are self-contained before the fusion
    stage has frozen one, and it is loud about having done it. It never
    overwrites an existing reference.
    """
    target = path or reference_path()
    try:
        return load_reference(target)
    except ScoringUnavailable:
        if os.getenv(ENV_FIT_REFERENCE, "").strip().lower() not in ("1", "true", "yes", "on"):
            raise

    import pandas as pd

    import mri.scoring as contract
    from mri.config import OUTPUT_DIR

    source = fused_path or Path(OUTPUT_DIR) / "final_fused_risk_v4.csv"
    if not source.exists():
        raise ScoringUnavailable(
            f"cannot freeze a calibration reference: {source} is missing. "
            "Run the pipeline first."
        )

    frame = pd.read_csv(source)
    calibration = contract.ReferenceCalibration()
    for name, candidates in RAW_COLUMNS.items():
        column = next((c for c in candidates if c in frame.columns), None)
        if column is None:
            raise ScoringUnavailable(
                f"batch output {source} has no column for the '{name}' signal "
                f"(looked for {list(candidates)})"
            )
        values = pd.to_numeric(frame[column], errors="coerce").dropna().tolist()
        calibration.ensure_signal(name, values, source=f"{source.name}:{column}")

    calibration.save(target)
    log.warning(
        "froze a calibration reference from the batch output because "
        "%s=1; the pipeline should own this artefact",
        ENV_FIT_REFERENCE,
        extra={"reference_path": str(target), "source": str(source), "rows": int(len(frame))},
    )
    return calibration


__all__ = [
    "ENV_FIT_REFERENCE",
    "ENV_REFERENCE_PATH",
    "NORM_COLUMNS",
    "RAW_COLUMNS",
    "SIGNAL_LABELS",
    "ScoringDefinition",
    "ScoringUnavailable",
    "ensure_reference",
    "load_reference",
    "load_scoring",
    "raw_signals_from_row",
    "reference_path",
]
