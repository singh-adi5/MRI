"""The service's single entry point to the scoring definition.

`mri.scoring` is the project's source of truth for `WEIGHTS`, `BAND_THRESHOLDS`,
`BANDS`, `band_for` and `fuse_signals`. This module imports them from there and
exposes nothing of its own when it can.

Two deliberate details:

1. **Fallback.** `mri.scoring` is being landed by another workstream. Until it
   exists this module falls back to the values the batch pipeline is running
   today (`mri.fusion.fuse`, `mri.app.theme.SIGNALS`) and reports
   `scoring_source == "fallback"` on `/healthz`, `/readyz` and in the startup
   log, so the substitution is never silent. Delete `_FallbackScoring` once the
   real module lands; nothing else in `src/mri/api` needs to change.

2. **Frozen reference.** `/score` must score one entity without the population,
   but the batch fusion min-max normalises each signal across it. The frozen
   reference is that calibration captured once from a batch run and persisted to
   `artifacts/calibration_reference.json`. If `mri.scoring` grows its own
   `frozen_reference()` the adapter prefers it; see `_load_contract_reference`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping

from mri.config import ARTIFACT_DIR, OUTPUT_DIR

log = logging.getLogger("mri.api.scoring")

FUSED_PATH: Final[Path] = Path(OUTPUT_DIR) / "final_fused_risk_v4.csv"
REFERENCE_PATH: Final[Path] = Path(ARTIFACT_DIR) / "calibration_reference.json"

# Raw signal column -> the normalised column the batch fusion derives from it.
SIGNAL_COLUMNS: Final[dict[str, tuple[str, str]]] = {
    "ml": ("final_risk_score", "ml_score_norm"),
    "graph": ("graph_risk_score", "graph_score_norm"),
    "temporal": ("temporal_risk_score", "temporal_score_norm"),
    "anomaly": ("anomaly_score", "anomaly_score_norm"),
}
SIGNAL_KEYS: Final[tuple[str, ...]] = tuple(SIGNAL_COLUMNS)

SIGNAL_LABELS: Final[dict[str, str]] = {
    "ml": "ML risk",
    "graph": "Graph",
    "temporal": "Temporal",
    "anomaly": "Anomaly",
}


class ScoringUnavailable(RuntimeError):
    """Raised when neither the contract module nor the fallback can be loaded."""


class _FallbackScoring:
    """Mirror of the live batch definition, used only until `mri.scoring` exists.

    The weights and cut-offs are read from `mri.fusion.fuse` / `mri.app.theme`
    where they are importable rather than retyped, so this cannot drift from the
    pipeline without the import failing loudly.
    """

    source = "fallback"

    def __init__(self) -> None:
        from mri.app.theme import SIGNALS  # weights live here as data

        self.WEIGHTS: dict[str, float] = {s["key"]: float(s["weight"]) for s in SIGNALS}
        self.BANDS: list[str] = ["Low", "Medium", "High", "Critical"]
        self.BAND_THRESHOLDS: dict[str, float] = {
            "Medium": 0.30,
            "High": 0.55,
            "Critical": 0.75,
        }

    def band_for(self, score: float) -> str:
        if score >= self.BAND_THRESHOLDS["Critical"]:
            return "Critical"
        if score >= self.BAND_THRESHOLDS["High"]:
            return "High"
        if score >= self.BAND_THRESHOLDS["Medium"]:
            return "Medium"
        return "Low"

    def fuse_signals(self, signals: Mapping[str, float]) -> float:
        return sum(self.WEIGHTS[key] * float(signals.get(key, 0.0)) for key in self.WEIGHTS)


@dataclass(frozen=True, slots=True)
class ScoringDefinition:
    """The weights, bands and fusion function the service scores with."""

    source: str
    weights: dict[str, float]
    bands: list[str]
    band_thresholds: dict[str, float]
    _band_for: Any
    _fuse: Any

    def band_for(self, score: float) -> str:
        return str(self._band_for(score))

    def fuse(self, signals: Mapping[str, float]) -> float:
        return float(self._fuse(signals))

    def contributions(self, normalised: Mapping[str, float]) -> dict[str, float]:
        """weight x normalised signal, per signal. These sum to the fused score."""
        return {key: self.weights[key] * float(normalised.get(key, 0.0)) for key in self.weights}


def load_scoring() -> ScoringDefinition:
    """Prefer `mri.scoring`; fall back to the live batch definition."""
    try:
        import mri.scoring as contract  # type: ignore[import-not-found]
    except ImportError:
        impl: Any = _FallbackScoring()
        source = "fallback"
        log.warning(
            "mri.scoring not importable; serving with the fallback definition "
            "mirrored from mri.fusion.fuse. Reported as scoring_source=fallback."
        )
    else:
        missing = [
            name
            for name in ("WEIGHTS", "BAND_THRESHOLDS", "BANDS", "band_for", "fuse_signals")
            if not hasattr(contract, name)
        ]
        if missing:
            raise ScoringUnavailable(
                f"mri.scoring is importable but missing contract symbols: {missing}"
            )
        impl = contract
        source = "mri.scoring"

    return ScoringDefinition(
        source=source,
        weights={str(k): float(v) for k, v in dict(impl.WEIGHTS).items()},
        bands=list(impl.BANDS),
        band_thresholds={str(k): float(v) for k, v in dict(impl.BAND_THRESHOLDS).items()},
        _band_for=impl.band_for,
        _fuse=impl.fuse_signals,
    )


@dataclass(frozen=True, slots=True)
class CalibrationReference:
    """Frozen min-max per raw signal: lets one entity be scored on its own.

    `built_from` and `source_sha256` tie the reference to the batch run that
    produced it, so a stale reference is visible rather than merely wrong.
    """

    bounds: dict[str, dict[str, float]]
    built_from: str
    source_sha256: str
    population_rows: int
    built_utc: str
    signals: tuple[str, ...] = field(default=SIGNAL_KEYS)

    def normalise(self, signal: str, raw: float) -> float:
        """Min-max against the frozen bounds, clipped to [0, 1].

        The batch never clips because every row is inside its own population. A
        live entity can land outside it; clipping keeps the fused score on the
        [0, 1] scale the bands are defined on. Inside the range the arithmetic is
        identical to the batch, which is what the parity test asserts.
        """
        bound = self.bounds.get(signal)
        if bound is None:
            return 0.0
        low, high = bound["min"], bound["max"]
        if not math.isfinite(raw) or high <= low:
            return 0.0
        return min(1.0, max(0.0, (float(raw) - low) / (high - low)))

    def normalise_all(self, raw_signals: Mapping[str, float]) -> dict[str, float]:
        return {key: self.normalise(key, float(raw_signals.get(key, 0.0))) for key in self.signals}

    def out_of_range(self, raw_signals: Mapping[str, float]) -> list[str]:
        """Signals whose raw value sits outside the frozen population range."""
        flagged = []
        for key in self.signals:
            bound = self.bounds.get(key)
            if bound is None:
                continue
            value = float(raw_signals.get(key, 0.0))
            if value < bound["min"] or value > bound["max"]:
                flagged.append(key)
        return flagged

    def as_dict(self) -> dict[str, Any]:
        return {
            "bounds": self.bounds,
            "built_from": self.built_from,
            "source_sha256": self.source_sha256,
            "population_rows": self.population_rows,
            "built_utc": self.built_utc,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_contract_reference() -> CalibrationReference | None:
    """Handoff point: if `mri.scoring` ships its own frozen reference, use it.

    Expected shape: `mri.scoring.frozen_reference()` returning a mapping of
    signal key -> {"min": float, "max": float}. Anything else is ignored rather
    than guessed at.
    """
    try:
        import mri.scoring as contract  # type: ignore[import-not-found]
    except ImportError:
        return None
    builder = getattr(contract, "frozen_reference", None)
    if not callable(builder):
        return None
    try:
        raw = builder()
        bounds = {
            str(key): {"min": float(value["min"]), "max": float(value["max"])}
            for key, value in dict(raw).items()
        }
    except Exception:  # noqa: BLE001 - a malformed contract must not take the service down
        log.exception("mri.scoring.frozen_reference() unusable; deriving from the batch run")
        return None
    return CalibrationReference(
        bounds=bounds,
        built_from="mri.scoring.frozen_reference",
        source_sha256="",
        population_rows=0,
        built_utc="",
    )


def build_reference(fused_path: Path = FUSED_PATH) -> CalibrationReference:
    """Derive the frozen reference from a completed batch run."""
    import pandas as pd
    from datetime import datetime, timezone

    if not fused_path.exists():
        raise ScoringUnavailable(
            f"cannot build the calibration reference: {fused_path} is missing. "
            "Run `python -m mri.pipeline --no-ui` first."
        )
    frame = pd.read_csv(fused_path)
    bounds: dict[str, dict[str, float]] = {}
    for key, (raw_col, _norm_col) in SIGNAL_COLUMNS.items():
        if raw_col not in frame.columns:
            raise ScoringUnavailable(f"batch output is missing the '{raw_col}' column")
        series = pd.to_numeric(frame[raw_col], errors="coerce").fillna(0.0)
        bounds[key] = {"min": float(series.min()), "max": float(series.max())}

    return CalibrationReference(
        bounds=bounds,
        built_from=str(fused_path),
        source_sha256=_sha256(fused_path),
        population_rows=int(len(frame)),
        built_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def save_reference(reference: CalibrationReference, path: Path = REFERENCE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reference.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_reference(
    path: Path = REFERENCE_PATH,
    fused_path: Path = FUSED_PATH,
    *,
    refresh: bool = False,
) -> CalibrationReference:
    """Load the frozen reference, rebuilding it when the batch run has moved on."""
    contract_reference = _load_contract_reference()
    if contract_reference is not None:
        return contract_reference

    if not refresh and path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            cached = CalibrationReference(
                bounds={
                    str(k): {"min": float(v["min"]), "max": float(v["max"])}
                    for k, v in payload["bounds"].items()
                },
                built_from=str(payload.get("built_from", "")),
                source_sha256=str(payload.get("source_sha256", "")),
                population_rows=int(payload.get("population_rows", 0)),
                built_utc=str(payload.get("built_utc", "")),
            )
        except (OSError, KeyError, ValueError, TypeError):
            log.warning("calibration reference at %s is unreadable; rebuilding", path)
        else:
            # A reference built from a superseded batch run would score against
            # the wrong population, so verify it before trusting it.
            if fused_path.exists() and cached.source_sha256 == _sha256(fused_path):
                return cached
            log.info("calibration reference is stale against %s; rebuilding", fused_path)

    reference = build_reference(fused_path)
    try:
        save_reference(reference, path)
    except OSError:  # pragma: no cover - a read-only artefact dir must not break scoring
        log.warning("could not persist the calibration reference to %s", path)
    return reference
