"""Single source of truth for every number that turns signals into a risk score.

THIS IS THE ONLY PLACE THE FUSION WEIGHTS AND THE BAND CUT-OFFS LIVE.
Do not re-declare them in a stage module, a dashboard, a CSS file or a test.
Import them: ``from mri.scoring import WEIGHTS, BAND_THRESHOLDS, band_for, fuse_signals``.

What this module fixes, and the invariants it holds
---------------------------------------------------
1. Published weights are the true weights. ``WEIGHTS`` sums to 1.0 and each
   signal contributes exactly its stated weight. The anomaly signal appears
   once, in the fusion, and is no longer also folded into the ML signal
   (``ML_BLEND``), so its effective weight is 0.15, not 0.27.

2. A score is a score, not a rank within whatever batch happened to run.
   Raw signals are mapped to [0, 1] against a *frozen reference distribution*
   (``ReferenceCalibration``) persisted as a model artefact at
   ``models/score_reference.json``. Scoring one entity produces the same number
   whether it is scored alone or inside a population of a million, and the band
   cut-offs therefore mean the same thing from run to run.

3. A missing signal is missing, not zero. ``fuse_signals`` treats NaN/None as
   "not observed", renormalises the weights over the signals that *are*
   present, and ``fuse_with_coverage`` reports what fraction of the weight mass
   was actually observed. Scoring an entity that no detector saw as 0.0 ("low
   risk") is a fail-open; this fails loud instead.

Single-entity entry point for the API: ``score_entity``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from mri.config import MODEL_DIR

__all__ = [
    "WEIGHTS",
    "SIGNAL_ORDER",
    "BANDS",
    "BAND_THRESHOLDS",
    "ML_BLEND",
    "CALIBRATION_METHODS",
    "REFERENCE_PATH",
    "band_for",
    "fuse_signals",
    "fuse_with_coverage",
    "score_entity",
    "SignalReference",
    "ReferenceCalibration",
    "load_reference",
]

# --------------------------------------------------------------------------
# The numbers.
# --------------------------------------------------------------------------

#: Fusion weights. Must sum to 1.0. These are the *published* weights and,
#: since the anomaly double-count was removed, also the effective ones.
WEIGHTS: dict[str, float] = {
    "ml": 0.40,
    "graph": 0.25,
    "temporal": 0.20,
    "anomaly": 0.15,
}

#: Stable ordering for anything that iterates signals (columns, reports, UI).
SIGNAL_ORDER: tuple[str, ...] = ("ml", "graph", "temporal", "anomaly")

#: Composition of the ML signal itself. The anomaly model used to be blended in
#: here at 0.30 *as well as* carrying 0.15 of the fusion, giving it an effective
#: weight of 0.40 * 0.30 + 0.15 = 0.27 against a published 0.15. The anomaly
#: signal now enters the score in exactly one place: the fusion.
ML_BLEND: dict[str, float] = {"supervised": 1.00, "anomaly": 0.00}

#: Band severities, ascending.
BANDS: list[str] = ["Low", "Medium", "High", "Critical"]

#: Lower bound of each band on the fused 0-1 scale. "Low" is everything below
#: the "Medium" cut.
BAND_THRESHOLDS: dict[str, float] = {"Medium": 0.30, "High": 0.55, "Critical": 0.75}

#: The cut at which an entity becomes an alert in the shipped configuration.
ALERT_THRESHOLD: float = BAND_THRESHOLDS["Critical"]

#: How each signal is mapped onto the 0-1 fusion scale.
#:   "identity" - already a bounded score (a probability); clip only.
#:   "quantile" - unbounded or arbitrarily scaled; map through the frozen ECDF.
CALIBRATION_METHODS: dict[str, str] = {
    "ml": "identity",        # supervised probability, already in [0, 1]
    "graph": "quantile",
    "temporal": "quantile",
    "anomaly": "quantile",   # -IsolationForest.score_samples, unbounded
}

#: Where the frozen reference distribution is persisted.
REFERENCE_PATH: Path = Path(MODEL_DIR) / "score_reference.json"

_WEIGHT_SUM_TOLERANCE = 1e-9
if abs(sum(WEIGHTS.values()) - 1.0) > _WEIGHT_SUM_TOLERANCE:  # pragma: no cover - guard
    raise ValueError(f"WEIGHTS must sum to 1.0, got {sum(WEIGHTS.values())}")


# --------------------------------------------------------------------------
# Bands.
# --------------------------------------------------------------------------

def band_for(score: float | None) -> str:
    """Map a fused 0-1 score onto a band name.

    A missing score is not "Low": it is ``Unknown``. Silently banding an
    unscored entity as Low is the same fail-open this module exists to remove.
    """
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "Unknown"
    score = float(score)
    if score >= BAND_THRESHOLDS["Critical"]:
        return "Critical"
    if score >= BAND_THRESHOLDS["High"]:
        return "High"
    if score >= BAND_THRESHOLDS["Medium"]:
        return "Medium"
    return "Low"


# --------------------------------------------------------------------------
# Fusion.
# --------------------------------------------------------------------------

def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(float(value)))
    except (TypeError, ValueError):
        return True


def fuse_with_coverage(**signals: float | None) -> dict:
    """Fuse calibrated signals, renormalising over the ones actually present.

    Each keyword must be a signal name from :data:`WEIGHTS` and a value already
    on the 0-1 calibrated scale (see :meth:`ReferenceCalibration.transform`).
    ``None`` or ``NaN`` means "this detector did not see the entity" and the
    signal is dropped from the weighted mean rather than counted as 0.0.

    Returns the fused score, the band, the fraction of total weight that was
    observed (``coverage``), and which signals were present or missing.
    """
    unknown = set(signals) - set(WEIGHTS)
    if unknown:
        raise ValueError(f"unknown signal(s): {sorted(unknown)}; expected {sorted(WEIGHTS)}")

    present: dict[str, float] = {}
    missing: list[str] = []
    for name in SIGNAL_ORDER:
        value = signals.get(name)
        if _is_missing(value):
            missing.append(name)
            continue
        present[name] = float(np.clip(float(value), 0.0, 1.0))

    observed_weight = sum(WEIGHTS[name] for name in present)
    if observed_weight <= 0.0:
        return {
            "fused_risk_score": float("nan"),
            "fused_risk_band": "Unknown",
            "coverage": 0.0,
            "signals_present": [],
            "signals_missing": list(missing),
            "contributions": {},
        }

    # Renormalise so the present signals keep their *relative* published
    # weights and the score stays on the same 0-1 scale as a full observation.
    contributions = {
        name: (WEIGHTS[name] / observed_weight) * value for name, value in present.items()
    }
    score = float(sum(contributions.values()))
    return {
        "fused_risk_score": score,
        "fused_risk_band": band_for(score),
        "coverage": float(observed_weight),
        "signals_present": list(present),
        "signals_missing": missing,
        "contributions": contributions,
    }


def fuse_signals(**signals: float | None) -> float:
    """Fused 0-1 risk score. NaN if no signal was observed. See fuse_with_coverage."""
    return fuse_with_coverage(**signals)["fused_risk_score"]


# --------------------------------------------------------------------------
# Frozen reference calibration.
# --------------------------------------------------------------------------

#: Percentile grid stored per signal. 201 knots keeps the tails resolved to
#: 0.5 percentile without making the artefact large.
_DEFAULT_PROBS = np.linspace(0.0, 1.0, 201)


@dataclass
class SignalReference:
    """The frozen distribution of one raw signal, as quantile knots."""

    name: str
    method: str
    n: int
    knots: list[float] = field(default_factory=list)
    probs: list[float] = field(default_factory=list)
    raw_min: float = 0.0
    raw_max: float = 1.0
    fitted_at: str = ""
    source: str = ""

    def transform(self, values) -> np.ndarray:
        """Map raw values onto [0, 1] against this frozen distribution."""
        arr = np.asarray(values, dtype=float)
        if self.method == "identity":
            return np.clip(arr, 0.0, 1.0)
        if not self.knots:
            raise ValueError(f"reference for {self.name!r} has no quantile knots")

        xp = np.asarray(self.knots, dtype=float)
        fp = np.asarray(self.probs, dtype=float)
        finite = np.isfinite(arr)
        out = np.full(arr.shape, np.nan, dtype=float)
        if not finite.any():
            return out

        # sklearn QuantileTransformer's tie handling: average the forward and
        # the reversed interpolation so a value sitting on a mass point (e.g.
        # the many entities with a graph score of exactly 0) lands in the
        # middle of that mass rather than at its top edge.
        x = arr[finite]
        lower = np.interp(x, xp, fp)
        upper = np.interp(-x, -xp[::-1], -fp[::-1])
        out[finite] = np.clip(0.5 * (lower - upper), 0.0, 1.0)
        return out

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "method": self.method,
            "n": self.n,
            "knots": self.knots,
            "probs": self.probs,
            "raw_min": self.raw_min,
            "raw_max": self.raw_max,
            "fitted_at": self.fitted_at,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "SignalReference":
        return cls(
            name=str(payload["name"]),
            method=str(payload["method"]),
            n=int(payload.get("n", 0)),
            knots=[float(v) for v in payload.get("knots", [])],
            probs=[float(v) for v in payload.get("probs", [])],
            raw_min=float(payload.get("raw_min", 0.0)),
            raw_max=float(payload.get("raw_max", 1.0)),
            fitted_at=str(payload.get("fitted_at", "")),
            source=str(payload.get("source", "")),
        )

    @classmethod
    def fit(
        cls,
        name: str,
        values: Sequence[float],
        method: str | None = None,
        source: str = "",
        probs: Iterable[float] = _DEFAULT_PROBS,
    ) -> "SignalReference":
        method = method or CALIBRATION_METHODS.get(name, "quantile")
        arr = np.asarray(list(values), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            raise ValueError(f"cannot fit reference for {name!r}: no finite values")
        probs_arr = np.asarray(list(probs), dtype=float)
        knots = np.quantile(arr, probs_arr) if method != "identity" else np.array([])
        return cls(
            name=name,
            method=method,
            n=int(arr.size),
            knots=[float(v) for v in knots],
            probs=[float(v) for v in probs_arr] if method != "identity" else [],
            raw_min=float(arr.min()),
            raw_max=float(arr.max()),
            fitted_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            source=source,
        )


class ReferenceCalibration:
    """A frozen per-signal reference distribution, persisted as plain JSON.

    JSON rather than a pickle on purpose: the artefact is auditable in review,
    diffable in git, and loading it cannot execute code.
    """

    FORMAT_VERSION = 1

    def __init__(self, signals: Mapping[str, SignalReference] | None = None, meta: Mapping | None = None):
        self.signals: dict[str, SignalReference] = dict(signals or {})
        self.meta: dict = dict(meta or {})

    # -- fitting -----------------------------------------------------------
    def fit_signal(self, name: str, values: Sequence[float], source: str = "") -> SignalReference:
        ref = SignalReference.fit(name, values, CALIBRATION_METHODS.get(name), source=source)
        self.signals[name] = ref
        return ref

    def ensure_signal(self, name: str, values: Sequence[float], source: str = "", refit: bool = False) -> SignalReference:
        """Freeze this signal's reference if it is not already frozen.

        Stages call this so the first run establishes the reference and every
        later run scores against it. Pass ``refit=True`` to deliberately rebase.
        """
        if refit or name not in self.signals:
            return self.fit_signal(name, values, source=source)
        return self.signals[name]

    # -- use ---------------------------------------------------------------
    def has(self, name: str) -> bool:
        return name in self.signals

    def transform(self, name: str, values) -> np.ndarray:
        if name not in self.signals:
            raise KeyError(
                f"no frozen reference for signal {name!r}. Run the training/fusion "
                f"stage once to create {REFERENCE_PATH}."
            )
        return self.signals[name].transform(values)

    def transform_one(self, name: str, value: float | None) -> float:
        if _is_missing(value):
            return float("nan")
        return float(self.transform(name, [value])[0])

    def staleness(self, name: str, current_n: int) -> float | None:
        """Ratio of current population size to the size the reference was frozen on."""
        ref = self.signals.get(name)
        if ref is None or not ref.n:
            return None
        return current_n / ref.n

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "format_version": self.FORMAT_VERSION,
            "weights": WEIGHTS,
            "band_thresholds": BAND_THRESHOLDS,
            "ml_blend": ML_BLEND,
            "meta": self.meta,
            "signals": {name: ref.to_dict() for name, ref in self.signals.items()},
        }

    def save(self, path: Path | str = REFERENCE_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str = REFERENCE_PATH) -> "ReferenceCalibration":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        signals = {
            name: SignalReference.from_dict(ref) for name, ref in payload.get("signals", {}).items()
        }
        return cls(signals=signals, meta=payload.get("meta", {}))

    @classmethod
    def load_or_empty(cls, path: Path | str = REFERENCE_PATH) -> "ReferenceCalibration":
        path = Path(path)
        if path.exists():
            try:
                return cls.load(path)
            except (json.JSONDecodeError, KeyError, ValueError):
                return cls()
        return cls()


_REFERENCE_CACHE: dict[str, ReferenceCalibration] = {}


def load_reference(path: Path | str = REFERENCE_PATH, cache: bool = True) -> ReferenceCalibration:
    """Load the frozen reference, memoised so an API process reads it once."""
    key = str(Path(path))
    if cache and key in _REFERENCE_CACHE:
        return _REFERENCE_CACHE[key]
    ref = ReferenceCalibration.load(path)
    if cache:
        _REFERENCE_CACHE[key] = ref
    return ref


# --------------------------------------------------------------------------
# Single-entity scoring: the API entry point.
# --------------------------------------------------------------------------

def score_entity(
    signals: Mapping[str, float | None],
    reference: ReferenceCalibration | None = None,
    reference_path: Path | str = REFERENCE_PATH,
) -> dict:
    """Score ONE entity from its raw signal values. No population required.

    ``signals`` maps signal names to *raw* values as the stages emit them:

        ml        supervised probability in [0, 1]  (mri.models.score)
        graph     raw graph_risk_score              (mri.detection.graph)
        temporal  raw temporal_risk_score           (mri.detection.temporal)
        anomaly   raw -IsolationForest.score_samples(mri.models.score)

    Each is mapped onto 0-1 against the frozen reference, then fused. A signal
    that is absent, ``None`` or NaN is reported as missing and excluded from
    the weighted mean; ``coverage`` says how much of the weight mass was real.

    >>> score_entity({"ml": 0.91, "graph": 0.42, "temporal": 0.10, "anomaly": 0.55})
    ...  # doctest: +SKIP
    {'fused_risk_score': 0.78, 'fused_risk_band': 'Critical', 'coverage': 1.0, ...}
    """
    ref = reference if reference is not None else load_reference(reference_path)

    calibrated: dict[str, float] = {}
    raw: dict[str, float | None] = {}
    for name in SIGNAL_ORDER:
        value = signals.get(name)
        raw[name] = None if _is_missing(value) else float(value)
        if _is_missing(value) or not ref.has(name):
            calibrated[name] = float("nan")
        else:
            calibrated[name] = ref.transform_one(name, value)

    result = fuse_with_coverage(**calibrated)
    result["raw_signals"] = raw
    result["calibrated_signals"] = calibrated
    return result
