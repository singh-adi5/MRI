"""Append-only persistence for analyst dispositions.

Before this existed an analyst decision left the workbench as a browser download
and was gone. Here it becomes a row, which is what a future retrain needs:
``label_set()`` hands back the confirmed dispositions in label shape.

Security posture (see SECURITY.md, A03): every statement is a module-level
literal and every value is bound with a ``?`` placeholder. ``list_decisions``
builds its WHERE clause from a fixed allow-list of fragments, never from caller
text, so no identifier is interpolated either.
"""
from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Iterable, Sequence

from mri.config import OUTPUT_DIR
from mri.store.schema import (
    DECISION_VALUES,
    ENTITY_TYPES,
    SCHEMA_VERSION,
    apply_migrations,
    current_version,
)

DEFAULT_DB_PATH: Final[Path] = Path(OUTPUT_DIR) / "decisions.db"

_MAX_LIMIT: Final[int] = 1000
_DEFAULT_LIMIT: Final[int] = 100

_INSERT: Final[str] = """
INSERT INTO decisions (
    entity_type, entity_id, analyst, decision, reason_code, notes,
    fused_score_at_decision, model_version_at_decision, created_utc
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_SELECT_COLUMNS: Final[str] = """
SELECT id, entity_type, entity_id, analyst, decision, reason_code, notes,
       fused_score_at_decision, model_version_at_decision, created_utc
FROM decisions
"""

# Fixed fragments. The key is chosen by the API layer; the caller's value is
# always bound, never spliced.
_FILTERS: Final[dict[str, str]] = {
    "entity_type": "entity_type = ?",
    "entity_id": "entity_id = ?",
    "analyst": "analyst = ?",
    "decision": "decision = ?",
    "since_utc": "created_utc >= ?",
}

# Confirmed dispositions only: needs_info is an open question, not a label.
_LABEL_MAP: Final[dict[str, int]] = {"confirmed_fraud": 1, "confirmed_good": 0}

_SELECT_LABELS: Final[str] = """
SELECT entity_type, entity_id, decision, MAX(created_utc) AS decided_utc, COUNT(*) AS n
FROM decisions
WHERE decision IN ('confirmed_fraud', 'confirmed_good')
GROUP BY entity_type, entity_id, decision
"""

_SELECT_LATEST_PER_ENTITY: Final[str] = """
SELECT d.entity_type, d.entity_id, d.decision, d.reason_code, d.created_utc
FROM decisions AS d
JOIN (
    SELECT entity_type, entity_id, MAX(id) AS max_id
    FROM decisions
    WHERE decision IN ('confirmed_fraud', 'confirmed_good')
    GROUP BY entity_type, entity_id
) AS latest
  ON d.id = latest.max_id
ORDER BY d.entity_type, d.entity_id
"""

_COUNT_ALL: Final[str] = "SELECT COUNT(*) FROM decisions"


class DecisionStoreError(RuntimeError):
    """Raised when a decision cannot be recorded or read."""


class ValidationError(DecisionStoreError):
    """Raised when a caller supplies a value the schema would reject."""


@dataclass(frozen=True, slots=True)
class Decision:
    """One recorded disposition, exactly as stored."""

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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Label:
    """One entity's current label, derived from its latest confirmed disposition."""

    entity_type: str
    entity_id: str
    label: int
    decision: str
    reason_code: str
    decided_utc: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    # Triggers are the engine-level half of the append-only guarantee; SQLite
    # only honours them when foreign/trigger enforcement is on for the handle.
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _clean_text(value: Any, field: str, *, max_len: int, required: bool = True) -> str | None:
    if value is None:
        _require(not required, f"{field} is required")
        return None
    if not isinstance(value, str):
        raise ValidationError(f"{field} must be a string")
    text = value.strip()
    if not text:
        _require(not required, f"{field} must not be blank")
        return None
    _require(len(text) <= max_len, f"{field} must be at most {max_len} characters")
    return text


class DecisionStore:
    """Append-only SQLite store for analyst dispositions.

    The store opens a connection per call. At demo volumes that costs nothing and
    it keeps the object safe to share between FastAPI worker threads without a
    lock of its own.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        self._bootstrapped = False

    def bootstrap(self) -> int:
        """Create or upgrade the schema. Idempotent; safe on every start."""
        with _connect(self.db_path) as conn:
            version = apply_migrations(conn)
        self._bootstrapped = True
        return version

    def _conn(self) -> sqlite3.Connection:
        if not self._bootstrapped:
            self.bootstrap()
        return _connect(self.db_path)

    @property
    def schema_version(self) -> int:
        with self._conn() as conn:
            return current_version(conn)

    def is_ready(self) -> bool:
        """True when the schema is present and at the version this code expects."""
        try:
            return self.schema_version == SCHEMA_VERSION
        except sqlite3.Error:
            return False

    def record_decision(
        self,
        *,
        entity_type: str,
        entity_id: str,
        analyst: str,
        decision: str,
        reason_code: str,
        notes: str | None = None,
        fused_score_at_decision: float | None = None,
        model_version_at_decision: str | None = None,
        created_utc: str | None = None,
    ) -> Decision:
        """Append one disposition and return it as stored."""
        entity_type = (_clean_text(entity_type, "entity_type", max_len=32) or "").lower()
        _require(entity_type in ENTITY_TYPES, f"entity_type must be one of {list(ENTITY_TYPES)}")

        entity_id = _clean_text(entity_id, "entity_id", max_len=64) or ""
        analyst = _clean_text(analyst, "analyst", max_len=128) or ""

        decision = (_clean_text(decision, "decision", max_len=32) or "").lower()
        _require(decision in DECISION_VALUES, f"decision must be one of {list(DECISION_VALUES)}")

        reason_code = _clean_text(reason_code, "reason_code", max_len=64) or ""
        notes_clean = _clean_text(notes, "notes", max_len=4000, required=False)
        version_clean = _clean_text(
            model_version_at_decision, "model_version_at_decision", max_len=128, required=False
        )

        if fused_score_at_decision is not None:
            try:
                score = float(fused_score_at_decision)
            except (TypeError, ValueError) as exc:
                raise ValidationError("fused_score_at_decision must be a number") from exc
            _require(
                0.0 <= score <= 1.0,
                "fused_score_at_decision must be within [0, 1] (the fused scale)",
            )
        else:
            score = None

        stamp = _clean_text(created_utc, "created_utc", max_len=64, required=False) or _utc_now()

        params = (
            entity_type,
            entity_id,
            analyst,
            decision,
            reason_code,
            notes_clean,
            score,
            version_clean,
            stamp,
        )
        try:
            with self._conn() as conn:
                cursor = conn.execute(_INSERT, params)
                new_id = int(cursor.lastrowid or 0)
        except sqlite3.IntegrityError as exc:
            raise ValidationError(f"decision rejected by the schema: {exc}") from exc
        except sqlite3.Error as exc:  # pragma: no cover - surfaced as a 503 by the API
            raise DecisionStoreError(f"could not record decision: {exc}") from exc

        return Decision(
            id=new_id,
            entity_type=entity_type,
            entity_id=entity_id,
            analyst=analyst,
            decision=decision,
            reason_code=reason_code,
            notes=notes_clean,
            fused_score_at_decision=score,
            model_version_at_decision=version_clean,
            created_utc=stamp,
        )

    def list_decisions(
        self,
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
        analyst: str | None = None,
        decision: str | None = None,
        since_utc: str | None = None,
        limit: int = _DEFAULT_LIMIT,
        offset: int = 0,
    ) -> list[Decision]:
        """Return dispositions newest-first, optionally filtered."""
        supplied: dict[str, Any] = {
            "entity_type": entity_type.lower().strip() if isinstance(entity_type, str) else None,
            "entity_id": entity_id.strip() if isinstance(entity_id, str) else None,
            "analyst": analyst.strip() if isinstance(analyst, str) else None,
            "decision": decision.lower().strip() if isinstance(decision, str) else None,
            "since_utc": since_utc.strip() if isinstance(since_utc, str) else None,
        }

        clauses: list[str] = []
        params: list[Any] = []
        for key, value in supplied.items():
            if value in (None, ""):
                continue
            # The fragment comes from the fixed _FILTERS map; the value is bound.
            clauses.append(_FILTERS[key])
            params.append(value)

        limit = max(1, min(int(limit), _MAX_LIMIT))
        offset = max(0, int(offset))

        sql = _SELECT_COLUMNS
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._conn() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [_row_to_decision(row) for row in rows]

    def count(self) -> int:
        with self._conn() as conn:
            return int(conn.execute(_COUNT_ALL).fetchone()[0])

    def label_set(self) -> list[Label]:
        """Current labels, one per entity, for a future supervised retrain.

        Only confirmed dispositions count. ``needs_info`` is an open question, so
        it yields no label. When an analyst has changed their mind the latest row
        wins, which is how an append-only log expresses a correction.
        """
        with self._conn() as conn:
            rows = conn.execute(_SELECT_LATEST_PER_ENTITY).fetchall()
        return [
            Label(
                entity_type=row["entity_type"],
                entity_id=row["entity_id"],
                label=_LABEL_MAP[row["decision"]],
                decision=row["decision"],
                reason_code=row["reason_code"],
                decided_utc=row["created_utc"],
            )
            for row in rows
            if row["decision"] in _LABEL_MAP
        ]

    def label_summary(self) -> dict[str, int]:
        labels = self.label_set()
        return {
            "entities_labelled": len(labels),
            "positive": sum(1 for item in labels if item.label == 1),
            "negative": sum(1 for item in labels if item.label == 0),
        }

    def disposition_counts(self) -> dict[str, int]:
        counts = {value: 0 for value in DECISION_VALUES}
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT decision, COUNT(*) AS n FROM decisions GROUP BY decision"
            ).fetchall()
        for row in rows:
            counts[row["decision"]] = int(row["n"])
        return counts


def _row_to_decision(row: sqlite3.Row) -> Decision:
    return Decision(
        id=int(row["id"]),
        entity_type=row["entity_type"],
        entity_id=row["entity_id"],
        analyst=row["analyst"],
        decision=row["decision"],
        reason_code=row["reason_code"],
        notes=row["notes"],
        fused_score_at_decision=(
            float(row["fused_score_at_decision"])
            if row["fused_score_at_decision"] is not None
            else None
        ),
        model_version_at_decision=row["model_version_at_decision"],
        created_utc=row["created_utc"],
    )


_default_store: DecisionStore | None = None


def get_store(db_path: str | Path | None = None) -> DecisionStore:
    """Process-wide store. Pass ``db_path`` in tests to get an isolated one."""
    global _default_store
    if db_path is not None:
        return DecisionStore(db_path)
    if _default_store is None:
        _default_store = DecisionStore()
        _default_store.bootstrap()
    return _default_store


# Module-level conveniences so a caller never has to hold the object.
def record_decision(**kwargs: Any) -> Decision:
    return get_store().record_decision(**kwargs)


def list_decisions(**kwargs: Any) -> list[Decision]:
    return get_store().list_decisions(**kwargs)


def label_set() -> list[Label]:
    return get_store().label_set()


def as_dicts(items: Iterable[Decision] | Sequence[Label]) -> list[dict[str, Any]]:
    return [item.as_dict() for item in items]
