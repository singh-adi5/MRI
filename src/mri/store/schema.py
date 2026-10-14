"""Schema definition and idempotent bootstrap for the decision store.

Migrations are forward-only and numbered. ``apply_migrations`` is safe to call on
every process start: it records what it has applied in ``schema_migrations`` and
replays nothing.

Security posture (see SECURITY.md, A03): every statement in this module is a
module-level literal. Nothing is formatted, concatenated or interpolated.
"""
from __future__ import annotations

import sqlite3
from typing import Final

SCHEMA_VERSION: Final[int] = 1

DECISION_VALUES: Final[tuple[str, ...]] = (
    "confirmed_fraud",
    "confirmed_good",
    "needs_info",
)
ENTITY_TYPES: Final[tuple[str, ...]] = ("customer", "seller")

_MIGRATION_TABLE: Final[str] = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER PRIMARY KEY,
    applied_utc TEXT NOT NULL
)
"""

# Append-only is enforced in three places: no UPDATE/DELETE exists in the Python
# API, triggers below reject them at the engine, and CHECK constraints keep the
# enumerations honest so a bad disposition cannot be written at all.
_MIGRATION_1: Final[tuple[str, ...]] = (
    """
    CREATE TABLE IF NOT EXISTS decisions (
        id                        INTEGER PRIMARY KEY AUTOINCREMENT,
        entity_type               TEXT NOT NULL
                                      CHECK (entity_type IN ('customer', 'seller')),
        entity_id                 TEXT NOT NULL CHECK (length(entity_id) BETWEEN 1 AND 64),
        analyst                   TEXT NOT NULL CHECK (length(analyst) BETWEEN 1 AND 128),
        decision                  TEXT NOT NULL
                                      CHECK (decision IN ('confirmed_fraud',
                                                          'confirmed_good',
                                                          'needs_info')),
        reason_code               TEXT NOT NULL CHECK (length(reason_code) BETWEEN 1 AND 64),
        notes                     TEXT CHECK (notes IS NULL OR length(notes) <= 4000),
        fused_score_at_decision   REAL CHECK (fused_score_at_decision IS NULL
                                              OR (fused_score_at_decision >= 0.0
                                                  AND fused_score_at_decision <= 1.0)),
        model_version_at_decision TEXT CHECK (model_version_at_decision IS NULL
                                              OR length(model_version_at_decision) <= 128),
        created_utc               TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_decisions_entity ON decisions (entity_type, entity_id)",
    "CREATE INDEX IF NOT EXISTS ix_decisions_created ON decisions (created_utc)",
    "CREATE INDEX IF NOT EXISTS ix_decisions_disposition ON decisions (decision)",
    """
    CREATE TRIGGER IF NOT EXISTS decisions_no_update
    BEFORE UPDATE ON decisions
    BEGIN
        SELECT RAISE(ABORT, 'decisions is append-only: UPDATE is not permitted');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS decisions_no_delete
    BEFORE DELETE ON decisions
    BEGIN
        SELECT RAISE(ABORT, 'decisions is append-only: DELETE is not permitted');
    END
    """,
)

MIGRATIONS: Final[dict[int, tuple[str, ...]]] = {1: _MIGRATION_1}


def apply_migrations(conn: sqlite3.Connection) -> int:
    """Bring ``conn`` up to ``SCHEMA_VERSION``. Returns the version now in force."""
    conn.execute(_MIGRATION_TABLE)
    applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
    for version in sorted(MIGRATIONS):
        if version in applied:
            continue
        for statement in MIGRATIONS[version]:
            conn.execute(statement)
        # Parameterised: the only value bound here is a timestamp.
        conn.execute(
            "INSERT INTO schema_migrations (version, applied_utc) "
            "VALUES (?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))",
            (version,),
        )
    conn.commit()
    return current_version(conn)


def current_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations").fetchone()
    return int(row[0]) if row else 0
