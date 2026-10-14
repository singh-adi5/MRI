"""Runtime configuration, read from the environment once at import.

Defaults are chosen for the local demo: auth OFF, generous rate limit, human
log lines off (JSON only). Nothing here reads a file or a secret store.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

ENV_AUTH_ENABLED: Final[str] = "MRI_API_AUTH_ENABLED"
ENV_API_KEY: Final[str] = "MRI_API_KEY"
ENV_RATE_LIMIT: Final[str] = "MRI_API_RATE_LIMIT"
ENV_RATE_WINDOW: Final[str] = "MRI_API_RATE_WINDOW_SECONDS"
ENV_LOG_LEVEL: Final[str] = "MRI_API_LOG_LEVEL"
ENV_DECISIONS_DB: Final[str] = "MRI_DECISIONS_DB"

API_KEY_HEADER: Final[str] = "X-API-Key"

_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable view of the service's environment."""

    auth_enabled: bool
    api_key: str | None
    rate_limit: int
    rate_window_seconds: int
    log_level: str
    decisions_db: str | None

    @property
    def auth_active(self) -> bool:
        """Auth only bites when it is enabled AND a key is actually configured."""
        return self.auth_enabled and bool(self.api_key)


def load_settings() -> Settings:
    key = os.getenv(ENV_API_KEY) or None
    if key is not None:
        key = key.strip() or None
    return Settings(
        # OFF by default: this is a public read-only demo (see SECURITY.md A07).
        # Set MRI_API_AUTH_ENABLED=1 and MRI_API_KEY=<secret> to require a key.
        auth_enabled=_flag(ENV_AUTH_ENABLED, False),
        api_key=key,
        rate_limit=_positive_int(ENV_RATE_LIMIT, 60),
        rate_window_seconds=_positive_int(ENV_RATE_WINDOW, 60),
        log_level=(os.getenv(ENV_LOG_LEVEL) or "INFO").upper(),
        decisions_db=os.getenv(ENV_DECISIONS_DB) or None,
    )
