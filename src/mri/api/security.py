"""API-key auth and a small in-process rate limiter.

Auth is a single shared key in a header, compared in constant time. There is no
user or tenant model here on purpose: the project is single-tenant by design
(SECURITY.md, A01/A07) and inventing identities would be a bigger claim than the
code supports.
"""
from __future__ import annotations

import hmac
import threading
import time
from collections import deque
from typing import Deque, Final

from fastapi import Depends, HTTPException, Request, status

from mri.api.settings import API_KEY_HEADER, Settings


def require_api_key(request: Request) -> None:
    """FastAPI dependency. A no-op when auth is disabled, which is the default."""
    settings: Settings = request.app.state.settings
    if not settings.auth_active:
        return

    presented = request.headers.get(API_KEY_HEADER)
    if not presented:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"missing {API_KEY_HEADER} header",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )
    # compare_digest: no early exit, so a wrong key leaks no timing signal.
    if not hmac.compare_digest(presented, settings.api_key or ""):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="invalid API key",
        )


class SlidingWindowLimiter:
    """Fixed-cost sliding window, keyed by client. In-process and non-durable.

    Good enough for one container serving a demo. A multi-replica deployment
    needs a shared counter; that is called out in SECURITY.md.
    """

    def __init__(self, limit: int, window_seconds: int, max_keys: int = 4096) -> None:
        self.limit = limit
        self.window = float(window_seconds)
        self.max_keys = max_keys
        self._hits: dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> tuple[bool, int, float]:
        """Returns (allowed, remaining, retry_after_seconds)."""
        moment = time.monotonic() if now is None else now
        cutoff = moment - self.window
        with self._lock:
            if key not in self._hits and len(self._hits) >= self.max_keys:
                # Bounded memory: drop the coldest bucket rather than grow forever.
                self._evict_coldest(cutoff)
            bucket = self._hits.setdefault(key, deque())
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self.limit:
                retry_after = max(0.0, bucket[0] + self.window - moment)
                return False, 0, retry_after
            bucket.append(moment)
            return True, self.limit - len(bucket), 0.0

    def _evict_coldest(self, cutoff: float) -> None:
        stale = [key for key, bucket in self._hits.items() if not bucket or bucket[-1] <= cutoff]
        for key in stale:
            del self._hits[key]
        if len(self._hits) >= self.max_keys:
            oldest = min(self._hits, key=lambda k: self._hits[k][-1])
            del self._hits[oldest]

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


def client_key(request: Request) -> str:
    """Identify the caller for rate-limiting.

    The API key when one is presented, otherwise the peer address. Not a
    forwarded header: behind a proxy that would be caller-controlled, and
    trusting it would let anyone reset their own bucket.
    """
    presented = request.headers.get(API_KEY_HEADER)
    if presented:
        return f"key:{presented[:16]}"
    client = request.client
    return f"ip:{client.host}" if client else "ip:unknown"


def enforce_rate_limit(request: Request) -> None:
    """FastAPI dependency applied to the scoring endpoint."""
    limiter: SlidingWindowLimiter = request.app.state.limiter
    allowed, remaining, retry_after = limiter.check(client_key(request))
    request.state.rate_remaining = remaining
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="rate limit exceeded",
            headers={"Retry-After": str(max(1, int(retry_after + 0.999)))},
        )


AuthDependency: Final = Depends(require_api_key)
RateLimitDependency: Final = Depends(enforce_rate_limit)
