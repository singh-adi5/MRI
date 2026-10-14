"""MRI scoring service.

This is what makes MRI a service rather than a notebook: the same fusion the
batch pipeline applies, exposed over HTTP for one entity at a time, with the
analyst's disposition written back to a durable store instead of a download.

Run locally:
    PYTHONPATH=src uvicorn mri.api.main:app --port 8000

Auth is OFF by default. Set MRI_API_AUTH_ENABLED=1 and MRI_API_KEY=<secret> to
require the X-API-Key header on every route except /healthz and /readyz.
"""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, Final

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from mri.api.artifacts import MODEL_VERSION, ArtifactStore
from mri.api.logging_config import REQUEST_ID_HEADER, configure_logging, request_id_var
from mri.api.scoring_adapter import SIGNAL_LABELS
from mri.api.schemas import (
    CalibrationInfo,
    DecisionListResponse,
    DecisionRequest,
    DecisionResponse,
    EntityResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    ReadyResponse,
    ScoreRequest,
    ScoreResponse,
    SignalContribution,
)
from mri.api.security import (
    SlidingWindowLimiter,
    enforce_rate_limit,
    require_api_key,
)
from mri.api.settings import Settings, load_settings
from mri.store import DecisionStore, DecisionStoreError
from mri.store import ValidationError as StoreValidationError

log = logging.getLogger("mri.api")

SERVICE_NAME: Final[str] = "mri-scoring-api"
ALERT_BANDS: Final[tuple[str, ...]] = ("High", "Critical")

_ERROR_RESPONSES: Final[dict[int | str, dict[str, Any]]] = {
    400: {"model": ErrorResponse},
    401: {"model": ErrorResponse},
    403: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    429: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


# --------------------------------------------------------------------------
# application factory
# --------------------------------------------------------------------------
def create_app(
    settings: Settings | None = None,
    artifacts: ArtifactStore | None = None,
    store: DecisionStore | None = None,
) -> FastAPI:
    """Build the app. Every collaborator is injectable so tests get isolation."""
    resolved = settings or load_settings()
    configure_logging(resolved.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.artifacts = (artifacts or ArtifactStore()).load()
        application.state.store = store or DecisionStore(resolved.decisions_db)
        application.state.store.bootstrap()
        log.info(
            "service started",
            extra={
                "auth_enabled": resolved.auth_active,
                "rate_limit": f"{resolved.rate_limit}/{resolved.rate_window_seconds}s",
                "scoring_source": application.state.artifacts.readiness_detail()[
                    "scoring_source"
                ],
                "ready": application.state.artifacts.is_ready(),
            },
        )
        yield
        log.info("service stopping")

    app = FastAPI(
        title="MRI scoring API",
        version="1.0.0",
        summary="Multi-signal marketplace fraud scoring, one entity at a time.",
        description=__doc__,
        lifespan=lifespan,
        responses=_ERROR_RESPONSES,
    )
    app.state.settings = resolved
    app.state.limiter = SlidingWindowLimiter(resolved.rate_limit, resolved.rate_window_seconds)

    _install_middleware(app)
    _install_error_handlers(app)
    _install_routes(app)
    return app


def _install_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        # Accept an inbound id so a trace survives a hop, but only if it is
        # short and printable: it ends up in logs and in a response header.
        inbound = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = (
            inbound
            if inbound.isascii() and inbound.isprintable() and 0 < len(inbound) <= 64
            else uuid.uuid4().hex
        )
        token = request_id_var.set(request_id)
        request.state.request_id = request_id
        try:
            response: Response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers[REQUEST_ID_HEADER] = request_id
        log.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
            },
        )
        return response


def _error(status_code: int, error: str, detail: str, request: Request) -> JSONResponse:
    """The single error shape. No traceback, no internal path, ever."""
    body = ErrorResponse(
        error=error,
        detail=detail,
        status_code=status_code,
        request_id=getattr(request.state, "request_id", request_id_var.get()),
    )
    return JSONResponse(status_code=status_code, content=body.model_dump())


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        response = _error(
            exc.status_code,
            "http_error",
            str(exc.detail),
            request,
        )
        for key, value in (getattr(exc, "headers", None) or {}).items():
            response.headers[key] = value
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Pydantic's own messages name the field and the rule, which is useful
        # to a client and reveals nothing about the server.
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err.get('loc', ()))}: {err.get('msg', 'invalid')}"
            for err in exc.errors()
        )
        return _error(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "validation_error",
            problems or "request body failed validation",
            request,
        )

    @app.exception_handler(PydanticValidationError)
    async def model_error(request: Request, exc: PydanticValidationError) -> JSONResponse:
        log.warning("response model validation failed", exc_info=exc)
        return _error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal_error",
            "the service produced a response it could not validate",
            request,
        )

    @app.exception_handler(StoreValidationError)
    async def store_validation(request: Request, exc: StoreValidationError) -> JSONResponse:
        return _error(status.HTTP_400_BAD_REQUEST, "invalid_decision", str(exc), request)

    @app.exception_handler(DecisionStoreError)
    async def store_error(request: Request, exc: DecisionStoreError) -> JSONResponse:
        log.error("decision store failure", exc_info=exc)
        return _error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "store_unavailable",
            "the decision store is not accepting writes",
            request,
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        # The traceback goes to the structured log under this request id; the
        # client gets the id and nothing else.
        log.exception("unhandled error", extra={"path": request.url.path})
        return _error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal_error",
            "an unexpected error occurred; quote the request id when reporting it",
            request,
        )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _contributions(
    scoring: Any,
    raw: dict[str, float],
    normalised: dict[str, float],
    contributions: dict[str, float],
    fused: float,
) -> list[SignalContribution]:
    return [
        SignalContribution(
            signal=key,
            label=SIGNAL_LABELS.get(key, key),
            weight=scoring.weights[key],
            raw=round(raw.get(key, 0.0), 6),
            normalised=round(normalised.get(key, 0.0), 6),
            contribution=round(contributions.get(key, 0.0), 6),
            share_of_score=round(contributions.get(key, 0.0) / fused, 6) if fused > 0 else 0.0,
        )
        for key in scoring.weights
    ]


def _explain(entity_type: str, entity_id: str, fused: float, band: str, parts: list[SignalContribution]) -> str:
    ranked = sorted(parts, key=lambda p: p.contribution, reverse=True)
    lead = ranked[0] if ranked else None
    drivers = ", ".join(f"{p.label} {p.contribution:.3f}" for p in ranked[:3])
    opener = (
        f"{entity_type.title()} {entity_id} scores {fused:.3f} ({band})."
    )
    if lead is None or lead.contribution == 0.0:
        return f"{opener} No signal contributed: every input was at or below the population floor."
    return (
        f"{opener} {lead.label} is the largest driver at "
        f"{lead.share_of_score * 100:.0f}% of the score. Top contributions: {drivers}."
    )


def _require_ready(request: Request) -> ArtifactStore:
    artifacts: ArtifactStore = request.app.state.artifacts
    if not artifacts.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "scoring artefacts are not loaded; run `python -m mri.pipeline --no-ui` "
                "then restart the service"
            ),
        )
    return artifacts


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------
def _install_routes(app: FastAPI) -> None:
    auth = Depends(require_api_key)
    rate_limit = Depends(enforce_rate_limit)

    @app.get("/healthz", response_model=HealthResponse, tags=["ops"])
    async def healthz() -> HealthResponse:
        """Liveness: the process is up. Never touches an artefact or the disk."""
        return HealthResponse(status="ok", service=SERVICE_NAME, model_version=MODEL_VERSION)

    @app.get("/readyz", response_model=ReadyResponse, tags=["ops"])
    async def readyz(request: Request, response: Response) -> ReadyResponse:
        """Readiness: scoring artefacts loaded and the decision store writable."""
        artifacts: ArtifactStore = request.app.state.artifacts
        store: DecisionStore = request.app.state.store
        detail = artifacts.readiness_detail()
        detail["decision_store"] = store.is_ready()
        detail["auth_enabled"] = request.app.state.settings.auth_active
        ready = artifacts.is_ready() and store.is_ready()
        if not ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadyResponse(status="ready" if ready else "not_ready", detail=detail)

    @app.post(
        "/score",
        response_model=ScoreResponse,
        dependencies=[auth, rate_limit],
        tags=["scoring"],
    )
    async def score(request: Request, payload: ScoreRequest) -> ScoreResponse:
        """Score one entity against the frozen calibration reference."""
        artifacts = _require_ready(request)
        scoring = artifacts.scoring
        reference = artifacts.reference
        assert scoring is not None and reference is not None  # nosec B101 - guarded above

        raw = payload.signals.model_dump()
        normalised = reference.normalise_all(raw)
        fused = round(scoring.fuse(normalised), 6)
        band = scoring.band_for(fused)
        parts = _contributions(scoring, raw, normalised, scoring.contributions(normalised), fused)

        log.info(
            "scored entity",
            extra={
                "entity_type": payload.entity_type,
                "entity_id": payload.entity_id,
                "fused_score": fused,
                "band": band,
            },
        )
        return ScoreResponse(
            entity_type=payload.entity_type,
            entity_id=payload.entity_id,
            fused_score=fused,
            band=band,
            contributions=parts,
            explanation=_explain(payload.entity_type, payload.entity_id, fused, band, parts),
            model_version=MODEL_VERSION,
            scoring_source=scoring.source,
            calibration=CalibrationInfo(
                source=reference.built_from,
                built_from=reference.built_from,
                population_rows=reference.population_rows,
                built_utc=reference.built_utc,
                out_of_range_signals=reference.out_of_range(raw),
            ),
            request_id=request.state.request_id,
        )

    @app.get(
        "/entities/{entity_type}/{entity_id}",
        response_model=EntityResponse,
        dependencies=[auth],
        tags=["entities"],
    )
    async def get_entity(
        request: Request,
        entity_type: str,
        entity_id: str,
    ) -> EntityResponse:
        """The stored score and evidence for an entity the batch run scored."""
        artifacts = _require_ready(request)
        normalised_type = entity_type.strip().lower()
        if normalised_type not in ("customer", "seller"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="entity_type must be 'customer' or 'seller'",
            )
        entity = artifacts.get(normalised_type, entity_id)
        if entity is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"no scored {normalised_type} with id '{entity_id}'",
            )

        scoring = artifacts.scoring
        assert scoring is not None  # nosec B101 - guarded by _require_ready
        store: DecisionStore = request.app.state.store
        parts = _contributions(
            scoring,
            entity.raw_signals,
            entity.normalised_signals,
            entity.contributions,
            entity.fused_score,
        )
        return EntityResponse(
            entity_type=entity.entity_type,
            entity_id=entity.entity_id,
            fused_score=entity.fused_score,
            band=entity.band,
            rank=entity.rank,
            population_size=len(artifacts.entities),
            contributions=parts,
            reason_codes=entity.reason_codes,
            explanation=entity.explanation
            or _explain(entity.entity_type, entity.entity_id, entity.fused_score, entity.band, parts),
            evidence=entity.evidence,
            model_version=MODEL_VERSION,
            decisions_recorded=len(
                store.list_decisions(
                    entity_type=entity.entity_type, entity_id=entity.entity_id, limit=1000
                )
            ),
        )

    @app.get("/metrics", response_model=MetricsResponse, dependencies=[auth], tags=["ops"])
    async def metrics(request: Request) -> MetricsResponse:
        """Alert rate, band counts and the evaluation summary."""
        artifacts = _require_ready(request)
        store: DecisionStore = request.app.state.store
        scoring = artifacts.scoring
        assert scoring is not None  # nosec B101 - guarded by _require_ready
        return MetricsResponse(
            population_size=len(artifacts.entities),
            band_counts=artifacts.band_counts,
            alert_rate=artifacts.alert_rate(ALERT_BANDS),
            alert_bands=list(ALERT_BANDS),
            top_decile_threshold=artifacts.top_decile_threshold(),
            decisions={**store.disposition_counts(), "total": store.count()},
            labels=store.label_summary(),
            evaluation=artifacts.evaluation,
            model_version=MODEL_VERSION,
            scoring_source=scoring.source,
        )

    @app.post(
        "/decisions",
        response_model=DecisionResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[auth],
        tags=["decisions"],
    )
    async def create_decision(request: Request, payload: DecisionRequest) -> DecisionResponse:
        """Record an analyst disposition. Append-only; nothing is ever overwritten."""
        store: DecisionStore = request.app.state.store
        recorded = store.record_decision(
            entity_type=payload.entity_type,
            entity_id=payload.entity_id,
            analyst=payload.analyst,
            decision=payload.decision,
            reason_code=payload.reason_code,
            notes=payload.notes,
            fused_score_at_decision=payload.fused_score_at_decision,
            model_version_at_decision=payload.model_version_at_decision or MODEL_VERSION,
        )
        log.info(
            "decision recorded",
            extra={
                "decision_id": recorded.id,
                "entity_type": recorded.entity_type,
                "entity_id": recorded.entity_id,
                "disposition": recorded.decision,
            },
        )
        return DecisionResponse(**recorded.as_dict())

    @app.get(
        "/decisions",
        response_model=DecisionListResponse,
        dependencies=[auth],
        tags=["decisions"],
    )
    async def list_decisions(
        request: Request,
        entity_type: str | None = Query(None, pattern=r"^(customer|seller)$"),
        entity_id: str | None = Query(None, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$"),
        analyst: str | None = Query(None, max_length=128),
        decision: str | None = Query(
            None, pattern=r"^(confirmed_fraud|confirmed_good|needs_info)$"
        ),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> DecisionListResponse:
        """Read back recorded dispositions, newest first."""
        store: DecisionStore = request.app.state.store
        rows = store.list_decisions(
            entity_type=entity_type,
            entity_id=entity_id,
            analyst=analyst,
            decision=decision,
            limit=limit,
            offset=offset,
        )
        return DecisionListResponse(
            count=len(rows),
            limit=limit,
            offset=offset,
            decisions=[DecisionResponse(**row.as_dict()) for row in rows],
        )


app = create_app()
