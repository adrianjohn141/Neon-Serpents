from __future__ import annotations

import asyncio
import base64
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse
from redis import Redis
from sqlalchemy import select

from .artifacts import ArtifactStore, manifest_public_key
from .auth import (
    AdminSession,
    SESSION_COOKIE,
    create_session,
    destroy_session,
    redis_client,
    require_admin,
    require_csrf,
    verify_admin_password,
)
from .config import get_settings
from .database import SessionLocal, initialize_database
from .jobs import run_training_job
from .intelligence import compare_releases, manifest_observation_hash, release_detail, release_summary
from .models import AuditEvent, Experiment, Release, TrainingMetricSample
from .purge import purge_experiment
from .queueing import training_queue
from .schemas import ActionResponse, ExperimentCreate, ExperimentView, IntelligenceOverview, LoginRequest, ReleaseView, TrainingMetricSampleView

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    yield


app = FastAPI(title="Neon Serpents Local Training API", version="1.0.0", lifespan=lifespan)


def audit(action: str, subject_id: str | None = None, detail: dict | None = None) -> None:
    with SessionLocal.begin() as db:
        db.add(AuditEvent(action=action, subject_id=subject_id, detail=detail or {}))


@app.get("/healthz")
def health() -> dict:
    with SessionLocal() as db:
        db.execute(select(Experiment.id).limit(1))
    Redis.from_url(settings.redis_url).ping()
    return {"status": "ok"}


@app.post("/auth/login")
def login(payload: LoginRequest, request: Request, response: Response) -> dict:
    client = redis_client(settings)
    address = request.client.host if request.client else "unknown"
    attempts_key = f"login-attempts:{address}"
    attempts = client.incr(attempts_key)
    if attempts == 1:
        client.expire(attempts_key, 900)
    if attempts > 10:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many login attempts.")
    if not verify_admin_password(payload.password, settings):
        audit("auth.login_failed", detail={"address": address})
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid administrator password.")
    client.delete(attempts_key)
    session = create_session(response, settings)
    audit("auth.login", detail={"address": address})
    return {"authenticated": True, "csrfToken": session.csrf}


@app.post("/auth/logout")
def logout(
    response: Response,
    session: Annotated[AdminSession, Depends(require_csrf)],
) -> dict:
    destroy_session(response, session.token, settings)
    audit("auth.logout")
    return {"authenticated": False}


@app.get("/auth/me")
def me(session: Annotated[AdminSession, Depends(require_admin)]) -> dict:
    return {"authenticated": True, "csrfToken": session.csrf}


@app.post("/experiments", response_model=ExperimentView, status_code=201)
def create_experiment(
    payload: ExperimentCreate,
    _: Annotated[AdminSession, Depends(require_csrf)],
) -> Experiment:
    with SessionLocal.begin() as db:
        active = db.scalar(select(Experiment).where(Experiment.status.in_(["queued", "preflight", "training", "evaluating", "paused"])))
        if active:
            raise HTTPException(status_code=409, detail="Only one local training experiment may run at a time.")
        experiment = Experiment(config=payload.model_dump(), progress={"environmentSteps": 0, "seedIndex": 0})
        db.add(experiment)
        db.flush()
        experiment_id = experiment.id
    training_queue().enqueue(run_training_job, experiment_id, job_id=f"experiment-{experiment_id}")
    audit("experiment.created", experiment_id, payload.model_dump())
    return experiment


@app.get("/experiments", response_model=list[ExperimentView])
def list_experiments(_: Annotated[AdminSession, Depends(require_admin)]) -> list[Experiment]:
    with SessionLocal() as db:
        return list(db.scalars(select(Experiment).order_by(Experiment.created_at.desc()).limit(100)))


@app.get("/experiments/{experiment_id}", response_model=ExperimentView)
def get_experiment(experiment_id: str, _: Annotated[AdminSession, Depends(require_admin)]) -> Experiment:
    with SessionLocal() as db:
        experiment = db.get(Experiment, experiment_id)
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found.")
        return experiment


def request_experiment_action(experiment_id: str, action: str) -> bool:
    purge_after_commit = False
    with SessionLocal.begin() as db:
        experiment = db.get(Experiment, experiment_id)
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found.")
        if action == "resume":
            if experiment.status != "paused":
                raise HTTPException(status_code=409, detail="Only a paused experiment can resume.")
            experiment.status = "queued"
            experiment.requested_action = None
            training_queue().enqueue(run_training_job, experiment_id, job_id=f"experiment-{experiment_id}-resume-{int(datetime.now().timestamp())}")
        elif action == "finish" and experiment.status == "paused":
            experiment.status = "queued"
            experiment.requested_action = "finish"
            training_queue().enqueue(run_training_job, experiment_id, job_id=f"experiment-{experiment_id}-finish-{int(datetime.now().timestamp())}")
        elif action == "cancel" and experiment.status in {"paused", "cancelled", "failed"}:
            purge_after_commit = True
        else:
            experiment.requested_action = action
    if purge_after_commit:
        purge_experiment(experiment_id)
    return purge_after_commit


@app.post("/experiments/{experiment_id}/{action}", response_model=ActionResponse)
def experiment_action(
    experiment_id: str,
    action: str,
    _: Annotated[AdminSession, Depends(require_csrf)],
) -> ActionResponse:
    if action not in {"pause", "resume", "cancel", "finish"}:
        raise HTTPException(status_code=404, detail="Unknown experiment action.")
    purged = request_experiment_action(experiment_id, action)
    if not purged:
        audit(f"experiment.{action}_requested", experiment_id)
    return ActionResponse()


@app.post("/legacy-baselines/{experiment_id}/continue", response_model=ExperimentView, status_code=201)
def continue_legacy_baseline(
    experiment_id: str,
    _: Annotated[AdminSession, Depends(require_csrf)],
) -> Experiment:
    with SessionLocal.begin() as db:
        active = db.scalar(select(Experiment).where(Experiment.status.in_(["queued", "preflight", "training", "evaluating", "paused"])))
        if active:
            raise HTTPException(status_code=409, detail="Finish or cancel the current experiment first.")
        baselines = db.scalars(select(Release).where(Release.status == "baseline")).all()
        if any(bool((release.metrics or {}).get("controlledBaseline")) for release in baselines):
            raise HTTPException(status_code=409, detail="The controlled v2 baseline already exists.")
        source = db.get(Experiment, experiment_id)
        if not source or source.status not in {"cancelled", "failed"} or not source.checkpoint_key:
            raise HTTPException(status_code=409, detail="A recoverable cancelled v2 checkpoint is required.")
        schema = source.progress.get("metricsSchemaVersion")
        if schema not in {None, 2} or int(source.progress.get("environmentSteps", 0)) <= 0:
            raise HTTPException(status_code=409, detail="This checkpoint is not a legacy v2 training run.")
        config = {
            **source.config,
            "training_spec_version": 2,
            "baseline_mode": True,
            "resume_checkpoint_key": source.checkpoint_key,
            "seed_count": 1,
            "benchmark_matches": min(40, max(2, int(source.config.get("benchmark_matches", 20)))),
        }
        experiment = Experiment(config=config, progress={"environmentSteps": int(source.progress.get("environmentSteps", 0)), "seedIndex": 0, "sourceExperimentId": source.id})
        db.add(experiment)
        db.flush()
        new_experiment_id = experiment.id
    training_queue().enqueue(run_training_job, new_experiment_id, job_id=f"experiment-{new_experiment_id}")
    audit("experiment.legacy_baseline_continued", new_experiment_id, {"sourceExperimentId": experiment_id})
    return experiment


@app.get("/experiments/{experiment_id}/events")
async def experiment_events(experiment_id: str, _: Annotated[AdminSession, Depends(require_admin)]) -> StreamingResponse:
    def snapshot() -> tuple[str, str] | None:
        with SessionLocal() as db:
            experiment = db.get(Experiment, experiment_id)
            if not experiment:
                return None
            payload = json.dumps(ExperimentView.model_validate(experiment).model_dump(mode="json"), separators=(",", ":"))
            return payload, experiment.status

    async def stream():
        last = ""
        while True:
            current = await asyncio.to_thread(snapshot)
            if not current:
                yield "event: error\ndata: {\"message\":\"Experiment not found\"}\n\n"
                return
            payload, experiment_status = current
            if payload != last:
                yield f"data: {payload}\n\n"
                last = payload
            if experiment_status in {"candidate", "failed", "cancelled", "completed"}:
                return
            await asyncio.sleep(1)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/releases", response_model=list[ReleaseView])
def list_releases(_: Annotated[AdminSession, Depends(require_admin)]) -> list[Release]:
    with SessionLocal() as db:
        return list(db.scalars(select(Release).order_by(Release.created_at.desc()).limit(100)))


@app.get("/intelligence/overview", response_model=IntelligenceOverview)
def intelligence_overview(_: Annotated[AdminSession, Depends(require_admin)]) -> dict:
    active_statuses = ["queued", "preflight", "training", "evaluating", "paused"]
    with SessionLocal() as db:
        active = db.scalar(select(Experiment).where(Experiment.status.in_(active_statuses)).order_by(Experiment.created_at.desc()).limit(1))
        releases = list(db.scalars(select(Release).order_by(Release.created_at.desc()).limit(100)))
        production = next((release for release in releases if release.status == "production"), None)
        candidate = next((release for release in releases if release.status == "candidate"), None)
        latest_sample = None
        if active:
            latest_sample = db.scalar(select(TrainingMetricSample).where(TrainingMetricSample.experiment_id == active.id).order_by(TrainingMetricSample.created_at.desc()).limit(1))
        coverage = "full" if latest_sample else "partial" if active or releases else "unavailable"
        return {
            "coverage": coverage,
            "activeExperiment": ExperimentView.model_validate(active).model_dump(mode="json") if active else None,
            "latestSample": latest_sample,
            "production": release_summary(production),
            "latestCandidate": release_summary(candidate),
            "releases": [release_summary(release) for release in releases],
        }


@app.get("/intelligence/experiments/{experiment_id}/series", response_model=list[TrainingMetricSampleView])
def intelligence_series(experiment_id: str, _: Annotated[AdminSession, Depends(require_admin)]) -> list[TrainingMetricSample]:
    with SessionLocal() as db:
        if not db.get(Experiment, experiment_id):
            raise HTTPException(status_code=404, detail="Experiment not found.")
        return list(db.scalars(select(TrainingMetricSample).where(TrainingMetricSample.experiment_id == experiment_id).order_by(TrainingMetricSample.seed_index, TrainingMetricSample.environment_steps, TrainingMetricSample.created_at)))


@app.get("/intelligence/releases/{release_id}")
def intelligence_release(release_id: str, _: Annotated[AdminSession, Depends(require_admin)]) -> dict:
    with SessionLocal() as db:
        release = db.get(Release, release_id)
        if not release:
            raise HTTPException(status_code=404, detail="Release not found.")
        return release_detail(release)


@app.get("/intelligence/compare")
def intelligence_compare(candidate: str, baseline: str, _: Annotated[AdminSession, Depends(require_admin)]) -> dict:
    with SessionLocal() as db:
        candidate_release = db.get(Release, candidate)
        baseline_release = db.get(Release, baseline)
        if not candidate_release or not baseline_release:
            raise HTTPException(status_code=404, detail="Comparison release not found.")
        candidate_hash = manifest_observation_hash(candidate_release)
        baseline_hash = manifest_observation_hash(baseline_release)
        compatible_cross_version = {candidate_hash, baseline_hash} == {
            "neon-serpents:v2:observation-159",
            "neon-serpents:v3:observation-228",
        }
        if candidate_hash and baseline_hash and candidate_hash != baseline_hash and not compatible_cross_version:
            raise HTTPException(status_code=409, detail="Releases use incompatible observation specifications.")
        return compare_releases(candidate_release, baseline_release)


@app.post("/releases/{release_id}/promote", response_model=ReleaseView)
def promote_release(
    release_id: str,
    _: Annotated[AdminSession, Depends(require_csrf)],
    override: bool = Query(False),
) -> Release:
    manual_override = False
    with SessionLocal.begin() as db:
        release = db.get(Release, release_id)
        if not release:
            raise HTTPException(status_code=404, detail="Release not found.")
        if release.status not in {"candidate", "archived"}:
            raise HTTPException(status_code=409, detail="Release cannot be promoted.")
        if release.metrics.get("eligible") is not True:
            if not (override and settings.allow_manual_promotion):
                raise HTTPException(status_code=409, detail="Release did not pass the benchmark gate.")
            if release.metrics.get("coverage") != "full":
                raise HTTPException(status_code=409, detail="Testing promotion requires full benchmark coverage.")
            manual_override = True
            release.metrics = {
                **(release.metrics or {}),
                "eligible": True,
                "manualPromotion": True,
                "promotionOverride": "administrator_testing_override",
            }
        current = db.scalar(select(Release).where(Release.status == "production"))
        if current:
            current.status = "archived"
        release.status = "production"
        release.promoted_at = datetime.now(timezone.utc)
    audit("release.promoted_manual" if manual_override else "release.promoted", release_id, {"override": manual_override})
    return release


@app.post("/releases/rollback", response_model=ReleaseView)
def rollback_release(_: Annotated[AdminSession, Depends(require_csrf)]) -> Release:
    with SessionLocal.begin() as db:
        current = db.scalar(select(Release).where(Release.status == "production"))
        previous = db.scalar(select(Release).where(Release.status == "archived").order_by(Release.promoted_at.desc()).limit(1))
        if not previous:
            raise HTTPException(status_code=409, detail="No previous release is available.")
        if current:
            current.status = "archived"
        previous.status = "production"
        previous.promoted_at = datetime.now(timezone.utc)
    audit("release.rollback", previous.id)
    return previous


@app.get("/production/manifest")
def production_manifest() -> Response:
    with SessionLocal() as db:
        release = db.scalar(select(Release).where(Release.status == "production"))
        if not release:
            raise HTTPException(status_code=404, detail="No production release is available.")
        response = JSONResponse(release.manifest)
        response.headers["Cache-Control"] = "public, max-age=60"
        response.headers["ETag"] = f'"{release.id}"'
        return response


@app.get("/production/public-key")
def production_public_key() -> dict:
    return {"algorithm": "Ed25519", "publicKey": manifest_public_key()}


@app.get("/production/bundles/{release_id}/{snake_id}")
def production_bundle(release_id: str, snake_id: str) -> StreamingResponse:
    with SessionLocal() as db:
        release = db.get(Release, release_id)
        if not release or release.status not in {"production", "archived"}:
            raise HTTPException(status_code=404, detail="Release not found.")
        manifest_payload = json.loads(base64.b64decode(release.manifest["payload"]))
        brain = next((entry for entry in manifest_payload.get("brains", []) if entry.get("snakeId") == snake_id), None)
        if not brain:
            raise HTTPException(status_code=404, detail="Snake brain not found in release.")
    stored = ArtifactStore().get_object(brain["key"])
    return StreamingResponse(stored["Body"], media_type="application/json", headers={
        "Cache-Control": "public, max-age=31536000, immutable",
        "ETag": f'"{brain["sha256"]}"',
    })
