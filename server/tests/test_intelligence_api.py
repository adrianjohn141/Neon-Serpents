from __future__ import annotations

import base64
import json

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from server.app import main
from server.app.auth import AdminSession, require_admin, require_csrf
from server.app.database import Base
from server.app.models import Experiment, Release, TrainingMetricSample


def manifest(spec_hash: str) -> dict:
    return {"payload": base64.b64encode(json.dumps({"observationSpecHash": spec_hash}).encode()).decode()}


def test_intelligence_endpoints_require_admin_and_return_stored_data(monkeypatch) -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    Base.metadata.create_all(engine)
    with sessions.begin() as db:
        db.add(Experiment(id="experiment", status="paused", config={"stepsPerSeed": 1000}, progress={"environmentSteps": 500}))
        db.add(TrainingMetricSample(id="sample", experiment_id="experiment", seed_index=0, environment_steps=500, policy_version=3, sample_kind="interval", metrics={"metricsSchemaVersion": 2, "claimRate": 0.4}))
        db.add(Release(id="candidate", status="candidate", metrics={"metricsSchemaVersion": 2}, manifest=manifest("v2")))
        db.add(Release(id="baseline", status="archived", metrics={"metricsSchemaVersion": 2}, manifest=manifest("v2")))

    monkeypatch.setattr(main, "SessionLocal", sessions)
    monkeypatch.setattr(main, "initialize_database", lambda: None)
    main.app.dependency_overrides.clear()
    with TestClient(main.app) as client:
        assert client.get("/intelligence/overview").status_code == 401
        main.app.dependency_overrides[require_admin] = lambda: AdminSession("test", "csrf")
        overview = client.get("/intelligence/overview")
        assert overview.status_code == 200
        assert overview.json()["activeExperiment"]["id"] == "experiment"
        assert overview.json()["latestSample"]["metrics"]["claimRate"] == 0.4
        series = client.get("/intelligence/experiments/experiment/series")
        assert series.status_code == 200
        assert series.json()[0]["sample_kind"] == "interval"
        assert client.get("/intelligence/releases/missing").status_code == 404
        assert client.get("/intelligence/compare?candidate=candidate&baseline=baseline").status_code == 200
    main.app.dependency_overrides.clear()


def test_comparison_rejects_incompatible_observations(monkeypatch) -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    Base.metadata.create_all(engine)
    with sessions.begin() as db:
        db.add_all([
            Release(id="candidate-v2", status="candidate", metrics={}, manifest=manifest("v2")),
            Release(id="baseline-v1", status="archived", metrics={}, manifest=manifest("v1")),
        ])
    monkeypatch.setattr(main, "SessionLocal", sessions)
    monkeypatch.setattr(main, "initialize_database", lambda: None)
    main.app.dependency_overrides[require_admin] = lambda: AdminSession("test", "csrf")
    with TestClient(main.app) as client:
        response = client.get("/intelligence/compare?candidate=candidate-v2&baseline=baseline-v1")
        assert response.status_code == 409
    main.app.dependency_overrides.clear()


def test_legacy_checkpoint_can_be_continued_as_a_controlled_baseline(monkeypatch) -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    Base.metadata.create_all(engine)
    with sessions.begin() as db:
        db.add(Experiment(
            id="legacy", status="cancelled", checkpoint_key="experiments/legacy/checkpoint.pt.zst",
            config={"step_budget_per_seed": 1_000_000, "seed_count": 5, "master_seed": 42, "actor_count": 8, "replay_capacity_per_snake": 250_000, "batch_size": 128, "benchmark_matches": 200},
            progress={"environmentSteps": 614_338, "metricsSchemaVersion": 2},
        ))
    queued: list[tuple] = []
    class Queue:
        def enqueue(self, *args, **kwargs):
            queued.append((args, kwargs))
    monkeypatch.setattr(main, "SessionLocal", sessions)
    monkeypatch.setattr(main, "initialize_database", lambda: None)
    monkeypatch.setattr(main, "training_queue", lambda: Queue())
    main.app.dependency_overrides[require_csrf] = lambda: AdminSession("test", "csrf")
    with TestClient(main.app) as client:
        response = client.post("/legacy-baselines/legacy/continue", headers={"X-CSRF-Token": "csrf"})
        assert response.status_code == 201
        created = response.json()
        assert created["config"]["training_spec_version"] == 2
        assert created["config"]["baseline_mode"] is True
        assert created["config"]["resume_checkpoint_key"].endswith("checkpoint.pt.zst")
        assert created["config"]["seed_count"] == 1
        assert queued
    main.app.dependency_overrides.clear()


def test_comparison_accepts_the_known_v2_v3_contract_pair(monkeypatch) -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    Base.metadata.create_all(engine)
    with sessions.begin() as db:
        db.add_all([
            Release(id="candidate-v3", status="candidate", metrics={}, manifest=manifest("neon-serpents:v3:observation-228")),
            Release(id="baseline-v2", status="baseline", metrics={}, manifest=manifest("neon-serpents:v2:observation-159")),
        ])
    monkeypatch.setattr(main, "SessionLocal", sessions)
    monkeypatch.setattr(main, "initialize_database", lambda: None)
    main.app.dependency_overrides[require_admin] = lambda: AdminSession("test", "csrf")
    with TestClient(main.app) as client:
        assert client.get("/intelligence/compare?candidate=candidate-v3&baseline=baseline-v2").status_code == 200
    main.app.dependency_overrides.clear()
