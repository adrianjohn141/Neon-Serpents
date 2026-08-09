from __future__ import annotations

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from server.app.database import Base
from server.app.models import AuditEvent, Experiment, Release, TrainingMetricSample
from server.app import purge


class FakeStore:
    def __init__(self) -> None:
        self.prefixes: list[str] = []

    def delete_prefix(self, prefix: str) -> int:
        self.prefixes.append(prefix)
        return 1


def test_purge_removes_experiment_metrics_releases_and_audit(monkeypatch) -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    sessions = sessionmaker(engine, expire_on_commit=False)
    Base.metadata.create_all(engine)
    with sessions.begin() as db:
        db.add(Experiment(id="experiment", config={}, progress={}))
        db.add(Release(id="release", experiment_id="experiment", status="candidate", metrics={}, manifest={}))
        db.add(TrainingMetricSample(id="sample", experiment_id="experiment", seed_index=0, environment_steps=1, policy_version=1, sample_kind="seed_start", metrics={}))
        db.add_all([AuditEvent(id="audit-experiment", subject_id="experiment", action="experiment.created"), AuditEvent(id="audit-release", subject_id="release", action="release.created")])
    monkeypatch.setattr(purge, "SessionLocal", sessions)
    store = FakeStore()
    purge.purge_experiment("experiment", store)  # type: ignore[arg-type]
    with sessions() as db:
        assert db.get(Experiment, "experiment") is None
        assert db.get(Release, "release") is None
        assert list(db.scalars(select(TrainingMetricSample))) == []
        assert list(db.scalars(select(AuditEvent))) == []
    assert store.prefixes == ["experiments/experiment/", "releases/release/"]
