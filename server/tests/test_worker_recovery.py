from __future__ import annotations

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from server.app.database import Base
from server.app.models import AuditEvent, Experiment
from server.trainer import worker_entry


def test_worker_restart_pauses_checkpointed_and_fails_uncheckpointed(monkeypatch) -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)
    with sessions.begin() as db:
        db.add_all([
            Experiment(status="training", config={}, checkpoint_key="checkpoints/saved.pt"),
            Experiment(status="preflight", config={}),
        ])
    monkeypatch.setattr(worker_entry, "initialize_database", lambda: None)
    monkeypatch.setattr(worker_entry, "SessionLocal", sessions)
    worker_entry.reconcile_interrupted_experiments()
    with sessions() as db:
        experiments = list(db.scalars(select(Experiment).order_by(Experiment.checkpoint_key.desc())))
        assert {experiment.status for experiment in experiments} == {"paused", "failed"}
        assert next(item for item in experiments if item.status == "paused").finished_at is None
        assert next(item for item in experiments if item.status == "failed").finished_at is not None
        assert len(list(db.scalars(select(AuditEvent)))) == 2
