from __future__ import annotations

from sqlalchemy import delete, select

from .artifacts import ArtifactStore
from .database import SessionLocal
from .models import AuditEvent, Experiment, Release, TrainingMetricSample


def purge_experiment(experiment_id: str, store: ArtifactStore | None = None) -> None:
    """Permanently remove one experiment and all of its database/object-store data."""
    with SessionLocal() as db:
        experiment = db.get(Experiment, experiment_id)
        release_ids = list(db.scalars(select(Release.id).where(Release.experiment_id == experiment_id)))
    if not experiment and not release_ids:
        return

    artifact_store = store or ArtifactStore()
    artifact_store.delete_prefix(f"experiments/{experiment_id}/")
    for release_id in release_ids:
        artifact_store.delete_prefix(f"releases/{release_id}/")

    with SessionLocal.begin() as db:
        subjects = [experiment_id, *release_ids]
        db.execute(delete(TrainingMetricSample).where(TrainingMetricSample.experiment_id == experiment_id))
        db.execute(delete(AuditEvent).where(AuditEvent.subject_id.in_(subjects)))
        db.execute(delete(Release).where(Release.experiment_id == experiment_id))
        db.execute(delete(Experiment).where(Experiment.id == experiment_id))
