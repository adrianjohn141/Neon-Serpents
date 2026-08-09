from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import select

from server.app.database import SessionLocal, initialize_database
from server.app.models import AuditEvent, Experiment


def reconcile_interrupted_experiments() -> None:
    initialize_database()
    with SessionLocal.begin() as db:
        interrupted = list(db.scalars(select(Experiment).where(Experiment.status.in_(["preflight", "training", "evaluating"]))))
        for experiment in interrupted:
            if experiment.checkpoint_key:
                experiment.status = "paused"
                experiment.error = "Training worker restarted; resume from the saved checkpoint."
            else:
                experiment.status = "failed"
                experiment.error = "Training worker restarted before a recoverable checkpoint was saved."
                experiment.finished_at = datetime.now(timezone.utc)
            experiment.requested_action = None
            db.add(AuditEvent(action="experiment.worker_recovered", subject_id=experiment.id, detail={"status": experiment.status}))


if __name__ == "__main__":
    reconcile_interrupted_experiments()
    os.execvp("rq", ["rq", "worker", "training", "--url", os.environ.get("REDIS_URL", "redis://redis:6379/0")])
