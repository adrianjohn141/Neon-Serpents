from __future__ import annotations

from datetime import datetime, timedelta, timezone

from server.trainer.run import _advance_phase


def test_benchmark_timer_resets_for_each_seed_and_closes_previous_phase() -> None:
    started = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    training = {"phase": "training", "seedIndex": 0, "timing": {"phase": "training", "phaseStartedAt": started}}

    first = _advance_phase(training, "benchmarking", seed_index=0)

    assert first["phase"] == "benchmarking"
    assert first["timing"]["trainingSeconds"] >= 9
    assert first["timing"]["benchmarkElapsedSeconds"] == 0.0
    assert first["timing"]["benchmarkSeedIndex"] == 0

    next_training = _advance_phase(first, "training", seed_index=1)
    second = _advance_phase(next_training, "benchmarking", seed_index=1)

    assert second["timing"]["benchmarkElapsedSeconds"] == 0.0
    assert second["timing"]["benchmarkSeedIndex"] == 1
    assert second["timing"]["lastBenchmarkSeedIndex"] == 0
