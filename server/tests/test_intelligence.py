from __future__ import annotations

import base64
import json
from datetime import datetime, timezone

from server.app.intelligence import compare_releases, manifest_observation_hash, release_coverage, release_detail
from server.app.models import Release
from server.trainer.telemetry import sample_id, telemetry_interval
from server.trainer.run import aggregate_seed_results, baseline_bundle_suffix


def stats(*, wins: int, claims: int, opportunities: int, deaths: float) -> dict:
    return {
        "wins": wins,
        "avgScore": 12.0,
        "foodEaten": 20,
        "avgSurvivalTicks": 500.0,
        "powerUpsClaimed": claims,
        "powerUpOpportunities": opportunities,
        "approachWithoutClaims": 2,
        "deathsPerThousandTicks": deaths,
        "rewardBreakdown": {"powerUpClaim": claims * 30, "death": -120},
    }


def result(wins: int, claims: int, deaths: float) -> dict:
    return {
        "matches": 10,
        "pairedWinInterval": [0.02, 0.21],
        "scriptedWinRate": 0.6,
        "gates": {"positivePairedInterval": True},
        "perSnake": [
            {"snakeId": "nova", "candidate": stats(wins=wins, claims=claims, opportunities=10, deaths=deaths)},
            {"snakeId": "ember", "candidate": stats(wins=wins, claims=claims, opportunities=10, deaths=deaths)},
        ],
    }


def release(release_id: str, selected: dict | None, *, baseline_id: str | None = None, schema: int | None = 2) -> Release:
    metrics = {"seedResults": [selected] if selected else []}
    if schema is not None:
        metrics.update({"metricsSchemaVersion": schema, "selectedResult": selected, "baselineReleaseId": baseline_id, "eligible": True})
    payload = base64.b64encode(json.dumps({"observationSpecHash": "spec-v2"}).encode()).decode()
    return Release(id=release_id, status="candidate", metrics=metrics, manifest={"payload": payload}, created_at=datetime.now(timezone.utc))


def test_telemetry_cadence_and_resume_deduplication_key() -> None:
    assert telemetry_interval(1_000_000) == 5_000
    assert telemetry_interval(10_000) == 1_000
    first = sample_id("experiment", 2, 5_000, "interval")
    assert first == sample_id("experiment", 2, 5_000, "interval")
    assert first != sample_id("experiment", 2, 5_001, "interval")


def test_release_coverage_is_backward_compatible() -> None:
    full = release("full", result(6, 7, 1.0))
    legacy = release("legacy", result(4, 3, 1.2), schema=None)
    empty = release("empty", None, schema=None)
    assert release_coverage(full) == "full"
    assert release_coverage(legacy) == "partial"
    assert release_coverage(empty) == "unavailable"
    assert release_detail(legacy)["selectedResult"] is not None


def test_comparison_respects_metric_direction_and_statistical_pairing() -> None:
    baseline = release("baseline", result(4, 4, 1.3))
    candidate = release("candidate", result(6, 7, 1.0), baseline_id=baseline.id)
    comparison = compare_releases(candidate, baseline)
    by_key = {row["key"]: row for row in comparison["metrics"]}
    assert by_key["winRate"]["improved"] is True
    assert by_key["powerUpClaimRate"]["delta"] > 0
    assert by_key["deathsPerThousandTicks"]["improved"] is True
    assert comparison["statisticalComparisonAvailable"] is True
    assert comparison["pairedWinInterval"] == [0.02, 0.21]
    assert comparison["gates"]["positivePairedInterval"] is True
    assert manifest_observation_hash(candidate) == "spec-v2"


def test_arbitrary_release_pair_does_not_claim_statistical_significance() -> None:
    candidate = release("candidate", result(6, 7, 1.0), baseline_id="someone-else")
    baseline = release("baseline", result(4, 4, 1.3))
    comparison = compare_releases(candidate, baseline)
    assert comparison["metrics"]
    assert comparison["statisticalComparisonAvailable"] is False
    assert comparison["pairedWinInterval"] is None
    assert comparison["gates"] == {}


def test_five_seed_aggregation_uses_pooled_pairing_and_requires_consistency() -> None:
    report = {"pairedOutcomes": [1, 1, 0], "gates": {"positivePairedInterval": True, "survivalStable": True}, "adaptive": {"zoneRepositionRateCandidate": 0.9}}
    insufficient = aggregate_seed_results([report] * 4)
    assert insufficient["eligible"] is False
    assert insufficient["gates"]["seedConsistency"] is False

    complete = aggregate_seed_results([report] * 5)
    assert complete["pairedWinInterval"][0] > 0
    assert complete["gates"]["positivePairedInterval"] is True
    assert complete["gates"]["seedConsistency"] is True
    assert complete["adaptive"]["zoneRepositionRateCandidate"] == 0.9

    inconsistent = aggregate_seed_results([report] * 4 + [{**report, "gates": {"positivePairedInterval": True, "survivalStable": False}}])
    assert inconsistent["eligible"] is False
    assert inconsistent["gates"]["survivalStable"] is False


def test_baseline_contract_accepts_recoverable_v2_and_current_v3_only() -> None:
    assert baseline_bundle_suffix({"trainingSpecVersion": 2, "observationSize": 159, "observationSpecHash": "neon-serpents:v2:observation-159"}) == "v2"
    assert baseline_bundle_suffix({"trainingSpecVersion": 3, "observationSize": 228, "observationSpecHash": "neon-serpents:v3:observation-228"}) == "v3"
    assert baseline_bundle_suffix({"trainingSpecVersion": 2, "observationSize": 228, "observationSpecHash": "wrong"}) is None
