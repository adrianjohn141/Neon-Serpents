from __future__ import annotations

import base64
import json
from typing import Any, Literal

from .models import Release

Coverage = Literal["full", "partial", "unavailable"]


def release_coverage(release: Release) -> Coverage:
    metrics = release.metrics or {}
    if metrics.get("metricsSchemaVersion") in (2, 3) and metrics.get("selectedResult"):
        return "full"
    if metrics.get("seedResults"):
        return "partial"
    return "unavailable"


def selected_result(release: Release) -> dict | None:
    metrics = release.metrics or {}
    if isinstance(metrics.get("selectedResult"), dict):
        return metrics["selectedResult"]
    results = metrics.get("seedResults")
    if not isinstance(results, list) or not results:
        return None
    return max(results, key=lambda result: sum(float(row.get("validationWinRate", 0)) for row in result.get("perSnake", [])))


def release_summary(release: Release | None) -> dict | None:
    if not release:
        return None
    metrics = release.metrics or {}
    result = selected_result(release)
    aggregate = aggregate_result(result) if result else None
    return {
        "id": release.id,
        "status": release.status,
        "experimentId": release.experiment_id,
        "createdAt": release.created_at.isoformat(),
        "promotedAt": release.promoted_at.isoformat() if release.promoted_at else None,
        "coverage": release_coverage(release),
        "eligible": metrics.get("eligible"),
        "selectedSeedIndex": metrics.get("selectedSeedIndex"),
        "seedPassCount": metrics.get("seedPassCount"),
        "seedCount": metrics.get("seedCount"),
        "baselineReleaseId": metrics.get("baselineReleaseId"),
        "aggregate": aggregate,
    }


def aggregate_result(result: dict | None) -> dict | None:
    if not result:
        return None
    rows = [row for row in result.get("perSnake", []) if isinstance(row.get("candidate"), dict)]
    if not rows:
        return None
    matches = max(1, int(result.get("matches", 1)))
    candidates = [row["candidate"] for row in rows]
    opportunities = sum(float(row.get("powerUpOpportunities", 0)) for row in candidates)
    claims = sum(float(row.get("powerUpsClaimed", 0)) for row in candidates)
    misses = sum(float(row.get("approachWithoutClaims", row.get("approachWithoutClaimRate", 0) * row.get("powerUpOpportunities", 0))) for row in candidates)
    rare_food = sum(float(row.get("rareFoodClaims", 0)) for row in candidates)
    objectives = sum(float(row.get("objectiveCaptures", 0)) for row in candidates)
    bounties = sum(float(row.get("bountyKills", 0)) for row in candidates)
    hazard_deaths = sum(float(row.get("hazardDeaths", 0)) for row in candidates)
    zone_deaths = sum(float(row.get("zoneDeaths", 0)) for row in candidates)
    rewards: dict[str, float] = {}
    for candidate in candidates:
        for key, value in candidate.get("rewardBreakdown", {}).items():
            rewards[key] = rewards.get(key, 0.0) + float(value)
    return {
        "matches": matches,
        "winRate": sum(float(row.get("wins", 0)) for row in candidates) / (matches * len(candidates)),
        "avgScore": sum(float(row.get("avgScore", 0)) for row in candidates) / len(candidates),
        "foodPerMatch": sum(float(row.get("foodEaten", 0)) for row in candidates) / (matches * len(candidates)),
        "avgSurvivalTicks": sum(float(row.get("avgSurvivalTicks", 0)) for row in candidates) / len(candidates),
        "powerUpClaimRate": claims / opportunities if opportunities else 0,
        "approachWithoutClaimRate": misses / opportunities if opportunities else 0,
        "deathsPerThousandTicks": sum(float(row.get("deathsPerThousandTicks", 0)) for row in candidates) / len(candidates),
        "rareFoodPerMatch": rare_food / (matches * len(candidates)),
        "objectiveCapturesPerMatch": objectives / (matches * len(candidates)),
        "bountyKillsPerMatch": bounties / (matches * len(candidates)),
        "hazardDeathsPerMatch": hazard_deaths / (matches * len(candidates)),
        "zoneDeathsPerMatch": zone_deaths / (matches * len(candidates)),
        "scriptedWinRate": float(result.get("scriptedWinRate", sum(float(row.get("scriptedWinRate", 0)) for row in rows) / len(rows))),
        "rewardBreakdown": rewards,
        "pairedWinInterval": result.get("pairedWinInterval"),
        "gates": result.get("gates", {}),
    }


def release_detail(release: Release) -> dict:
    result = selected_result(release)
    return {
        "summary": release_summary(release),
        "metricsSchemaVersion": (release.metrics or {}).get("metricsSchemaVersion"),
        "coverage": release_coverage(release),
        "selectedResult": result,
        "seedResults": (release.metrics or {}).get("seedResults", []),
        "gates": {**((result or {}).get("gates", {})), **((release.metrics or {}).get("gates", {}))},
    }


def manifest_observation_hash(release: Release) -> str | None:
    try:
        payload = json.loads(base64.b64decode(release.manifest["payload"]))
        return payload.get("observationSpecHash")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


METRIC_DIRECTIONS = {
    "winRate": "higher",
    "avgScore": "higher",
    "foodPerMatch": "higher",
    "avgSurvivalTicks": "higher",
    "powerUpClaimRate": "higher",
    "approachWithoutClaimRate": "lower",
    "deathsPerThousandTicks": "lower",
    "scriptedWinRate": "higher",
    "rareFoodPerMatch": "higher",
    "objectiveCapturesPerMatch": "higher",
    "bountyKillsPerMatch": "higher",
    "hazardDeathsPerMatch": "lower",
    "zoneDeathsPerMatch": "lower",
}


def compare_releases(candidate: Release, baseline: Release) -> dict:
    candidate_detail = release_detail(candidate)
    baseline_detail = release_detail(baseline)
    candidate_aggregate = candidate_detail["summary"]["aggregate"]
    baseline_aggregate = baseline_detail["summary"]["aggregate"]
    metrics = []
    if candidate_aggregate and baseline_aggregate:
        for key, direction in METRIC_DIRECTIONS.items():
            left = candidate_aggregate.get(key)
            right = baseline_aggregate.get(key)
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                delta = left - right
                metrics.append({
                    "key": key, "candidate": left, "baseline": right, "delta": delta,
                    "direction": direction,
                    "improved": delta > 0 if direction == "higher" else delta < 0,
                })
    is_benchmark_baseline = (candidate.metrics or {}).get("baselineReleaseId") == baseline.id
    selected = selected_result(candidate) or {}
    gates = {**selected.get("gates", {}), **(candidate.metrics or {}).get("gates", {})} if is_benchmark_baseline else {}
    return {
        "candidate": candidate_detail,
        "baseline": baseline_detail,
        "compatible": True,
        "statisticalComparisonAvailable": is_benchmark_baseline,
        "metrics": metrics,
        "pairedWinInterval": selected.get("pairedWinInterval") if is_benchmark_baseline else None,
        "gates": gates,
    }
