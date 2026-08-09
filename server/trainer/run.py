from __future__ import annotations

import json
import base64
import queue
import random
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import zstandard
from sqlalchemy import select

from server.app.artifacts import ArtifactStore, canonical_json, sign_manifest
from server.app.config import get_settings
from server.app.database import SessionLocal
from server.app.models import Experiment, Release
from server.app.purge import purge_experiment
from server.app.schemas import DEFAULT_SNAKE_ROSTER
from .grpc_service import serve
from .learner import LearnerPopulation
from .telemetry import TelemetrySampler


class TrainingControlRequest(Exception):
    def __init__(self, action: str) -> None:
        super().__init__(f"Training {action} requested")
        self.action = action


def _phase_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _advance_phase(progress: dict, phase: str, seed_index: int | None = None) -> dict:
    """Persist phase transitions and timing without relying on browser clocks."""
    updated = dict(progress or {})
    timing = dict(updated.get("timing") or {})
    now = datetime.now(timezone.utc)
    now_text = now.isoformat()
    previous_phase = timing.get("phase")
    previous_started = _phase_timestamp(timing.get("phaseStartedAt"))
    if previous_phase and previous_phase != phase and previous_started:
        elapsed = max(0.0, (now - previous_started).total_seconds())
        timing[f"{previous_phase}Seconds"] = float(timing.get(f"{previous_phase}Seconds", 0.0)) + elapsed
        if previous_phase == "benchmarking":
            timing["lastBenchmarkSeconds"] = elapsed
            timing["lastBenchmarkSeedIndex"] = timing.get("benchmarkSeedIndex")

    timing["phase"] = phase
    timing["phaseStartedAt"] = now_text
    if phase == "training":
        timing["trainingStartedAt"] = now_text
        timing["trainingSeedIndex"] = seed_index if seed_index is not None else updated.get("seedIndex", 0)
        if previous_phase == "paused":
            timing["resumeCount"] = int(timing.get("resumeCount", 0)) + 1
            timing["lastResumedAt"] = now_text
        timing["pauseStartedAt"] = None
    elif phase == "benchmarking":
        # This deliberately resets for every seed benchmark.
        timing["benchmarkStartedAt"] = now_text
        timing["benchmarkSeedIndex"] = seed_index if seed_index is not None else updated.get("seedIndex", 0)
        timing["benchmarkElapsedSeconds"] = 0.0
    elif phase == "paused":
        timing["pauseStartedAt"] = now_text
    elif phase not in {"preflight", "queued"}:
        timing["lastTransitionAt"] = now_text
    updated["phase"] = phase
    updated["timing"] = timing
    return updated


def bootstrap_interval(values: list[float], seed: int = 42, samples: int = 10_000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    size = len(values)
    estimates = sorted(sum(values[rng.randrange(size)] for _ in range(size)) / size for _ in range(samples))
    return [estimates[int(samples * 0.025)], estimates[int(samples * 0.975)]]


def aggregate_seed_results(reports: list[dict]) -> dict:
    paired_outcomes = [float(value) for report in reports for value in report.get("pairedOutcomes", [])]
    paired_interval = bootstrap_interval(paired_outcomes)
    gate_names = sorted({name for report in reports for name in report.get("gates", {}) if name != "positivePairedInterval"})
    gates = {
        name: len(reports) >= 5 and all(bool(report.get("gates", {}).get(name, False)) for report in reports)
        for name in gate_names
    }
    gates["positivePairedInterval"] = len(reports) >= 5 and paired_interval[0] > 0
    gates["seedConsistency"] = len(reports) >= 5 and all(bool(value) for value in gates.values())
    adaptive_keys = sorted({
        key for report in reports for key, value in report.get("adaptive", {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    })
    return {
        "pairedOutcomes": paired_outcomes,
        "pairedWinInterval": paired_interval,
        "gates": gates,
        "adaptive": {
            key: sum(float(report.get("adaptive", {}).get(key, 0.0)) for report in reports) / max(1, len(reports))
            for key in adaptive_keys
        },
        "eligible": bool(gates) and all(bool(value) for value in gates.values()),
    }


def baseline_bundle_suffix(manifest_payload: dict) -> str | None:
    contract = (
        manifest_payload.get("trainingSpecVersion"),
        manifest_payload.get("observationSize"),
        manifest_payload.get("observationSpecHash"),
    )
    if contract == (2, 159, "neon-serpents:v2:observation-159"):
        return "v2"
    if contract == (3, 228, "neon-serpents:v3:observation-228"):
        return "v3"
    return None


def set_experiment(experiment_id: str, **values) -> Experiment:
    with SessionLocal.begin() as db:
        experiment = db.get(Experiment, experiment_id)
        if not experiment:
            raise ValueError("Training experiment no longer exists.")
        current_progress = dict(experiment.progress or {})
        incoming_progress = values.get("progress")
        if isinstance(incoming_progress, dict):
            merged_progress = {**current_progress, **incoming_progress}
        else:
            merged_progress = current_progress
        status_phase = {
            "queued": "queued", "preflight": "preflight", "training": "training",
            "evaluating": "evaluating", "paused": "paused", "failed": "failed",
            "cancelled": "cancelled", "candidate": "completed", "completed": "completed",
        }.get(values.get("status"))
        requested_phase = (incoming_progress.get("phase") if isinstance(incoming_progress, dict) else None) or status_phase or current_progress.get("phase")
        current_phase = current_progress.get("phase") or (current_progress.get("timing") or {}).get("phase")
        if requested_phase and requested_phase != current_phase:
            merged_progress = _advance_phase(merged_progress, requested_phase, merged_progress.get("seedIndex"))
        elif isinstance(incoming_progress, dict) and current_progress.get("timing") and "timing" not in incoming_progress:
            merged_progress["timing"] = current_progress["timing"]
        if isinstance(incoming_progress, dict) or requested_phase:
            values["progress"] = merged_progress
        for key, value in values.items():
            setattr(experiment, key, value)
        return experiment


def request_for(experiment_id: str) -> str | None:
    with SessionLocal() as db:
        experiment = db.get(Experiment, experiment_id)
        return experiment.requested_action if experiment else "cancel"


def checkpoint(experiment_id: str, seed_index: int, population: LearnerPopulation, store: ArtifactStore) -> str:
    payload = {"seed_index": seed_index, "population": population.state_dict()}
    import io

    buffer = io.BytesIO()
    torch.save(payload, buffer)
    compressed = zstandard.ZstdCompressor(level=3).compress(buffer.getvalue())
    key = f"experiments/{experiment_id}/checkpoints/seed-{seed_index}-step-{population.environment_steps}.pt.zst"
    store.put_bytes(key, compressed, "application/zstd")
    set_experiment(experiment_id, checkpoint_key=key)
    return key


def export_population(population: LearnerPopulation, directory: Path, roster: list[dict] | None = None) -> None:
    roster = roster or [entry for entry in DEFAULT_SNAKE_ROSTER if entry["id"] in population.learners]
    source = {
        "brains": [
            {
                "snakeId": snake_id,
                "trainingSpecVersion": population.training_spec_version,
                "environmentSteps": learner.environment_steps,
                "learningSteps": learner.learning_steps,
                "episodes": learner.episodes,
                "epsilon": learner.epsilon,
                "scenarioSteps": learner.scenario_steps,
                "powerUpOpportunities": sum(item["opportunities"] for item in population.metrics["bySnake"][snake_id]["powerUps"].values()),
                "powerUpsClaimed": sum(item["claims"] for item in population.metrics["bySnake"][snake_id]["powerUps"].values()),
                "powerUpApproachMisses": sum(item["approachMisses"] for item in population.metrics["bySnake"][snake_id]["powerUps"].values()),
                "rareFoodClaims": population.metrics["bySnake"][snake_id]["rareFoodClaims"],
                "objectiveCaptures": population.metrics["bySnake"][snake_id]["objectiveCaptures"],
                "bountyKills": population.metrics["bySnake"][snake_id]["bountyKills"],
                "hazardDeaths": population.metrics["bySnake"][snake_id]["hazardDeaths"],
                "zoneDeaths": population.metrics["bySnake"][snake_id]["zoneDeaths"],
                "tensors": learner.wire_policy(population.policy_version)["tensors"],
            }
            for snake_id, learner in population.learners.items()
        ]
    }
    input_path = directory / "weights.json"
    input_path.write_text(json.dumps(source), encoding="utf-8")
    (directory / "roster.json").write_text(json.dumps(roster), encoding="utf-8")
    subprocess.run(["node", "/app/dist-server/server/bundle-exporter.js", str(input_path), str(directory)], check=True)


def download_baseline(directory: Path, store: ArtifactStore) -> str | None:
    with SessionLocal() as db:
        baseline_releases = db.scalars(select(Release).where(Release.status == "baseline").order_by(Release.created_at.desc())).all()
        release = next((item for item in baseline_releases if bool((item.metrics or {}).get("controlledBaseline"))), None)
        if not release:
            release = db.scalar(select(Release).where(Release.status == "production").order_by(Release.created_at.desc()).limit(1))
        if not release:
            return None
        manifest_payload = json.loads(base64.b64decode(release.manifest["payload"]))
        suffix = baseline_bundle_suffix(manifest_payload)
        if not suffix:
            return None
        for brain in manifest_payload.get("brains", []):
            body = store.get_object(brain["key"])["Body"].read()
            (directory / f"{brain['snakeId']}-{suffix}.nsbrain.json").write_bytes(body)
        if manifest_payload.get("roster"):
            (directory / "roster.json").write_text(json.dumps(manifest_payload["roster"]), encoding="utf-8")
        return release.id


def evaluate(experiment_id: str, candidate: Path, baseline: Path | None, output: Path, matches: int = 200) -> dict:
    process = subprocess.Popen([
        "node", "/app/dist-server/server/evaluate-release.js", str(candidate), str(baseline) if baseline else "-", str(output), str(matches),
    ])
    while process.poll() is None:
        action = request_for(experiment_id)
        if action in {"pause", "cancel"}:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise TrainingControlRequest(action)
        time.sleep(1)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, process.args)
    return json.loads(output.read_text(encoding="utf-8"))


def persist_seed_result(experiment_id: str, seed_index: int, directory: Path, score: float, result: dict, store: ArtifactStore) -> dict:
    prefix = f"experiments/{experiment_id}/seeds/{seed_index}"
    bundles = list(directory.glob("*-v3.nsbrain.json")) + list(directory.glob("*-v2.nsbrain.json"))
    for path in bundles:
        store.put_bytes(f"{prefix}/{path.name}", path.read_bytes(), "application/json")
    roster_path = directory / "roster.json"
    if roster_path.exists():
        store.put_bytes(f"{prefix}/roster.json", roster_path.read_bytes(), "application/json")
    suffix = "v2" if any(path.name.endswith("-v2.nsbrain.json") for path in bundles) else "v3"
    snake_ids = sorted(path.name.removesuffix(f"-{suffix}.nsbrain.json") for path in bundles)
    return {"seedIndex": seed_index, "score": score, "prefix": prefix, "suffix": suffix, "snakeIds": snake_ids, "evaluation": result}


def restore_seed_result(entry: dict, directory: Path, store: ArtifactStore) -> tuple[float, Path, dict]:
    directory.mkdir(parents=True, exist_ok=True)
    suffix = entry.get("suffix", "v3")
    for snake_id in entry.get("snakeIds", ("nova", "ember", "volt", "echo")):
        name = f"{snake_id}-{suffix}.nsbrain.json"
        (directory / name).write_bytes(store.get_object(f"{entry['prefix']}/{name}")["Body"].read())
    try:
        (directory / "roster.json").write_bytes(store.get_object(f"{entry['prefix']}/roster.json")["Body"].read())
    except Exception:
        pass
    return float(entry["score"]), directory, entry["evaluation"]


def create_release(experiment_id: str, directory: Path, metrics: dict, store: ArtifactStore, training_spec_version: int = 3, release_status: str = "candidate", expected_snake_ids: list[str] | None = None, roster: list[dict] | None = None) -> Release:
    suffix = "v2" if training_spec_version == 2 else "v3"
    bundle_paths = sorted(directory.glob(f"*-{suffix}.nsbrain.json"))
    snake_ids = {path.name.removesuffix(f"-{suffix}.nsbrain.json") for path in bundle_paths}
    expected = set(expected_snake_ids or ("nova", "ember", "volt", "echo"))
    if snake_ids != expected or len(bundle_paths) != len(expected):
        raise ValueError(f"A release requires exactly one validated {suffix} bundle for every snake.")
    with SessionLocal.begin() as db:
        release = Release(experiment_id=experiment_id, status=release_status, metrics=metrics, manifest={})
        db.add(release)
        db.flush()
        release_id = release.id
    brains = []
    for path in bundle_paths:
        snake_id = path.name.removesuffix(f"-{suffix}.nsbrain.json")
        data = path.read_bytes()
        key = f"releases/{release_id}/{path.name}"
        checksum = store.put_bytes(key, data, "application/json")
        brains.append({"snakeId": snake_id, "key": key, "sha256": checksum, "url": f"/server-api/production/bundles/{release_id}/{snake_id}"})
    payload = {
        "releaseId": release_id,
        "trainingSpecVersion": training_spec_version,
        "observationSize": 159 if training_spec_version == 2 else 228,
        "observationSpecHash": "neon-serpents:v2:observation-159" if training_spec_version == 2 else "neon-serpents:v3:observation-228",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "roster": roster or [entry for entry in DEFAULT_SNAKE_ROSTER if entry["id"] in expected],
        "brains": brains,
    }
    manifest = {
        "payload": base64.b64encode(canonical_json(payload)).decode(),
        "signature": sign_manifest(payload),
    }
    with SessionLocal.begin() as db:
        release = db.get(Release, release_id)
        release.manifest = manifest
        return release


def run_experiment(experiment_id: str) -> None:
    settings = get_settings()
    store = ArtifactStore(settings)
    server, bridge = serve(settings.trainer_grpc_bind)
    try:
        with SessionLocal() as db:
            existing = db.get(Experiment, experiment_id)
            initial_action = existing.requested_action if existing else "cancel"
        if initial_action == "cancel":
            purge_experiment(experiment_id, store)
            return
        experiment = set_experiment(
            experiment_id,
            status="preflight",
            started_at=datetime.now(timezone.utc),
            error=None,
            requested_action="finish" if initial_action == "finish" else None,
        )
        config = experiment.config
        training_spec_version = int(config.get("training_spec_version", 3))
        observation_size = 159 if training_spec_version == 2 else 228
        roster = list(config.get("roster") or DEFAULT_SNAKE_ROSTER)
        snake_ids = [str(entry["id"]) for entry in roster]
        if len(snake_ids) < 2 or len(snake_ids) > 8 or len(set(snake_ids)) != len(snake_ids):
            raise ValueError("Experiment roster must contain 2-8 unique snake IDs.")
        battle_size = max(2, min(len(snake_ids), int(config.get("battle_size", 4))))
        requested_actor_count = int(config.get("actor_count", 8))
        effective_actor_count = min(16, max(1, min(requested_actor_count, len(snake_ids) * 2)))
        baseline_mode = bool(config.get("baseline_mode", False))
        if baseline_mode and training_spec_version != 2:
            raise ValueError("Only a v2 experiment can be registered as the controlled legacy baseline.")
        seed_results: list[tuple[float, Path, dict]] = []
        forced_finish = False
        settings.trainer_cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"neon-{experiment_id}-", dir=settings.trainer_cache_dir) as temporary:
            workspace = Path(temporary)
            baseline_dir = workspace / "baseline"
            baseline_dir.mkdir()
            baseline_release_id = None if baseline_mode else download_baseline(baseline_dir, store)
            completed_seeds = list(experiment.progress.get("completedSeeds", []))
            for entry in completed_seeds:
                seed_results.append(restore_seed_result(entry, workspace / f"candidate-{entry['seedIndex']}", store))
            resume_payload = None
            resume_key = experiment.checkpoint_key or config.get("resume_checkpoint_key")
            if resume_key:
                import io

                compressed = store.get_object(resume_key)["Body"].read()
                resume_payload = torch.load(io.BytesIO(zstandard.ZstdDecompressor().decompress(compressed)), map_location="cpu", weights_only=False)
            start_seed_index = int(resume_payload["seed_index"]) if resume_payload else (max((int(item["seedIndex"]) for item in completed_seeds), default=-1) + 1)
            for seed_index in range(start_seed_index, config["seed_count"]):
                seed = (config["master_seed"] + seed_index * 104_729) & 0xFFFFFFFF
                population = LearnerPopulation(
                    config["replay_capacity_per_snake"], config["batch_size"], seed,
                    observation_size=observation_size,
                    training_spec_version=training_spec_version,
                    snake_ids=snake_ids,
                )
                if resume_payload and seed_index == start_seed_index:
                    population.load_state_dict(resume_payload["population"])
                    resume_payload = None
                telemetry = TelemetrySampler(experiment_id, seed_index, config["step_budget_per_seed"], population)
                telemetry.record("seed_start")
                bridge.start_job(
                    experiment_id, effective_actor_count, config["step_budget_per_seed"], seed, population.policies(),
                    training_spec_version=training_spec_version,
                    observation_size=observation_size,
                    roster=roster,
                    battle_size=battle_size,
                )
                if not bridge.connected.wait(timeout=180):
                    raise TimeoutError("No TypeScript simulation actors connected within 180 seconds. Check the actors service logs and LEARNER_GRPC_TARGET.")
                set_experiment(experiment_id, status="training", progress={"seedIndex": seed_index})
                seen_sequences: set[tuple[str, int]] = set()
                last_checkpoint = time.monotonic()
                last_batch_at = time.monotonic()
                next_checkpoint_step = population.environment_steps + 50_000
                last_policy_step = 0
                while population.environment_steps < config["step_budget_per_seed"]:
                    action = request_for(experiment_id)
                    if action in {"pause", "cancel", "finish"}:
                        bridge.stop_job(action)
                        if action == "cancel":
                            purge_experiment(experiment_id, store)
                            return
                        checkpoint(experiment_id, seed_index, population, store)
                        telemetry.record("checkpoint")
                        if action == "finish":
                            forced_finish = True
                            set_experiment(experiment_id, status="evaluating", requested_action=None, progress={
                                **(dict(experiment.progress) if experiment.progress else {}),
                                "phase": "finishing",
                                "environmentSteps": population.environment_steps,
                            })
                            break
                        set_experiment(
                            experiment_id,
                            status="paused" if action == "pause" else "cancelled",
                            requested_action=None,
                            finished_at=datetime.now(timezone.utc) if action == "cancel" else None,
                        )
                        return
                    try:
                        batch = bridge.incoming.get(timeout=5)
                    except queue.Empty:
                        if time.monotonic() - last_batch_at > 60:
                            raise TimeoutError("TypeScript actors connected but produced no transition batches for 60 seconds. Check actor worker errors and the observation contract.")
                        continue
                    last_batch_at = time.monotonic()
                    sequence = int(batch.sequence)
                    batch_key = (batch.actor_id, sequence)
                    if batch_key not in seen_sequences:
                        for transition in batch.transitions:
                            population.ingest(transition, batch.actor_id)
                            if population.environment_steps >= config["step_budget_per_seed"]:
                                break
                        seen_sequences.add(batch_key)
                    bridge.acknowledge(sequence)
                    learning_step = sum(item.learning_steps for item in population.learners.values())
                    if learning_step - last_policy_step >= 500:
                        population.policy_version += 1
                        bridge.publish_policies(population.policies())
                        last_policy_step = learning_step
                    progress = {
                        "metricsSchemaVersion": 3,
                        "phase": "training",
                        "seedIndex": seed_index,
                        "seedCount": config["seed_count"],
                        "environmentSteps": population.environment_steps,
                        "stepBudget": config["step_budget_per_seed"],
                        "policyVersion": population.policy_version,
                        "roster": roster,
                        "rosterSize": len(snake_ids),
                        "effectiveActorCount": effective_actor_count,
                        "replayCapacityTotal": int(config["replay_capacity_per_snake"]) * len(snake_ids),
                        "losses": {key: value.last_loss for key, value in population.learners.items()},
                        **population.metrics,
                        "completedSeeds": completed_seeds,
                    }
                    set_experiment(experiment_id, progress=progress)
                    if telemetry.due():
                        telemetry.record("interval")
                    if population.environment_steps >= next_checkpoint_step or time.monotonic() - last_checkpoint >= 300:
                        checkpoint(experiment_id, seed_index, population, store)
                        telemetry.record("checkpoint")
                        next_checkpoint_step = population.environment_steps + 50_000
                        last_checkpoint = time.monotonic()
                bridge.stop_job("seed-complete")
                telemetry.record("seed_end")
                candidate_dir = workspace / f"candidate-{seed_index}"
                candidate_dir.mkdir()
                export_population(population, candidate_dir, roster)
                with SessionLocal() as db:
                    current = db.get(Experiment, experiment_id)
                    benchmark_progress = dict(current.progress) if current else {}
                set_experiment(experiment_id, progress={**benchmark_progress, "phase": "benchmarking", "completedSeeds": completed_seeds})
                pending_action = request_for(experiment_id)
                if pending_action in {"pause", "cancel"}:
                    if pending_action == "cancel":
                        purge_experiment(experiment_id, store)
                        return
                    checkpoint(experiment_id, seed_index, population, store)
                    telemetry.record("checkpoint")
                    set_experiment(
                        experiment_id,
                        status="paused" if pending_action == "pause" else "cancelled",
                        requested_action=None,
                        finished_at=datetime.now(timezone.utc) if pending_action == "cancel" else None,
                    )
                    return
                if pending_action == "finish":
                    forced_finish = True
                    set_experiment(experiment_id, status="evaluating", requested_action=None)
                result = evaluate(experiment_id, candidate_dir, baseline_dir if baseline_release_id else None, workspace / f"evaluation-{seed_index}.json", config.get("benchmark_matches", 200))
                score = sum(row.get("validationWinRate", 0) for row in result["perSnake"]) / max(1, len(result["perSnake"]))
                seed_results.append((score, candidate_dir, result))
                completed_seeds.append(persist_seed_result(experiment_id, seed_index, candidate_dir, score, result, store))
                with SessionLocal() as db:
                    current = db.get(Experiment, experiment_id)
                    current_progress = dict(current.progress) if current else {}
                set_experiment(experiment_id, progress={**current_progress, "completedSeeds": completed_seeds}, checkpoint_key=None)
                if forced_finish:
                    break
            set_experiment(experiment_id, status="evaluating")
            selected = max(seed_results, key=lambda item: item[0])
            reports = [item[2] for item in seed_results]
            aggregate = aggregate_seed_results(reports)
            all_eligible = aggregate["eligible"] if not baseline_mode and not forced_finish else False
            metrics = {
                "metricsSchemaVersion": 3,
                "coverage": "partial" if forced_finish or len(seed_results) < int(config["seed_count"]) else "full",
                "eligible": all_eligible,
                "forcedFinish": forced_finish,
                "selectedValidationWinRate": selected[0],
                "selectedSeedIndex": int(selected[1].name.rsplit("-", 1)[-1]),
                "selectedResult": selected[2],
                "baselineReleaseId": baseline_release_id,
                "trainingSpecVersion": training_spec_version,
                "controlledBaseline": baseline_mode and len(seed_results) == int(config["seed_count"]),
                "stepBudgetPerSeed": int(config["step_budget_per_seed"]),
                "seedPassCount": sum(1 for report in reports if report.get("eligible", False)),
                "seedCount": len(seed_results),
                "requestedSeedCount": int(config["seed_count"]),
                "pairedWinInterval": aggregate["pairedWinInterval"],
                "pairedOutcomeCount": len(aggregate["pairedOutcomes"]),
                "adaptive": aggregate["adaptive"],
                "gates": aggregate["gates"],
                "seedResults": reports,
                "roster": roster,
                "rosterSize": len(snake_ids),
                "effectiveActorCount": effective_actor_count,
                "replayCapacityTotal": int(config["replay_capacity_per_snake"]) * len(snake_ids),
            }
            create_release(
                experiment_id,
                selected[1],
                metrics,
                store,
                training_spec_version=training_spec_version,
                release_status="baseline" if baseline_mode else "candidate",
                expected_snake_ids=snake_ids,
                roster=roster,
            )
            with SessionLocal() as db:
                current = db.get(Experiment, experiment_id)
                final_progress = dict(current.progress) if current else {}
            set_experiment(
                experiment_id,
                status="completed" if baseline_mode else "candidate",
                progress={**final_progress, "evaluation": metrics},
                finished_at=datetime.now(timezone.utc),
            )
    except TrainingControlRequest as control:
        if control.action == "cancel":
            purge_experiment(experiment_id, store)
            return
        set_experiment(
            experiment_id,
            status="paused" if control.action == "pause" else "cancelled",
            requested_action=None,
            finished_at=datetime.now(timezone.utc) if control.action == "cancel" else None,
        )
        return
    except Exception as error:
        set_experiment(experiment_id, status="failed", error=str(error), finished_at=datetime.now(timezone.utc))
        raise
    finally:
        bridge.stop_job("trainer-stopped")
        server.stop(grace=5)
