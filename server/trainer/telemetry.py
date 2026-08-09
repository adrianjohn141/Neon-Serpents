from __future__ import annotations

import math
import time
import uuid

from server.app.database import SessionLocal
from server.app.models import TrainingMetricSample
from .learner import LearnerPopulation


def telemetry_interval(step_budget: int) -> int:
    return max(1_000, math.ceil(step_budget / 200 / 1_000) * 1_000)


def sample_id(experiment_id: str, seed_index: int, environment_steps: int, sample_kind: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"neon-serpents:{experiment_id}:{seed_index}:{environment_steps}:{sample_kind}"))


class TelemetrySampler:
    def __init__(self, experiment_id: str, seed_index: int, step_budget: int, population: LearnerPopulation) -> None:
        self.experiment_id = experiment_id
        self.seed_index = seed_index
        self.step_budget = step_budget
        self.population = population
        self.interval = telemetry_interval(step_budget)
        self.next_step = (population.environment_steps // self.interval + 1) * self.interval
        self.last_step = population.environment_steps
        self.last_time = time.monotonic()

    def due(self) -> bool:
        return self.population.environment_steps >= self.next_step

    def record(self, sample_kind: str) -> TrainingMetricSample:
        now = time.monotonic()
        elapsed = max(now - self.last_time, 1e-6)
        advanced = max(0, self.population.environment_steps - self.last_step)
        throughput = advanced / elapsed
        metrics = {
            "metricsSchemaVersion": 3,
            "stepBudget": self.step_budget,
            "throughputStepsPerSecond": throughput,
            "etaSeconds": (self.step_budget - self.population.environment_steps) / throughput if throughput > 0 else None,
            "opportunities": self.population.metrics["opportunities"],
            "claims": self.population.metrics["claims"],
            "approachMisses": self.population.metrics["approachMisses"],
            "claimRate": self.population.metrics["claims"] / self.population.metrics["opportunities"] if self.population.metrics["opportunities"] else 0,
            "bySnake": {
                snake_id: {
                    "environmentSteps": learner.environment_steps,
                    "learningSteps": learner.learning_steps,
                    "loss": learner.last_loss,
                    "epsilon": learner.epsilon,
                    "scenarioSteps": learner.scenario_steps,
                    **self.population.metrics["bySnake"][snake_id],
                }
                for snake_id, learner in self.population.learners.items()
            },
        }
        sample = TrainingMetricSample(
            id=sample_id(self.experiment_id, self.seed_index, self.population.environment_steps, sample_kind),
            experiment_id=self.experiment_id,
            seed_index=self.seed_index,
            environment_steps=self.population.environment_steps,
            policy_version=self.population.policy_version,
            sample_kind=sample_kind,
            metrics=metrics,
        )
        with SessionLocal.begin() as db:
            sample = db.merge(sample)
        self.last_step = self.population.environment_steps
        self.last_time = now
        while self.next_step <= self.population.environment_steps:
            self.next_step += self.interval
        return sample
