from __future__ import annotations

import math

import numpy as np
import pytest

from server.trainer.learner import LearnerPopulation, OBSERVATION_SIZE, PrioritizedReplay, RawTransition, SnakeLearner


def observation(value: float) -> np.ndarray:
    return np.full(OBSERVATION_SIZE, value, dtype=np.float32)


def test_three_step_return_and_terminal_flush() -> None:
    learner = SnakeLearner("nova", 32, 16, 42)
    for index in range(3):
        learner.remember("episode", RawTransition(observation(index), 1, 1.0, observation(index + 1), index == 2))
    assert learner.replay.size == 3
    assert math.isclose(float(learner.replay.rewards[0]), 1 + 0.95 + 0.95**2, rel_tol=1e-6)
    assert list(learner.replay.steps[:3]) == [3, 2, 1]
    assert learner.replay.terminals[:3].all()


def test_prioritized_replay_weights_are_finite_and_bounded() -> None:
    replay = PrioritizedReplay(32)
    for index in range(16):
        replay.add(observation(index), index % 3, float(index), observation(index + 1), False, 3)
    replay.update(np.arange(16), np.linspace(0.01, 4, 16, dtype=np.float32))
    indices, weights = replay.sample(8, 0.7, np.random.default_rng(7))
    assert len(set(indices.tolist())) == 8
    assert np.isfinite(weights).all()
    assert (weights > 0).all() and (weights <= 1).all()


def test_scenario_counters_survive_state_restore() -> None:
    learner = SnakeLearner("nova", 32, 16, 42)
    learner.environment_steps = 17
    learner.scenario_steps.update({"survival": 5, "powerup": 7, "battle": 5, "safezone": 1, "hazard": 2, "objective": 3, "series": 4})
    wire = learner.wire_policy(3)
    assert wire["environmentSteps"] == 17
    assert wire["scenarioSteps"]["powerup"] == 7
    assert wire["scenarioSteps"]["objective"] == 3


def test_legacy_population_restores_the_159_input_contract_only() -> None:
    legacy = LearnerPopulation(32, 16, 42, observation_size=159, training_spec_version=2)
    state = legacy.state_dict()
    restored = LearnerPopulation(32, 16, 42, observation_size=159, training_spec_version=2)
    restored.load_state_dict(state)
    assert restored.learners["nova"].online.fc1.weight.shape == (128, 159)
    assert restored.training_spec_version == 2
    with pytest.raises(ValueError, match="incompatible"):
        LearnerPopulation(32, 16, 42).load_state_dict(state)


def test_population_uses_custom_roster_for_learners_metrics_and_restore() -> None:
    roster = ("nova", "ember", "volt", "echo", "comet")
    population = LearnerPopulation(32, 16, 42, snake_ids=roster)
    assert tuple(population.learners) == roster
    assert set(population.metrics["bySnake"]) == set(roster)
    restored = LearnerPopulation(32, 16, 42, snake_ids=roster)
    restored.load_state_dict(population.state_dict())
    assert tuple(restored.learners) == roster
    with pytest.raises(ValueError, match="roster"):
        LearnerPopulation(32, 16, 42).load_state_dict(population.state_dict())
