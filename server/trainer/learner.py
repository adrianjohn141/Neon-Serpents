from __future__ import annotations

import io
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

OBSERVATION_SIZE = 228
ACTIONS = 3
DEFAULT_SNAKE_IDS = ("nova", "ember", "volt", "echo")
SNAKE_IDS = DEFAULT_SNAKE_IDS
REWARD_COMPONENTS = (
    "step", "survival", "food_approach", "food_claim", "power_up_approach", "power_up_claim",
    "zone_positioning", "objective_approach", "objective_capture", "bounty_kill", "rare_food_claim",
    "kill", "death", "win",
)
TRAINING_SCENARIOS = ("survival", "powerup", "safezone", "hazard", "objective", "battle", "series")
POWER_UP_KINDS = ("shield", "phase", "haste", "double", "magnet", "growth", "trim", "secondChance", "warp", "freeze", "crown", "vision")


class DqnNetwork(nn.Module):
    def __init__(self, observation_size: int = OBSERVATION_SIZE) -> None:
        super().__init__()
        self.fc1 = nn.Linear(observation_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, ACTIONS)
        for layer in (self.fc1, self.fc2):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.out(F.relu(self.fc2(F.relu(self.fc1(value)))))


class PrioritizedReplay:
    def __init__(self, capacity: int, alpha: float = 0.6, observation_size: int = OBSERVATION_SIZE) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.states = np.zeros((capacity, observation_size), dtype=np.float32)
        self.next_states = np.zeros((capacity, observation_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.uint8)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminals = np.zeros(capacity, dtype=np.bool_)
        self.steps = np.ones(capacity, dtype=np.uint8)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.cursor = 0
        self.max_priority = 1.0

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, terminal: bool, steps: int) -> None:
        index = self.cursor
        self.states[index] = state
        self.next_states[index] = next_state
        self.actions[index] = action
        self.rewards[index] = reward
        self.terminals[index] = terminal
        self.steps[index] = steps
        self.priorities[index] = self.max_priority
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, count: int, beta: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        priorities = np.power(np.maximum(self.priorities[: self.size], 1e-6), self.alpha)
        probabilities = priorities / priorities.sum()
        indices = rng.choice(self.size, size=min(count, self.size), replace=False, p=probabilities)
        weights = np.power(self.size * probabilities[indices], -beta)
        weights /= weights.max()
        return indices, weights.astype(np.float32)

    def update(self, indices: np.ndarray, errors: np.ndarray) -> None:
        values = np.abs(errors).astype(np.float32) + 1e-5
        self.priorities[indices] = values
        self.max_priority = max(self.max_priority, float(values.max(initial=1.0)))

    def state_dict(self) -> dict:
        used = self.size
        return {
            "capacity": self.capacity,
            "size": used,
            "cursor": self.cursor,
            "max_priority": self.max_priority,
            "states": self.states[:used],
            "next_states": self.next_states[:used],
            "actions": self.actions[:used],
            "rewards": self.rewards[:used],
            "terminals": self.terminals[:used],
            "steps": self.steps[:used],
            "priorities": self.priorities[:used],
        }

    def load_state_dict(self, value: dict) -> None:
        used = min(int(value["size"]), self.capacity)
        self.size = used
        self.cursor = int(value["cursor"]) % self.capacity
        self.max_priority = float(value["max_priority"])
        for name in ("states", "next_states", "actions", "rewards", "terminals", "steps", "priorities"):
            getattr(self, name)[:used] = value[name][:used]


@dataclass
class RawTransition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminal: bool


class SnakeLearner:
    def __init__(self, snake_id: str, capacity: int, batch_size: int, seed: int, observation_size: int = OBSERVATION_SIZE) -> None:
        self.snake_id = snake_id
        self.online = DqnNetwork(observation_size)
        self.target = DqnNetwork(observation_size)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=0.0005)
        self.replay = PrioritizedReplay(capacity, observation_size=observation_size)
        self.batch_size = batch_size
        self.learning_steps = 0
        self.environment_steps = 0
        self.episodes = 0
        self.scenario_steps = {scenario: 0 for scenario in TRAINING_SCENARIOS}
        self.last_loss: float | None = None
        self.rng = np.random.default_rng(seed)
        self.nstep: dict[str, deque[RawTransition]] = defaultdict(deque)

    @property
    def epsilon(self) -> float:
        progress = min(1.0, self.environment_steps / 250_000)
        return max(0.05, 1.0 + (0.10 - 1.0) * progress)

    def remember(self, episode_key: str, transition: RawTransition) -> None:
        queue = self.nstep[episode_key]
        queue.append(transition)
        self.environment_steps += 1
        if transition.terminal:
            self.episodes += 1
            while queue:
                self._emit(queue, min(3, len(queue)), True)
            self.nstep.pop(episode_key, None)
        elif len(queue) >= 3:
            self._emit(queue, 3, False)

    def _emit(self, queue: deque[RawTransition], count: int, forced_terminal: bool) -> None:
        window = list(queue)[:count]
        reward = sum((0.95**index) * item.reward for index, item in enumerate(window))
        last = window[-1]
        first = queue.popleft()
        self.replay.add(first.state, first.action, reward, last.next_state, forced_terminal or last.terminal, count)

    def flush_episode(self, episode_key: str) -> None:
        queue = self.nstep.pop(episode_key, deque())
        while queue:
            self._emit(queue, min(3, len(queue)), True)

    def train(self) -> float | None:
        if self.replay.size < max(1_000, self.batch_size) or self.environment_steps % 4:
            return None
        beta = 0.4 + 0.6 * min(1.0, self.learning_steps / 250_000)
        indices, importance = self.replay.sample(self.batch_size, beta, self.rng)
        states = torch.from_numpy(self.replay.states[indices])
        next_states = torch.from_numpy(self.replay.next_states[indices])
        actions = torch.from_numpy(self.replay.actions[indices].astype(np.int64))
        rewards = torch.from_numpy(self.replay.rewards[indices])
        terminals = torch.from_numpy(self.replay.terminals[indices].astype(np.float32))
        steps = torch.from_numpy(self.replay.steps[indices].astype(np.float32))
        weights = torch.from_numpy(importance)
        selected = self.online(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            best = self.online(next_states).argmax(1)
            future = self.target(next_states).gather(1, best[:, None]).squeeze(1)
            target = rewards + (1 - terminals) * torch.pow(torch.tensor(0.95), steps) * future
        errors = target - selected
        loss = (F.smooth_l1_loss(selected, target, reduction="none") * weights).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.replay.update(indices, errors.detach().numpy())
        self.learning_steps += 1
        if self.learning_steps % 1_500 == 0:
            self.target.load_state_dict(self.online.state_dict())
        self.last_loss = float(loss.item())
        return self.last_loss

    def wire_policy(self, version: int) -> dict:
        tensors = []
        for name, value in self.online.state_dict().items():
            array = value.detach().cpu().numpy().astype(np.float32)
            tensors.append({"name": name, "shape": list(array.shape), "values": array.ravel().tolist()})
        return {
            "snakeId": self.snake_id, "version": version, "epsilon": self.epsilon,
            "environmentSteps": self.environment_steps, "scenarioSteps": self.scenario_steps,
            "tensors": tensors,
        }


class LearnerPopulation:
    def __init__(self, capacity: int, batch_size: int, seed: int, observation_size: int = OBSERVATION_SIZE, training_spec_version: int = 3, snake_ids: Sequence[str] | None = None) -> None:
        torch.manual_seed(seed)
        random.seed(seed)
        if (training_spec_version, observation_size) not in {(2, 159), (3, 228)}:
            raise ValueError("Unsupported learner training contract.")
        self.observation_size = observation_size
        self.training_spec_version = training_spec_version
        self.snake_ids = tuple(snake_ids or DEFAULT_SNAKE_IDS)
        if len(self.snake_ids) < 2 or len(self.snake_ids) > 8 or len(set(self.snake_ids)) != len(self.snake_ids):
            raise ValueError("A learner population requires 2-8 unique snake IDs.")
        self.learners = {snake_id: SnakeLearner(snake_id, capacity, batch_size, seed + index * 7_919, observation_size) for index, snake_id in enumerate(self.snake_ids)}
        self.policy_version = 1
        self.environment_steps = 0
        self.metrics = {
            "opportunities": 0, "claims": 0, "approachMisses": 0,
            "bySnake": {
                snake_id: {
                    "reward": {component: 0.0 for component in REWARD_COMPONENTS},
                    "rewardTotal": 0.0,
                    "rareFoodClaims": 0, "objectiveCaptures": 0, "bountyKills": 0,
                    "hazardDeaths": 0, "zoneDeaths": 0,
                    "powerUps": {kind: {"opportunities": 0, "claims": 0, "approachMisses": 0} for kind in POWER_UP_KINDS},
                }
                for snake_id in self.snake_ids
            },
        }

    def policies(self) -> list[dict]:
        return [learner.wire_policy(self.policy_version) for learner in self.learners.values()]

    def ingest(self, transition: object, actor_id: str) -> None:
        learner = self.learners.get(transition.snake_id)
        if not learner:
            return
        state = np.frombuffer(transition.state, dtype="<f4").copy()
        next_state = np.frombuffer(transition.next_state, dtype="<f4").copy()
        if state.size != self.observation_size or next_state.size != self.observation_size or not np.isfinite(state).all() or not np.isfinite(next_state).all():
            raise ValueError("Actor transition contains an invalid observation tensor.")
        if not math.isfinite(transition.reward) or transition.action not in (0, 1, 2):
            raise ValueError("Actor transition contains an invalid action or reward.")
        key = f"{actor_id}:{transition.episode_id}"
        learner.remember(key, RawTransition(state, transition.action, transition.reward, next_state, transition.terminal))
        if transition.scenario not in learner.scenario_steps:
            raise ValueError("Actor transition contains an invalid training scenario.")
        learner.scenario_steps[transition.scenario] += 1
        learner.train()
        self.environment_steps += 1
        self.metrics["opportunities"] += int(transition.opportunity)
        self.metrics["claims"] += int(transition.claimed)
        self.metrics["approachMisses"] += int(transition.approach_miss)
        snake_metrics = self.metrics["bySnake"][transition.snake_id]
        snake_metrics["rewardTotal"] += float(transition.reward)
        snake_metrics["rareFoodClaims"] += int(transition.rare_food_claimed)
        snake_metrics["objectiveCaptures"] += int(transition.objective_captured)
        snake_metrics["bountyKills"] += int(transition.bounty_kill)
        snake_metrics["hazardDeaths"] += int(transition.death_cause == "hazard")
        snake_metrics["zoneDeaths"] += int(transition.death_cause == "zone")
        components = transition.reward_components
        for component in REWARD_COMPONENTS:
            snake_metrics["reward"][component] += float(getattr(components, component, 0.0))
        if transition.power_up_kind:
            if transition.power_up_kind not in POWER_UP_KINDS:
                raise ValueError("Actor transition contains an invalid power-up kind.")
            kind_metrics = snake_metrics["powerUps"][transition.power_up_kind]
            kind_metrics["opportunities"] += int(transition.opportunity)
            kind_metrics["claims"] += int(transition.claimed)
            kind_metrics["approachMisses"] += int(transition.approach_miss)

    def state_dict(self) -> dict:
        return {
            "observation_size": self.observation_size,
            "training_spec_version": self.training_spec_version,
            "snake_ids": list(self.snake_ids),
            "policy_version": self.policy_version,
            "environment_steps": self.environment_steps,
            "metrics": self.metrics,
            "torch_rng": torch.get_rng_state(),
            "learners": {
                snake_id: {
                    "online": learner.online.state_dict(),
                    "target": learner.target.state_dict(),
                    "optimizer": learner.optimizer.state_dict(),
                    "replay": learner.replay.state_dict(),
                    "learning_steps": learner.learning_steps,
                    "environment_steps": learner.environment_steps,
                    "episodes": learner.episodes,
                    "scenario_steps": learner.scenario_steps,
                    "last_loss": learner.last_loss,
                    "rng": learner.rng.bit_generator.state,
                }
                for snake_id, learner in self.learners.items()
            },
        }

    def load_state_dict(self, value: dict) -> None:
        stored_observation_size = int(value.get("observation_size", next(iter(value["learners"].values()))["online"]["fc1.weight"].shape[1]))
        stored_spec = int(value.get("training_spec_version", 2 if stored_observation_size == 159 else 3))
        if stored_observation_size != self.observation_size or stored_spec != self.training_spec_version:
            raise ValueError("Checkpoint training contract is incompatible with this experiment.")
        stored_snake_ids = tuple(value.get("snake_ids") or value.get("learners", {}).keys() or DEFAULT_SNAKE_IDS)
        if stored_snake_ids != self.snake_ids:
            raise ValueError("Checkpoint snake roster is incompatible with this experiment.")
        self.policy_version = int(value["policy_version"])
        self.environment_steps = int(value["environment_steps"])
        self.metrics = value["metrics"]
        if "bySnake" not in self.metrics:
            self.metrics["bySnake"] = {
                snake_id: {
                    "reward": {component: 0.0 for component in REWARD_COMPONENTS}, "rewardTotal": 0.0,
                    "rareFoodClaims": 0, "objectiveCaptures": 0, "bountyKills": 0,
                    "hazardDeaths": 0, "zoneDeaths": 0,
                    "powerUps": {kind: {"opportunities": 0, "claims": 0, "approachMisses": 0} for kind in POWER_UP_KINDS},
                }
                for snake_id in self.snake_ids
            }
        for snake_id in self.snake_ids:
            row = self.metrics["bySnake"].setdefault(snake_id, {})
            reward = row.setdefault("reward", {})
            for component in REWARD_COMPONENTS:
                reward.setdefault(component, 0.0)
            row.setdefault("rewardTotal", 0.0)
            row.setdefault("rareFoodClaims", 0)
            row.setdefault("objectiveCaptures", 0)
            row.setdefault("bountyKills", 0)
            row.setdefault("hazardDeaths", 0)
            row.setdefault("zoneDeaths", 0)
            powerups = row.setdefault("powerUps", {})
            for kind in POWER_UP_KINDS:
                powerups.setdefault(kind, {"opportunities": 0, "claims": 0, "approachMisses": 0})
        torch.set_rng_state(value["torch_rng"])
        for snake_id, state in value["learners"].items():
            learner = self.learners[snake_id]
            learner.online.load_state_dict(state["online"])
            learner.target.load_state_dict(state["target"])
            learner.optimizer.load_state_dict(state["optimizer"])
            learner.replay.load_state_dict(state["replay"])
            learner.learning_steps = int(state["learning_steps"])
            learner.environment_steps = int(state["environment_steps"])
            learner.episodes = int(state.get("episodes", 0))
            restored_steps = state.get("scenario_steps", {})
            learner.scenario_steps = {scenario: int(restored_steps.get(scenario, 0)) for scenario in TRAINING_SCENARIOS}
            learner.last_loss = state["last_loss"]
            learner.rng.bit_generator.state = state["rng"]

    def serialize(self) -> bytes:
        buffer = io.BytesIO()
        torch.save(self.state_dict(), buffer)
        return buffer.getvalue()
