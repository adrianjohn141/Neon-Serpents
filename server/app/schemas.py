from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


DEFAULT_SNAKE_ROSTER = [
    {"id": "nova", "name": "Nova Viper", "color": "#68f7c1", "accent": "#d7fff1"},
    {"id": "ember", "name": "Ember Fang", "color": "#ff6b7a", "accent": "#ffd7dc"},
    {"id": "volt", "name": "Volt Coil", "color": "#ffd166", "accent": "#fff1bd"},
    {"id": "echo", "name": "Echo Wyrm", "color": "#9d83ff", "accent": "#e5dcff"},
]


class SnakeRosterEntry(BaseModel):
    id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    name: str = Field(min_length=1, max_length=48)
    color: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")
    accent: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")


class LoginRequest(BaseModel):
    password: str = Field(min_length=12, max_length=256)


class ExperimentCreate(BaseModel):
    training_spec_version: Literal[3] = 3
    step_budget_per_seed: int = Field(default=1_000_000, ge=1_000, le=100_000_000)
    seed_count: int = Field(default=5, ge=1, le=5)
    master_seed: int = Field(default=42, ge=0, le=2**32 - 1)
    actor_count: int = Field(default=8, ge=1, le=16)
    replay_capacity_per_snake: int = Field(default=250_000, ge=10_000, le=1_000_000)
    batch_size: int = Field(default=128, ge=16, le=512)
    benchmark_matches: int = Field(default=200, ge=2, le=1_000)
    roster: list[SnakeRosterEntry] = Field(default_factory=lambda: [SnakeRosterEntry(**entry) for entry in DEFAULT_SNAKE_ROSTER], min_length=2, max_length=8)
    battle_size: int = Field(default=4, ge=2, le=8)

    @field_validator("roster")
    @classmethod
    def unique_roster_ids(cls, value: list[SnakeRosterEntry]) -> list[SnakeRosterEntry]:
        ids = [entry.id for entry in value]
        if len(ids) != len(set(ids)):
            raise ValueError("roster snake IDs must be unique")
        return value


class ExperimentView(BaseModel):
    id: str
    status: str
    requested_action: str | None
    config: dict
    progress: dict
    error: str | None
    checkpoint_key: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    model_config = {"from_attributes": True}


class ReleaseView(BaseModel):
    id: str
    experiment_id: str | None
    status: str
    metrics: dict
    manifest: dict
    created_at: datetime
    promoted_at: datetime | None

    model_config = {"from_attributes": True}


class ActionResponse(BaseModel):
    status: Literal["accepted"] = "accepted"


class TrainingMetricSampleView(BaseModel):
    id: str
    experiment_id: str
    seed_index: int
    environment_steps: int
    policy_version: int
    sample_kind: str
    metrics: dict
    created_at: datetime

    model_config = {"from_attributes": True}


class IntelligenceOverview(BaseModel):
    coverage: Literal["full", "partial", "unavailable"]
    activeExperiment: dict | None
    latestSample: TrainingMetricSampleView | None
    production: dict | None
    latestCandidate: dict | None
    releases: list[dict]
