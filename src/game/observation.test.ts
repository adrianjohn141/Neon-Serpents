import { describe, expect, it } from "vitest";
import { POWER_UP_KINDS, SNAKES } from "./constants";
import { createGame } from "./engine";
import { encodeObservation, OBSERVATION_LAYOUT, OBSERVATION_SIZE } from "./observation";

describe("version 3 observation contract", () => {
  it("encodes power-up kind, expiry, vector, spawn timing, and rival headings", () => {
    const state = createGame({ mode: "battle", snakes: SNAKES.slice(0, 3), seed: 7 });
    const snake = state.snakes[0];
    const prepared = {
      ...state,
      tick: 20,
      powerUp: { id: 9, kind: POWER_UP_KINDS[3], position: { x: snake.segments[0].x + 5, y: snake.segments[0].y }, expiresAt: 100 },
      snakes: state.snakes.map((entry, index) => index === 1 ? { ...entry, direction: "left" as const } : entry),
    };
    const observation = encodeObservation(prepared, prepared.snakes[0]);
    expect(observation).toHaveLength(OBSERVATION_SIZE);
    expect(observation[OBSERVATION_LAYOUT.powerUp.start]).toBe(1);
    expect(observation[OBSERVATION_LAYOUT.powerUp.start + 1 + 3]).toBe(1);
    expect(observation[OBSERVATION_LAYOUT.powerUp.start + 16]).toBe(0);
    expect(Array.from(observation.slice(OBSERVATION_LAYOUT.enemies.start + 6, OBSERVATION_LAYOUT.enemies.start + 14)).reduce((sum, value) => sum + value, 0)).toBe(2);
  });

  it("encodes a normalized next-spawn countdown only when no drop is active", () => {
    const state = createGame({ mode: "battle", snakes: SNAKES.slice(0, 2), seed: 9 });
    const observation = encodeObservation({ ...state, tick: 15, nextPowerUpAt: 45 }, state.snakes[0]);
    expect(observation[OBSERVATION_LAYOUT.powerUp.start + 16]).toBeCloseTo(30 / 180);
  });

  it("looks two cells ahead for danger while haste is active", () => {
    const state = createGame({ mode: "battle", snakes: SNAKES.slice(0, 2), seed: 11 });
    const baseSnake = {
      ...state.snakes[0],
      direction: "right" as const,
      segments: [{ x: 10, y: 10 }, { x: 9, y: 10 }],
    };
    const prepared = { ...state, tick: 5, obstacles: [{ x: 12, y: 10 }], snakes: [baseSnake, state.snakes[1]] };
    const normal = encodeObservation(prepared, baseSnake);
    const hastedSnake = { ...baseSnake, buffs: { ...baseSnake.buffs, hasteUntil: 20 } };
    const hasted = encodeObservation({ ...prepared, snakes: [hastedSnake, state.snakes[1]] }, hastedSnake);
    expect(normal[OBSERVATION_LAYOUT.danger.start]).toBe(0);
    expect(hasted[OBSERVATION_LAYOUT.danger.start]).toBe(1);
  });

  it("encodes phase, safe zone, objective, hazard, map, bounty, rare food, profiles, and series round", () => {
    const state = createGame({ mode: "battle", snakes: SNAKES.slice(0, 3), seed: 17, mapArchetype: "fortress", seriesRound: 2 });
    const snake = state.snakes[0];
    const prepared = {
      ...state,
      tick: 700,
      food: [{ id: 1, position: { x: snake.segments[0].x + 2, y: snake.segments[0].y }, kind: "rare" as const, value: 3 as const, spawnedAt: 600, expiresAt: 800 }],
      arena: {
        ...state.arena,
        phase: "midgame" as const,
        leaderId: state.snakes[1].id,
        objective: { id: 2, position: { x: 30, y: 20 }, radius: 2, expiresAt: 800, captureRequired: 45, progress: { [snake.id]: 12 }, contested: false },
        hazards: [{ id: 3, kind: "laser" as const, origin: { x: 1, y: 10 }, direction: "right" as const, length: 70, telegraphAt: 690, activatesAt: 710, activeUntil: 722 }],
        opponentProfiles: { [state.snakes[1].id]: { turnBias: .5, aggressionRate: .8, commonTarget: "leader" as const, powerUpRate: .6, typicalDeathCause: "headOn" as const, samples: 3 } },
      },
    };
    const observation = encodeObservation(prepared, prepared.snakes[0]);
    expect(observation).toHaveLength(228);
    expect(observation[OBSERVATION_LAYOUT.phase.start + 1]).toBe(1);
    expect(observation[OBSERVATION_LAYOUT.objective.start]).toBe(1);
    expect(observation[OBSERVATION_LAYOUT.hazard.start]).toBe(1);
    expect(observation[OBSERVATION_LAYOUT.map.start + 3]).toBe(1);
    expect(observation[OBSERVATION_LAYOUT.bounty.start]).toBe(0);
    expect(observation[OBSERVATION_LAYOUT.rareFood.start]).toBe(1);
    expect(Array.from(observation.slice(OBSERVATION_LAYOUT.opponentProfiles.start, OBSERVATION_LAYOUT.opponentProfiles.start + 13)).some((value) => value !== 0)).toBe(true);
    expect(Array.from(observation.slice(OBSERVATION_LAYOUT.seriesRound.start, OBSERVATION_LAYOUT.seriesRound.start + 3))).toEqual([0, 1, 0]);
  });
});
