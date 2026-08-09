import { describe, expect, it } from "vitest";
import { POWER_UP_KINDS, SNAKES } from "./constants";
import { createGame } from "./engine";
import {
  AdaptiveBehaviorTracker, emptyAdaptiveBehavior, emptyDeathCauses, emptyPowerUpBehavior,
  finalizePowerUpBehavior, normalizeDeathCause, PowerUpBehaviorTracker,
} from "./evaluation-intelligence";
import type { GameState } from "./types";

function stateFor(kind: (typeof POWER_UP_KINDS)[number], distance = 5, lifetime = 20): GameState {
  const base = createGame({ mode: "powerup", snakes: [SNAKES[0]], seed: 91 });
  const snake = { ...base.snakes[0], segments: [{ x: 5, y: 5 }, { x: 4, y: 5 }], direction: "right" as const };
  return { ...base, tick: 10, snakes: [snake], obstacles: [], food: [], powerUp: { id: 99, kind, position: { x: 5 + distance, y: 5 }, expiresAt: 10 + lifetime } };
}

describe("evaluation intelligence", () => {
  it("records a reachable claim for every powerup kind", () => {
    for (const kind of POWER_UP_KINDS) {
      const metrics = { nova: emptyPowerUpBehavior() };
      const tracker = new PowerUpBehaviorTracker(metrics);
      const before = stateFor(kind);
      const after = { ...before, tick: 15, powerUp: null, snakes: [{ ...before.snakes[0], powerUps: 1, segments: [{ x: 10, y: 5 }] }] };
      tracker.observe(before, after);
      const result = finalizePowerUpBehavior(metrics.nova)[kind];
      expect(result).toMatchObject({ seen: 1, reachable: 1, pursued: 1, claimed: 1, ignored: 0, pursuitWithoutClaim: 0 });
      expect(result.avgClaimTicks).toBe(5);
    }
  });

  it("distinguishes intelligent avoidance, pursuit misses, and pursuit deaths", () => {
    const metrics = { nova: emptyPowerUpBehavior() };
    const ignored = new PowerUpBehaviorTracker(metrics);
    const unreachable = stateFor("shield", 5, 2);
    ignored.observe(unreachable, { ...unreachable, tick: 12, powerUp: null });
    expect(finalizePowerUpBehavior(metrics.nova).shield).toMatchObject({ seen: 1, reachable: 0, ignored: 0 });

    const pursuit = new PowerUpBehaviorTracker(metrics);
    const before = stateFor("shield");
    const closer = { ...before, tick: 11, snakes: [{ ...before.snakes[0], segments: [{ x: 7, y: 5 }] }] };
    pursuit.observe(before, closer);
    pursuit.observe(closer, { ...closer, tick: 12, powerUp: null, snakes: [{ ...closer.snakes[0], alive: false, deathReason: "hit wall" }] });
    expect(finalizePowerUpBehavior(metrics.nova).shield).toMatchObject({ pursued: 1, pursuitWithoutClaim: 1, pursuitDeaths: 1 });
  });

  it("normalizes every death category", () => {
    expect(normalizeDeathCause("hit wall")).toBe("wall");
    expect(normalizeDeathCause("hit obstacle")).toBe("obstacle");
    expect(normalizeDeathCause("head-on collision")).toBe("headOn");
    expect(normalizeDeathCause("hit another snake")).toBe("snakeBody");
    expect(normalizeDeathCause("timeout")).toBe("other");
    expect(emptyDeathCauses()).toEqual({ wall: 0, obstacle: 0, headOn: 0, snakeBody: 0, other: 0 });
  });

  it("measures warned zone repositioning and hazard counter use", () => {
    const base = createGame({ mode: "battle", snakes: SNAKES.slice(0, 2), seed: 92, mapArchetype: "open" });
    const metrics = Object.fromEntries(base.snakes.map((snake) => [snake.id, emptyAdaptiveBehavior()]));
    const tracker = new AdaptiveBehaviorTracker(metrics);
    const exposed = { ...base.snakes[0], segments: [{ x: 0, y: 10 }, { x: 1, y: 10 }] };
    const warnedBefore = {
      ...base, tick: 1_199, snakes: [exposed, base.snakes[1]],
      arena: { ...base.arena, phase: "endgame" as const, safeZone: { ...base.arena.safeZone, telegraphAt: 1_200, closesAt: 1_245, inset: 0, pendingInset: 1 } },
    };
    const warnedAfter = { ...warnedBefore, tick: 1_200 };
    tracker.observe(warnedBefore, warnedAfter);
    const repositioned = { ...exposed, segments: [{ x: 2, y: 10 }, { x: 1, y: 10 }] };
    tracker.observe(
      { ...warnedAfter, tick: 1_244, snakes: [repositioned, base.snakes[1]] },
      { ...warnedAfter, tick: 1_245, snakes: [repositioned, base.snakes[1]], arena: { ...warnedAfter.arena, safeZone: { ...warnedAfter.arena.safeZone, inset: 1, pendingInset: 2, telegraphAt: 1_290, closesAt: 1_335 } } },
    );
    expect(metrics.nova).toMatchObject({ zoneWarnings: 1, zoneRepositions: 1, zoneDeaths: 0 });

    const hazard = { id: 7, kind: "laser" as const, origin: { x: 1, y: 10 }, direction: "right" as const, length: 20, telegraphAt: 500, activatesAt: 545, activeUntil: 557 };
    const shielded = { ...repositioned, buffs: { ...repositioned.buffs, shield: 1 } };
    const hazardBefore = { ...base, tick: 500, snakes: [shielded, base.snakes[1]], arena: { ...base.arena, phase: "midgame" as const, hazards: [] } };
    const hazardTelegraph = { ...hazardBefore, arena: { ...hazardBefore.arena, hazards: [hazard] } };
    tracker.observe(hazardBefore, hazardTelegraph);
    tracker.observe(
      { ...hazardTelegraph, tick: 545 },
      { ...hazardTelegraph, tick: 546, snakes: [{ ...shielded, buffs: { ...shielded.buffs, shield: 0 } }, base.snakes[1]] },
    );
    tracker.observe(
      { ...hazardTelegraph, tick: 557, snakes: [{ ...shielded, buffs: { ...shielded.buffs, shield: 0 } }, base.snakes[1]] },
      { ...hazardTelegraph, tick: 558, snakes: [{ ...shielded, buffs: { ...shielded.buffs, shield: 0 } }, base.snakes[1]], arena: { ...hazardTelegraph.arena, hazards: [] } },
    );
    expect(metrics.nova).toMatchObject({ hazardEncounters: 1, hazardShieldBlocks: 1, hazardEvasions: 1 });
    expect(metrics.nova.powerUpCounters.shield).toEqual({ opportunities: 1, successes: 1 });
  });

  it("tracks objective pursuit, contest, capture, and food decay", () => {
    const base = createGame({ mode: "battle", snakes: SNAKES.slice(0, 2), seed: 93, mapArchetype: "open" });
    const metrics = Object.fromEntries(base.snakes.map((snake) => [snake.id, emptyAdaptiveBehavior()]));
    const tracker = new AdaptiveBehaviorTracker(metrics);
    const nova = { ...base.snakes[0], segments: [{ x: 5, y: 5 }, { x: 4, y: 5 }] };
    const objective = { id: 8, position: { x: 15, y: 5 }, radius: 2, expiresAt: 800, captureRequired: 45, progress: {}, contested: false };
    const before = { ...base, tick: 600, snakes: [nova, base.snakes[1]], arena: { ...base.arena, phase: "midgame" as const, objective: null } };
    const spawned = { ...before, arena: { ...before.arena, objective } };
    tracker.observe(before, spawned);
    const closer = { ...nova, segments: [{ x: 8, y: 5 }, { x: 7, y: 5 }] };
    tracker.observe(spawned, { ...spawned, tick: 601, snakes: [closer, base.snakes[1]] });
    const contesting = { ...closer, segments: [{ x: 14, y: 5 }, { x: 13, y: 5 }] };
    const contested = { ...spawned, tick: 602, snakes: [contesting, base.snakes[1]], arena: { ...spawned.arena, objective: { ...objective, contested: true } } };
    tracker.observe({ ...spawned, tick: 601, snakes: [closer, base.snakes[1]] }, contested);
    const captured = { ...contested, tick: 603, snakes: [{ ...contesting, objectiveCaptures: 1 }, base.snakes[1]], arena: { ...contested.arena, objective: null } };
    tracker.observe(contested, captured);
    expect(metrics.nova).toMatchObject({ objectiveOpportunities: 1, objectivePursuits: 1, objectiveContests: 1, objectiveCaptures: 1 });

    const expiringFood = { ...captured, food: [{ id: 77, position: { x: 30, y: 20 }, kind: "rare" as const, value: 3 as const, spawnedAt: 500, expiresAt: 604 }] };
    tracker.observe(expiringFood, { ...expiringFood, tick: 604, food: [] });
    expect(metrics.nova.foodExpirationsObserved).toBe(1);
  });
});
