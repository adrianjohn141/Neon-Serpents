import { describe, expect, it } from "vitest";
import { MAP_ARCHETYPES, POWER_UP_KINDS, SNAKES } from "./constants";
import { advanceSafeZone, createArenaState, insideSafeBounds, isHeldOutMapSeed, makeMapObstacles, mapForSeed } from "./adaptive-arena";
import { createGame, isDangerAt, relativeDirection, stepGame } from "./engine";
import type { ArenaHazard, RelativeAction } from "./types";

const actions = [0, 1, 2] as RelativeAction[];

describe("Adaptive Arena mechanics", () => {
  it("keeps all six deterministic map archetypes connected and separates held-out seeds", () => {
    for (const map of MAP_ARCHETYPES) {
      const obstacles = new Set(makeMapObstacles(map, 72, 44).map((point) => `${point.x},${point.y}`));
      const open: string[] = [];
      for (let y = 0; y < 44; y += 1) for (let x = 0; x < 72; x += 1) if (!obstacles.has(`${x},${y}`)) open.push(`${x},${y}`);
      const seen = new Set([open[0]]); const queue = [open[0]];
      for (let cursor = 0; cursor < queue.length; cursor += 1) {
        const [x, y] = queue[cursor].split(",").map(Number);
        for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          const key = `${x + dx},${y + dy}`;
          if (x + dx >= 0 && y + dy >= 0 && x + dx < 72 && y + dy < 44 && !obstacles.has(key) && !seen.has(key)) { seen.add(key); queue.push(key); }
        }
      }
      expect(seen.size, map).toBe(open.length);
    }
    for (let seed = 0; seed < 100; seed += 1) {
      expect(isHeldOutMapSeed(seed, mapForSeed(seed, false))).toBe(false);
      expect(isHeldOutMapSeed(seed, mapForSeed(seed, true))).toBe(true);
    }
  });

  it("shrinks no smaller than 56 by 28 and telegraphs every contraction", () => {
    let safe = createArenaState(72, 44, "open", true, 1, {}).safeZone;
    for (let index = 0; index < 20; index += 1) safe = advanceSafeZone(safe, 72, 44);
    expect(72 - safe.inset * 2).toBeGreaterThanOrEqual(56);
    expect(44 - safe.inset * 2).toBeGreaterThanOrEqual(28);
    expect(safe.inset).toBe(8);
  });

  it("changes phase food density at ticks 500 and 1200", () => {
    let state = createGame({ mode: "battle", snakes: SNAKES.slice(0, 2), seed: 40, mapArchetype: "open" });
    state = { ...state, tick: 499, obstacles: [] };
    state = stepGame(state, { nova: 0, ember: 0 }).state;
    expect(state.arena.phase).toBe("midgame");
    expect(state.food).toHaveLength(4);
    state = { ...state, tick: 1199, status: "running", snakes: state.snakes.map((snake) => ({ ...snake, alive: true })) };
    state = stepGame(state, { nova: 0, ember: 0 }).state;
    expect(state.arena.phase).toBe("endgame");
    expect(state.food).toHaveLength(2);
  });

  it("eliminates snakes outside a closing zone and lets Second Chance rescue one", () => {
    const base = createGame({ mode: "battle", snakes: SNAKES.slice(0, 2), seed: 41, mapArchetype: "open" });
    const outside = { ...base.snakes[0], direction: "down" as const, segments: [{ x: 0, y: 10 }, { x: 0, y: 9 }] };
    const safe = { ...base.snakes[1], buffs: { ...base.snakes[1].buffs, frozenUntil: 2_000 }, segments: [{ x: 35, y: 22 }, { x: 34, y: 22 }] };
    const prepared = { ...base, tick: 1_244, obstacles: [], snakes: [outside, safe], arena: { ...base.arena, phase: "endgame" as const, safeZone: { ...base.arena.safeZone, inset: 0, pendingInset: 1, closesAt: 1_245 } } };
    const eliminated = stepGame(prepared, { nova: 0, ember: 0 }).state.snakes[0];
    expect(eliminated.alive).toBe(false);
    expect(eliminated.deathReason).toContain("zone");
    const rescuedInput = { ...prepared, snakes: [{ ...outside, buffs: { ...outside.buffs, secondChance: 1 } }, safe] };
    const rescued = stepGame(rescuedInput, { nova: 0, ember: 0 }).state.snakes[0];
    expect(rescued.alive).toBe(true);
    expect(rescued.buffs.secondChance).toBe(0);
    expect(insideSafeBounds(rescued.segments[0], prepared.width, prepared.height, 1)).toBe(true);
  });

  it("captures an uncontested energy core after the 45th tick", () => {
    const base = createGame({ mode: "battle", snakes: SNAKES.slice(0, 2), seed: 42, mapArchetype: "open" });
    const nova = { ...base.snakes[0], direction: "right" as const, segments: [{ x: 9, y: 10 }, { x: 8, y: 10 }] };
    const ember = { ...base.snakes[1], buffs: { ...base.snakes[1].buffs, frozenUntil: 2_000 }, segments: [{ x: 50, y: 30 }, { x: 49, y: 30 }] };
    const prepared = { ...base, tick: 600, obstacles: [], snakes: [nova, ember], arena: { ...base.arena, phase: "midgame" as const, nextObjectiveAt: 9_999, objective: { id: 1, position: { x: 10, y: 10 }, radius: 2, expiresAt: 900, captureRequired: 45, progress: { nova: 44 }, contested: false } } };
    const result = stepGame(prepared, { nova: 0, ember: 0 });
    expect(result.state.arena.objective).toBeNull();
    expect(result.state.snakes[0].objectiveCaptures).toBe(1);
    expect(result.rewardBreakdowns.nova.objectiveCapture).toBe(8);
  });

  it("lets Shield absorb lasers and Phase cross non-laser hazards", () => {
    const base = createGame({ mode: "battle", snakes: SNAKES.slice(0, 2), seed: 43, mapArchetype: "open" });
    const laser: ArenaHazard = { id: 1, kind: "laser", origin: { x: 11, y: 10 }, direction: "right", length: 1, telegraphAt: 590, activatesAt: 601, activeUntil: 620 };
    const blocks: ArenaHazard = { ...laser, id: 2, kind: "blocks" };
    const rival = { ...base.snakes[1], buffs: { ...base.snakes[1].buffs, frozenUntil: 2_000 } };
    const shielded = { ...base.snakes[0], direction: "right" as const, segments: [{ x: 10, y: 10 }, { x: 9, y: 10 }], buffs: { ...base.snakes[0].buffs, shield: 1 } };
    const shieldState = { ...base, tick: 600, obstacles: [], snakes: [shielded, rival], arena: { ...base.arena, phase: "midgame" as const, nextHazardAt: 9_999, hazards: [laser] } };
    const afterShield = stepGame(shieldState, { nova: 0, ember: 0 }).state.snakes[0];
    expect(afterShield.alive).toBe(true); expect(afterShield.buffs.shield).toBe(0);
    const phased = { ...shielded, buffs: { ...shielded.buffs, shield: 0, phaseUntil: 700 } };
    const phaseState = { ...shieldState, snakes: [phased, rival], arena: { ...shieldState.arena, hazards: [blocks] } };
    expect(stepGame(phaseState, { nova: 0, ember: 0 }).state.snakes[0].alive).toBe(true);
  });

  it("keeps walls absolute even when Shield is active", () => {
    const base = createGame({ mode: "battle", snakes: SNAKES.slice(0, 2), seed: 45, mapArchetype: "open" });
    const shielded = {
      ...base.snakes[0], direction: "left" as const,
      segments: [{ x: 0, y: 10 }, { x: 1, y: 10 }],
      buffs: { ...base.snakes[0].buffs, shield: 1 },
    };
    const rival = { ...base.snakes[1], buffs: { ...base.snakes[1].buffs, frozenUntil: 2_000 } };
    const result = stepGame({ ...base, obstacles: [], snakes: [shielded, rival] }, { nova: 0, ember: 0 }).state.snakes[0];
    expect(result.alive).toBe(false);
    expect(result.buffs.shield).toBe(1);
    expect(result.deathReason).toContain("wall");
  });

  it("spawns powerups inside the pending safe area", () => {
    const base = createGame({ mode: "battle", snakes: SNAKES.slice(0, 2), seed: 46, mapArchetype: "open" });
    const snakes = base.snakes.map((snake) => ({ ...snake, buffs: { ...snake.buffs, frozenUntil: 2_000 } }));
    const prepared = {
      ...base, tick: 1_300, obstacles: [], snakes, nextPowerUpAt: 1_301,
      arena: {
        ...base.arena, phase: "endgame" as const,
        safeZone: { ...base.arena.safeZone, inset: 3, pendingInset: 4, telegraphAt: 9_000, closesAt: 9_045 },
      },
    };
    const powerUp = stepGame(prepared, { nova: 0, ember: 0 }).state.powerUp;
    expect(powerUp).not.toBeNull();
    expect(insideSafeBounds(powerUp!.position, prepared.width, prepared.height, 4)).toBe(true);
  });

  it("limits opening hazards to the rapid-hazard map and relocates endgame cores faster", () => {
    const rapid = createGame({ mode: "battle", snakes: SNAKES.slice(0, 2), seed: 47, mapArchetype: "hazard" });
    const frozen = rapid.snakes.map((snake) => ({ ...snake, buffs: { ...snake.buffs, frozenUntil: 2_000 } }));
    const withHazard = stepGame({ ...rapid, tick: 179, snakes: frozen, obstacles: [], arena: { ...rapid.arena, nextHazardAt: 180 } }, { nova: 0, ember: 0 }).state;
    expect(withHazard.arena.phase).toBe("opening");
    expect(withHazard.arena.hazards).toHaveLength(1);

    const endgame = stepGame({
      ...rapid, tick: 1_200, snakes: frozen, obstacles: [],
      arena: { ...rapid.arena, phase: "endgame" as const, nextObjectiveAt: 1_201, nextHazardAt: 9_999 },
    }, { nova: 0, ember: 0 }).state;
    expect(endgame.arena.objective).not.toBeNull();
    expect(endgame.arena.objective!.expiresAt - endgame.tick).toBe(120);
  });

  it("cycles through all twelve powerups in a deterministic lab episode", () => {
    let state = createGame({ mode: "powerup", snakes: [SNAKES[0]], seed: 44 });
    const seen = new Set<string>(); let lastId = -1;
    while (state.status === "running") {
      if (state.powerUp && state.powerUp.id !== lastId) { lastId = state.powerUp.id; seen.add(state.powerUp.kind); }
      const snake = state.snakes[0];
      const action = actions.find((candidate) => !isDangerAt(state, snake, relativeDirection(snake.direction, candidate), 2)) ?? 0;
      state = stepGame(state, { nova: action }).state;
    }
    expect([...seen].sort()).toEqual([...POWER_UP_KINDS].sort());
  });
});
