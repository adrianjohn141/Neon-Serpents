import { describe, expect, it } from "vitest";
import { POWER_UP_KINDS, SNAKES } from "./constants";
import { createGame, movePoint, stepGame } from "./engine";
import type { SnakeDefinition } from "./types";

const trainingSnake = SNAKES[0];

function definitions(count: number): SnakeDefinition[] {
  return Array.from({ length: count }, (_, index) => ({
    id: `agent-${index}`,
    name: `Agent ${index}`,
    color: `#${(0x40a060 + index * 0x111111).toString(16).slice(-6)}`,
    accent: "#ffffff",
  }));
}

describe("battle engine", () => {
  it("is deterministic for the same seed and actions", () => {
    let first = createGame({ seed: 424242 });
    let second = createGame({ seed: 424242 });
    for (let tick = 0; tick < 120 && first.status === "running"; tick += 1) {
      const actions = Object.fromEntries(first.snakes.map((snake, index) => [snake.id, ((tick + index) % 3) as 0 | 1 | 2]));
      first = stepGame(first, actions).state;
      second = stepGame(second, actions).state;
    }
    expect(second).toEqual(first);
  });

  it("kills a snake when it hits an obstacle", () => {
    const state = createGame({ mode: "training", snakes: [trainingSnake], seed: 1 });
    const snake = state.snakes[0];
    const blocked = { ...state, obstacles: [movePoint(snake.segments[0], snake.direction)] };
    const result = stepGame(blocked, { nova: 0 });
    expect(result.state.snakes[0].alive).toBe(false);
    expect(result.state.snakes[0].deathReason).toContain("obstacle");
    expect(result.rewardBreakdowns.nova.death).toBe(-120);
    expect(result.rewards.nova).toBeLessThan(-120);
  });

  it("applies every power-up and removes the single arena drop", () => {
    for (const kind of POWER_UP_KINDS) {
      const state = createGame({ mode: "battle", snakes: SNAKES.slice(0, 2), seed: 2 });
      const snake = state.snakes[0];
      const prepared = {
        ...state,
        obstacles: [],
        powerUp: { id: 1, kind, position: movePoint(snake.segments[0], snake.direction), expiresAt: 99 },
      };
      const result = stepGame(prepared, { nova: 0, ember: 0 });
      expect(result.state.powerUp, kind).toBeNull();
      expect(result.state.snakes.find((snake) => snake.id === "nova")?.powerUps, kind).toBe(1);
    }
  });

  it("adds approach and claim rewards without overwriting the base step reward", () => {
    const state = createGame({ mode: "powerup", snakes: [trainingSnake], seed: 22 });
    const snake = state.snakes[0];
    const prepared = {
      ...state,
      obstacles: [],
      food: [{ id: 999, position: { x: state.width - 2, y: state.height - 2 }, kind: "normal" as const, value: 1 as const, spawnedAt: 0, expiresAt: 300 }],
      powerUp: { id: 1, kind: "shield" as const, position: movePoint(snake.segments[0], snake.direction), expiresAt: 99 },
    };
    const result = stepGame(prepared, { nova: 0 });
    expect(result.rewardBreakdowns.nova.step).toBeLessThan(0);
    expect(result.rewardBreakdowns.nova.powerUpApproach).toBeGreaterThan(0);
    expect(result.rewardBreakdowns.nova.powerUpClaim).toBe(30);
    expect(result.rewards.nova).toBeCloseTo(Object.values(result.rewardBreakdowns.nova).reduce((sum, value) => sum + value, 0));
  });

  it("keeps shaping bounded when phase occupies an obstacle cell", () => {
    const state = createGame({ mode: "powerup", snakes: [trainingSnake], seed: 23 });
    const snake = {
      ...state.snakes[0],
      direction: "right" as const,
      segments: [{ x: 10, y: 10 }, { x: 9, y: 10 }],
      buffs: { ...state.snakes[0].buffs, phaseUntil: 50 },
    };
    const prepared = {
      ...state,
      tick: 5,
      snakes: [snake],
      obstacles: [{ x: 10, y: 10 }],
      food: [],
      powerUp: { id: 1, kind: "shield" as const, position: { x: 12, y: 10 }, expiresAt: 99 },
    };
    const shaping = stepGame(prepared, { nova: 0 }).rewardBreakdowns.nova.powerUpApproach;
    expect(shaping).toBeGreaterThan(0);
    expect(shaping).toBeLessThan(1);
  });

  it("awards the killer for forcing a rival into its body", () => {
    const state = createGame({ mode: "battle", snakes: SNAKES.slice(0, 3), seed: 5 });
    const nova = { ...state.snakes[0], id: "nova", segments: [{ x: 5, y: 5 }, { x: 4, y: 5 }, { x: 3, y: 5 }], direction: "right" as const };
    const ember = { ...state.snakes[1], id: "ember", buffs: { ...state.snakes[1].buffs, frozenUntil: 1000 }, segments: [{ x: 8, y: 5 }, { x: 7, y: 5 }, { x: 6, y: 5 }], direction: "down" as const };
    const volt = { ...state.snakes[2], id: "volt", buffs: { ...state.snakes[2].buffs, frozenUntil: 1000 }, segments: [{ x: 40, y: 40 }, { x: 39, y: 40 }, { x: 38, y: 40 }], direction: "down" as const };
    const prepared = { ...state, snakes: [nova, ember, volt], obstacles: [], food: [] };
    const result = stepGame(prepared, { nova: 0, ember: 0, volt: 0 });
    expect(result.state.snakes.find((snake) => snake.id === "nova")?.alive).toBe(false);
    expect(result.rewards.ember).toBe(20);
    expect(result.state.snakes.find((snake) => snake.id === "ember")?.kills).toBe(1);
  });

  it("absorbs exactly one lethal collision with a shield", () => {
    const state = createGame({ mode: "training", snakes: [trainingSnake], seed: 3 });
    const snake = state.snakes[0];
    const prepared = {
      ...state,
      snakes: [{ ...snake, buffs: { ...snake.buffs, shield: 1 } }],
      obstacles: [movePoint(snake.segments[0], snake.direction)],
    };
    const first = stepGame(prepared, { nova: 0 }).state;
    expect(first.snakes[0].alive).toBe(true);
    expect(first.snakes[0].buffs.shield).toBe(0);
    const second = stepGame(first, { nova: 0 }).state;
    expect(second.snakes[0].alive).toBe(false);
  });

  it("spawns 2 through 8 snakes safely and points every head inward", () => {
    for (let count = 2; count <= 8; count += 1) {
      const state = createGame({ mode: "battle", snakes: definitions(count), seed: count });
      const occupied = new Set<string>();
      const obstacles = new Set(state.obstacles.map((point) => `${point.x},${point.y}`));
      const center = { x: state.width / 2, y: state.height / 2 };
      for (const snake of state.snakes) {
        for (const segment of snake.segments) {
          const key = `${segment.x},${segment.y}`;
          expect(occupied.has(key), `${count} snakes overlap at ${key}`).toBe(false);
          expect(obstacles.has(key), `${count} snakes overlap an obstacle at ${key}`).toBe(false);
          expect(segment.x).toBeGreaterThanOrEqual(0);
          expect(segment.y).toBeGreaterThanOrEqual(0);
          expect(segment.x).toBeLessThan(state.width);
          expect(segment.y).toBeLessThan(state.height);
          occupied.add(key);
        }
        const head = snake.segments[0];
        const next = movePoint(head, snake.direction);
        const before = Math.abs(head.x - center.x) + Math.abs(head.y - center.y);
        const after = Math.abs(next.x - center.x) + Math.abs(next.y - center.y);
        expect(after).toBeLessThan(before);
      }
    }
  });

  it("rejects battle rosters outside the 2 through 8 range", () => {
    expect(() => createGame({ mode: "battle", snakes: definitions(1) })).toThrow(/between 2 and 8/);
    expect(() => createGame({ mode: "battle", snakes: definitions(9) })).toThrow(/between 2 and 8/);
  });
});
