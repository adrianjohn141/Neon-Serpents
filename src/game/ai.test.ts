import { describe, expect, it } from "vitest";
import { bellmanTarget, createBrain, DEFAULT_HYPERPARAMETERS, encodeObservation, OBSERVATION_SIZE, ReplayBuffer } from "./ai";
import { OBSERVATION_LAYOUT } from "./observation";
import { SNAKES } from "./constants";
import { castRays, createGame, movePoint, relativeDirection } from "./engine";
import type { Experience } from "./types";

const experience = (reward: number): Experience => ({ state: Array(OBSERVATION_SIZE).fill(0), action: 0, reward, nextState: Array(OBSERVATION_SIZE).fill(1), terminal: false });

describe("DQN observations and replay", () => {
  it("encodes exactly 16 rays and 228 normalized inputs", () => {
    const state = createGame({ mode: "training", snakes: [SNAKES[0]], seed: 4 });
    const observation = encodeObservation(state, state.snakes[0]);
    expect(castRays(state, state.snakes[0])).toHaveLength(16);
    expect(observation).toHaveLength(OBSERVATION_SIZE);
    expect(OBSERVATION_SIZE).toBe(228);
    expect([...observation].every((value) => value >= -1 && value <= 1)).toBe(true);
  });

  it("keeps the observation layout slices consistent with the encoder length", () => {
    const state = createGame({ mode: "training", snakes: [SNAKES[0]], seed: 4 });
    const observation = encodeObservation(state, state.snakes[0]);
    const danger = observation.slice(OBSERVATION_LAYOUT.danger.start, OBSERVATION_LAYOUT.danger.start + OBSERVATION_LAYOUT.danger.length);
    expect(Array.from(danger)).toEqual([0, 0, 0]);
  });

  it("detects obstacles on forward and left rays", () => {
    const state = createGame({ mode: "training", snakes: [SNAKES[0]], seed: 4 });
    const snake = state.snakes[0];
    const boxed = { ...state, obstacles: [
      movePoint(snake.segments[0], relativeDirection(snake.direction, 0)),
      movePoint(snake.segments[0], relativeDirection(snake.direction, 1)),
    ] };
    const rays = castRays(boxed, boxed.snakes[0]);
    expect(rays[0].entity).toBe("obstacle");
    expect(rays[12].entity).toBe("obstacle");
  });

  it("wraps its circular replay buffer and samples without replacement", () => {
    const buffer = new ReplayBuffer(3);
    [1, 2, 3, 4].forEach((value) => buffer.add(experience(value)));
    expect(buffer.size).toBe(3);
    const sampled = buffer.sample(3, () => 0);
    expect(sampled.map((entry) => entry.reward).sort()).toEqual([2, 3, 4]);
  });

  it("computes terminal and continuing Bellman targets", () => {
    expect(bellmanTarget(2, false, 10, .95)).toBeCloseTo(11.5);
    expect(bellmanTarget(-120, true, 50, .95)).toBe(-120);
  });

  it("creates independent neural metadata", () => {
    const nova = createBrain("nova");
    const ember = createBrain("ember");
    expect(nova.modelKey).not.toBe(ember.modelKey);
    expect(nova.epsilon).toBe(DEFAULT_HYPERPARAMETERS.epsilonStart);
  });
});
