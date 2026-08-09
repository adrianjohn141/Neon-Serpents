import { describe, expect, it } from "vitest";
import { OBSERVATION_SIZE } from "./observation";
import { PrioritizedReplayBuffer } from "./per";
import type { Experience } from "./types";

const experience = (reward: number): Experience => ({
  state: new Float32Array(OBSERVATION_SIZE), action: 0, reward,
  nextState: new Float32Array(OBSERVATION_SIZE), terminal: false,
});

describe("prioritized replay", () => {
  it("samples, normalizes importance weights, and updates priorities", () => {
    const replay = new PrioritizedReplayBuffer(4, .6);
    [1, 2, 3, 4].forEach((reward) => replay.add(experience(reward)));
    replay.updateMany([0, 1, 2, 3], [1, 1, 1, 100]);
    const sampled = replay.sample(4, .4, () => .5);
    expect(sampled.experiences).toHaveLength(4);
    expect(sampled.weights.every((weight) => weight > 0 && weight <= 1)).toBe(true);
    expect(sampled.indices).toContain(3);
  });

  it("retains the newest entries when resized", () => {
    const replay = new PrioritizedReplayBuffer(4);
    [1, 2, 3, 4].forEach((reward) => replay.add(experience(reward)));
    replay.resize(2);
    expect(replay.size).toBe(2);
    expect(replay.sample(2, 1, () => .5).experiences.map((entry) => entry.reward).sort()).toEqual([3, 4]);
  });
});
