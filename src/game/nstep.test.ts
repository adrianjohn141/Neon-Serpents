import { describe, expect, it } from "vitest";
import { NStepAccumulator } from "./nstep";
import type { Experience } from "./types";

const transition = (reward: number, terminal = false): Experience => ({ state: [reward], action: 0, reward, nextState: [reward + 1], terminal });

describe("three-step returns", () => {
  it("discounts three rewards and flushes shorter terminal windows", () => {
    const accumulator = new NStepAccumulator(3, .5);
    expect(accumulator.add(transition(1))).toEqual([]);
    expect(accumulator.add(transition(2))).toEqual([]);
    const completed = accumulator.add(transition(4, true));
    expect(completed.map((entry) => entry.reward)).toEqual([3, 4, 4]);
    expect(completed.map((entry) => entry.nSteps)).toEqual([3, 2, 1]);
    expect(completed.every((entry) => entry.terminal)).toBe(true);
  });

  it("flushes partial non-terminal tails at an external episode cap", () => {
    const accumulator = new NStepAccumulator(3, .5);
    expect(accumulator.add(transition(1))).toEqual([]);
    expect(accumulator.add(transition(2))).toEqual([]);
    const flushed = accumulator.flush(true);
    expect(flushed).toHaveLength(2);
    expect(flushed[0]).toMatchObject({ reward: 2, nSteps: 2, terminal: true });
    expect(flushed[1]).toMatchObject({ reward: 2, nSteps: 1, terminal: true });
  });
});
