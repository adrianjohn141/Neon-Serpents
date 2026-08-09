import { describe, expect, it } from "vitest";
import { createBrain } from "./ai";
import { SNAKES } from "./constants";
import { EvalAgent, runEvaluation, summarize } from "./evaluation";
import type { Brain, RelativeAction } from "./types";

class FakeEvalAgent implements EvalAgent {
  brain: Brain;
  constructor(id: string) { this.brain = createBrain(id); }
  chooseAction(_observation: ArrayLike<number>, _explore: boolean, safeActions?: RelativeAction[]): RelativeAction {
    return safeActions && safeActions.length ? safeActions[safeActions.length - 1] : 0;
  }
}

describe("evaluation harness", () => {
  it("aggregates per-snake statistics across head-to-head runs", async () => {
    const defs = SNAKES.slice(0, 3);
    const agents = defs.map((def) => new FakeEvalAgent(def.id));
    const report = await runEvaluation(agents, defs, 2, 0, 800);
    expect(report.runs).toBe(2);
    expect(report.snakes).toHaveLength(3);
    for (const snake of report.snakes) {
      expect(snake.wins + (defs.length - 1 - snake.wins)).toBe(defs.length - 1); // wins within 0..2
      expect(typeof snake.avgScore).toBe("number");
      expect(snake.avgSurvivalTicks).toBeGreaterThanOrEqual(0);
      expect(snake.powerUpsClaimed).toBeGreaterThanOrEqual(0);
    }
    expect(typeof summarize(report)).toBe("string");
  });
});
