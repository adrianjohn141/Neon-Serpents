import { describe, expect, it } from "vitest";
import { createBrain } from "./ai";
import { DEFAULT_CURRICULUM, SNAKES } from "./constants";
import { runBattleEpisode, runHazardEpisode, runObjectiveEpisode, runPowerUpEpisode, runSafeZoneEpisode, runSurvivalEpisode, selectTrainingScenario, type CurriculumAgent } from "./curriculum";
import type { Brain, RelativeAction, TrainingScenario } from "./types";

class FakeAgent implements CurriculumAgent {
  brain: Brain;
  remembered = 0;
  finishCalls = 0;
  scenarios: TrainingScenario[] = [];
  constructor(snakeId: string) { this.brain = createBrain(snakeId); }
  chooseAction(_observation: ArrayLike<number>, _explore: boolean, safeActions?: RelativeAction[]): RelativeAction {
    return safeActions && safeActions.length ? safeActions[0] : 0;
  }
  remember(_experience: unknown, scenario: TrainingScenario = "battle"): void {
    this.remembered += 1;
    this.scenarios.push(scenario);
    this.brain.scenarioSteps[scenario] += 1;
  }
  async train(): Promise<number | null> { return this.remembered; }
  finishEpisode(): void { this.finishCalls += 1; }
}

describe("persistent curriculum", () => {
  it("selects warmups from persisted per-brain step counters then uses the configured mixture", () => {
    const brain = createBrain("nova");
    const config = { ...DEFAULT_CURRICULUM, navigationWarmupSteps: 10, powerUpWarmupSteps: 20, safeZoneWarmupSteps: 0, hazardWarmupSteps: 0, objectiveWarmupSteps: 0 };
    expect(selectTrainingScenario(brain, config, () => .99)).toBe("survival");
    brain.scenarioSteps.survival = 10;
    expect(selectTrainingScenario(brain, config, () => .99)).toBe("powerup");
    brain.scenarioSteps.powerup = 20;
    expect(selectTrainingScenario(brain, config, () => .01)).toBe("survival");
    expect(selectTrainingScenario(brain, config, () => .2)).toBe("powerup");
    expect(selectTrainingScenario(brain, config, () => .7)).toBe("battle");
    expect(selectTrainingScenario(brain, config, () => .9)).toBe("series");
  });

  it("runs all five solo labs with the correct scenario labels", async () => {
    const agent = new FakeAgent("nova");
    await runSurvivalEpisode(agent, SNAKES[0], 1234);
    expect(agent.scenarios).toContain("survival");
    agent.scenarios = [];
    await runPowerUpEpisode(agent, SNAKES[0], 1235);
    expect(agent.scenarios).toContain("powerup");
    agent.scenarios = []; await runSafeZoneEpisode(agent, SNAKES[0], 1236); expect(agent.scenarios).toContain("safezone");
    agent.scenarios = []; await runHazardEpisode(agent, SNAKES[0], 1237); expect(agent.scenarios).toContain("hazard");
    agent.scenarios = []; await runObjectiveEpisode(agent, SNAKES[0], 1238); expect(agent.scenarios).toContain("objective");
    expect(agent.finishCalls).toBe(5);
  });

  it("trains only selected current agents when frozen opponents are present", async () => {
    const defs = SNAKES.slice(0, 3);
    const agents = defs.map((def) => new FakeAgent(def.id));
    const result = await runBattleEpisode(agents, defs, 4242, 2000, new Set([defs[0].id]));
    expect(result.ticks).toBeGreaterThan(0);
    expect(agents[0].remembered).toBeGreaterThan(0);
    expect(agents[0].finishCalls).toBe(1);
    expect(agents[1].remembered).toBe(0);
    expect(agents[1].finishCalls).toBe(0);
  });
});
