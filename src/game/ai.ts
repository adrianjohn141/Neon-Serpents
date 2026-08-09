import { OBSERVATION_SIZE } from "./observation";
import type { Brain, Experience, Hyperparameters, RelativeAction } from "./types";

export { ACTION_COUNT, OBSERVATION_SIZE, encodeObservation } from "./observation";

function configuredReplaySize(): number {
  const parsed = Number(process.env.NEXT_PUBLIC_REPLAY_BUFFER_SIZE ?? 10_000);
  return Number.isFinite(parsed) ? Math.max(1_000, Math.min(50_000, Math.round(parsed))) : 10_000;
}

export const DEFAULT_HYPERPARAMETERS: Hyperparameters = {
  learningRate: 0.0005,
  discountFactor: 0.95,
  epsilonStart: 1,
  epsilonMin: 0.05,
  epsilonDecaySteps: 250_000,
  batchSize: 64,
  replayBufferSize: configuredReplaySize(),
  warmupTransitions: 1_000,
  trainEverySteps: 4,
  targetSyncSteps: 1_500,
  nStep: 3,
  priorityAlpha: .6,
  priorityBetaStart: .4,
};

export function modelKey(snakeId: string): string {
  return `snake-${snakeId}-dqn`;
}

export function createBrain(snakeId: string, parameters = DEFAULT_HYPERPARAMETERS): Brain {
  return {
    snakeId,
    modelKey: `${modelKey(snakeId)}-v3`,
    modelVersion: 3,
    trainingSpecVersion: 3,
    observationSize: OBSERVATION_SIZE,
    generation: 0,
    episodes: 0,
    epsilon: parameters.epsilonStart,
    bestScore: 0,
    totalFood: 0,
    wins: 0,
    environmentSteps: 0,
    learningSteps: 0,
    lastLoss: null,
    scenarioSteps: { survival: 0, powerup: 0, safezone: 0, hazard: 0, objective: 0, battle: 0, series: 0 },
    powerUpOpportunities: 0,
    powerUpsClaimed: 0,
    powerUpApproachMisses: 0,
    rareFoodClaims: 0,
    objectiveCaptures: 0,
    bountyKills: 0,
    hazardDeaths: 0,
    zoneDeaths: 0,
    lastBenchmark: null,
  };
}

export function effectiveHyperparameters(global: Hyperparameters, overrides?: Partial<Hyperparameters>): Hyperparameters {
  return { ...global, ...overrides };
}

export function explorationRate(environmentSteps: number, parameters: Hyperparameters): number {
  const progress = Math.max(0, Math.min(1, environmentSteps / Math.max(1, parameters.epsilonDecaySteps)));
  return Math.max(parameters.epsilonMin, parameters.epsilonStart + (0.10 - parameters.epsilonStart) * progress);
}

export function completeEpisode(brain: Brain, score: number, food: number, won: boolean, parameters: Hyperparameters): Brain {
  const episodes = brain.episodes + 1;
  return {
    ...brain,
    generation: brain.generation + 1,
    episodes,
    epsilon: explorationRate(brain.environmentSteps, parameters),
    bestScore: Math.max(brain.bestScore, score),
    totalFood: brain.totalFood + food,
    wins: brain.wins + (won ? 1 : 0),
  };
}

export function actionLabel(action: RelativeAction): string {
  return action === 0 ? "FORWARD" : action === 1 ? "LEFT" : "RIGHT";
}

export class ReplayBuffer {
  private items: Experience[] = [];
  private cursor = 0;
  constructor(public capacity: number) {}
  get size(): number { return this.items.length; }
  add(value: Experience): void {
    if (this.items.length < this.capacity) this.items.push(value);
    else { this.items[this.cursor] = value; this.cursor = (this.cursor + 1) % this.capacity; }
  }
  sample(count: number, random = Math.random): Experience[] {
    const pool = this.items.slice();
    const result: Experience[] = [];
    const take = Math.min(count, pool.length);
    for (let index = 0; index < take; index += 1) {
      const selected = Math.floor(random() * pool.length);
      result.push(pool[selected]);
      pool.splice(selected, 1);
    }
    return result;
  }
  resize(capacity: number): void {
    this.capacity = capacity;
    if (this.items.length > capacity) this.items = this.items.slice(-capacity);
    this.cursor = this.items.length % capacity;
  }
  clear(): void { this.items = []; this.cursor = 0; }
}

export function bellmanTarget(reward: number, terminal: boolean, nextMaximum: number, discount: number, nSteps = 1): number {
  return reward + (terminal ? 0 : Math.pow(discount, nSteps) * nextMaximum);
}
