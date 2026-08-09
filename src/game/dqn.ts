import * as tf from "@tensorflow/tfjs";
import { ACTION_COUNT, bellmanTarget, completeEpisode, explorationRate, OBSERVATION_SIZE } from "./ai";
import { NStepAccumulator } from "./nstep";
import { PrioritizedReplayBuffer } from "./per";
import { createBrainBundle, loadModelFromBundle, type BrainBundle } from "./model-bundle";
import type { Brain, EvalReport, EvalSnakeStats, Experience, Hyperparameters, RelativeAction, TrainingScenario } from "./types";

const modelUrl = (key: string) => `indexeddb://${key}`;
const backupKey = (key: string) => `${key}-pre-hive`;

export async function initializeTensorFlow(): Promise<void> {
  await tf.setBackend("cpu");
  await tf.ready();
}

export function createDqnModel(parameters: Hyperparameters, observationSize = OBSERVATION_SIZE): tf.Sequential {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [observationSize], units: 128, activation: "relu", kernelInitializer: "heNormal" }));
  model.add(tf.layers.dense({ units: 64, activation: "relu", kernelInitializer: "heNormal" }));
  model.add(tf.layers.dense({ units: ACTION_COUNT, activation: "linear" }));
  return model;
}

function copyWeights(source: tf.LayersModel, target: tf.LayersModel): void {
  const weights = source.getWeights();
  target.setWeights(weights);
}

export function synchronizeTarget(source: tf.LayersModel, target: tf.LayersModel): void {
  copyWeights(source, target);
}

export function averageWeightTensors(weightSets: tf.Tensor[][]): tf.Tensor[] {
  if (!weightSets.length) return [];
  return weightSets[0].map((_, weightIndex) => tf.tidy(() =>
    tf.stack(weightSets.map((weights) => weights[weightIndex])).mean(0),
  ));
}

async function removeModelIfPresent(key: string): Promise<void> {
  try { await tf.io.removeModel(modelUrl(key)); } catch { /* Model does not exist yet. */ }
}

export class DqnAgent {
  online: tf.LayersModel;
  target: tf.LayersModel;
  replay: PrioritizedReplayBuffer;
  brain: Brain;
  parameters: Hyperparameters;
  private optimizer: tf.Optimizer;
  private dirtyEpisodes = 0;
  private lastSavedAt = Date.now();
  private nStep: NStepAccumulator;
  private readOnly = false;

  private constructor(brain: Brain, parameters: Hyperparameters, online: tf.LayersModel) {
    this.brain = brain;
    this.parameters = parameters;
    this.online = online;
    this.optimizer = tf.train.adam(parameters.learningRate);
    this.target = createDqnModel(parameters);
    copyWeights(this.online, this.target);
    this.replay = new PrioritizedReplayBuffer(parameters.replayBufferSize, parameters.priorityAlpha);
    this.nStep = new NStepAccumulator(parameters.nStep, parameters.discountFactor);
  }

  static async load(brain: Brain, parameters: Hyperparameters): Promise<DqnAgent> {
    let model: tf.LayersModel;
    try {
      model = await tf.loadLayersModel(modelUrl(brain.modelKey));
      const inputSize = model.inputs[0]?.shape.at(-1);
      const outputSize = model.outputs[0]?.shape.at(-1);
      if (inputSize !== OBSERVATION_SIZE || outputSize !== ACTION_COUNT) {
        model.dispose();
        throw new Error("Stored model architecture is incompatible.");
      }
    }
    catch {
      await removeModelIfPresent(brain.modelKey);
      model = createDqnModel(parameters);
      await model.save(modelUrl(brain.modelKey), { includeOptimizer: false });
    }
    return new DqnAgent(brain, parameters, model);
  }

  static createFresh(brain: Brain, parameters: Hyperparameters): DqnAgent {
    return new DqnAgent(brain, parameters, createDqnModel(parameters));
  }

  cloneFrozen(): DqnAgent {
    const model = createDqnModel(this.parameters);
    copyWeights(this.online, model);
    return new DqnAgent(structuredClone(this.brain), { ...this.parameters }, model);
  }

  chooseAction(observation: ArrayLike<number>, explore: boolean, safeActions?: RelativeAction[], random = Math.random): RelativeAction {
    if (explore && random() < this.brain.epsilon) {
      if (safeActions && safeActions.length > 0) return safeActions[Math.floor(random() * safeActions.length)];
      return Math.floor(random() * ACTION_COUNT) as RelativeAction;
    }
    return tf.tidy(() => {
      const input = tf.tensor2d([Array.from(observation)], [1, OBSERVATION_SIZE]);
      const prediction = this.online.predict(input) as tf.Tensor;
      const values = Array.from(prediction.dataSync());
      if (safeActions && safeActions.length > 0) {
        for (let i = 0; i < ACTION_COUNT; i++) {
          if (!safeActions.includes(i as RelativeAction)) {
            values[i] = -Infinity;
          }
        }
      }
      return values.indexOf(Math.max(...values)) as RelativeAction;
    });
  }

  remember(
    experience: Experience,
    scenario: TrainingScenario = "battle",
    metrics: { opportunity?: boolean; claimed?: boolean; approachMiss?: boolean } = {},
  ): void {
    if (this.readOnly) return;
    const packed: Experience = {
      ...experience,
      state: experience.state instanceof Float32Array ? experience.state : Float32Array.from(experience.state),
      nextState: experience.nextState instanceof Float32Array ? experience.nextState : Float32Array.from(experience.nextState),
    };
    this.nStep.add(packed).forEach((entry) => this.replay.add(entry));
    const environmentSteps = this.brain.environmentSteps + 1;
    this.brain = {
      ...this.brain,
      environmentSteps,
      epsilon: explorationRate(environmentSteps, this.parameters),
      scenarioSteps: { ...this.brain.scenarioSteps, [scenario]: this.brain.scenarioSteps[scenario] + 1 },
      powerUpOpportunities: this.brain.powerUpOpportunities + Number(metrics.opportunity),
      powerUpsClaimed: this.brain.powerUpsClaimed + Number(metrics.claimed),
      powerUpApproachMisses: this.brain.powerUpApproachMisses + Number(metrics.approachMiss),
      rareFoodClaims: this.brain.rareFoodClaims + Number((experience.rewardBreakdown?.rareFoodClaim ?? 0) !== 0),
      objectiveCaptures: this.brain.objectiveCaptures + Number((experience.rewardBreakdown?.objectiveCapture ?? 0) !== 0),
      bountyKills: this.brain.bountyKills + Number((experience.rewardBreakdown?.bountyKill ?? 0) !== 0),
      hazardDeaths: this.brain.hazardDeaths + Number(experience.deathCause === "hazard"),
      zoneDeaths: this.brain.zoneDeaths + Number(experience.deathCause === "zone"),
    };
  }

  async train(force = false): Promise<number | null> {
    if (this.readOnly) return null;
    const minimum = force ? Math.min(this.parameters.batchSize, this.replay.size) : this.parameters.warmupTransitions;
    if (this.replay.size < Math.max(1, minimum)) return null;
    if (!force && this.brain.environmentSteps % this.parameters.trainEverySteps !== 0) return null;
    const betaProgress = Math.min(1, this.brain.learningSteps / Math.max(1, this.parameters.epsilonDecaySteps));
    const beta = this.parameters.priorityBetaStart + (1 - this.parameters.priorityBetaStart) * betaProgress;
    const sampled = this.replay.sample(this.parameters.batchSize, beta);
    const batch = sampled.experiences;
    if (!batch.length) return null;

    const states = tf.tensor2d(batch.map((entry) => Array.from(entry.state)), [batch.length, OBSERVATION_SIZE]);
    const nextStates = tf.tensor2d(batch.map((entry) => Array.from(entry.nextState)), [batch.length, OBSERVATION_SIZE]);
    const currentTensor = this.online.predict(states) as tf.Tensor2D;
    const nextOnlineTensor = this.online.predict(nextStates) as tf.Tensor2D;
    const nextTargetTensor = this.target.predict(nextStates) as tf.Tensor2D;
    const current = await currentTensor.array();
    const nextOnline = await nextOnlineTensor.array();
    const nextTarget = await nextTargetTensor.array();
    currentTensor.dispose();
    nextOnlineTensor.dispose();
    nextTargetTensor.dispose();
    nextStates.dispose();

    const tdErrors: number[] = [];
    for (let index = 0; index < batch.length; index += 1) {
      const bestNextAction = nextOnline[index].indexOf(Math.max(...nextOnline[index]));
      const nextMaximum = nextTarget[index][bestNextAction];
      const targetValue = bellmanTarget(
        batch[index].reward,
        batch[index].terminal,
        nextMaximum,
        this.parameters.discountFactor,
        batch[index].nSteps ?? 1,
      );
      tdErrors.push(targetValue - current[index][batch[index].action]);
      current[index][batch[index].action] = targetValue;
    }
    const targets = tf.tensor2d(current, [batch.length, ACTION_COUNT]);
    const sampleWeights = tf.tensor1d(sampled.weights);
    const lossTensor = this.optimizer.minimize(() => tf.tidy(() => {
      const predictions = this.online.apply(states, { training: true }) as tf.Tensor2D;
      const absoluteError = targets.sub(predictions).abs();
      const quadratic = absoluteError.minimum(1);
      const linear = absoluteError.sub(quadratic);
      const huber = quadratic.square().mul(.5).add(linear);
      // Only the selected action differs from its target, so summing the
      // action axis preserves the exact per-transition Huber magnitude.
      return huber.sum(1).mul(sampleWeights).mean();
    }), true);
    const loss = lossTensor ? lossTensor.dataSync()[0] : Number.NaN;
    lossTensor?.dispose();
    states.dispose();
    targets.dispose();
    sampleWeights.dispose();
    this.replay.updateMany(sampled.indices, tdErrors);
    const learningSteps = this.brain.learningSteps + 1;
    this.brain = { ...this.brain, learningSteps, lastLoss: Number.isFinite(loss) ? loss : null };
    if (learningSteps % this.parameters.targetSyncSteps === 0) copyWeights(this.online, this.target);
    return this.brain.lastLoss;
  }

  finishEpisode(score: number, food: number, won = false): void {
    if (this.readOnly) return;
    this.nStep.flush(true).forEach((entry) => this.replay.add(entry));
    this.brain = completeEpisode(this.brain, score, food, won, this.parameters);
    this.dirtyEpisodes += 1;
  }

  recordBenchmark(stats: EvalSnakeStats, report: EvalReport): void {
    this.brain = {
      ...this.brain,
      lastBenchmark: {
        evaluatedAt: report.finishedAt,
        matches: report.runs,
        winRate: stats.wins / Math.max(1, report.runs),
        avgSurvivalTicks: stats.avgSurvivalTicks,
        foodPerMatch: stats.foodEaten / Math.max(1, report.runs),
        powerUpClaimRate: stats.powerUpClaimRate,
        deathsPerThousandTicks: stats.deathsPerThousandTicks,
      },
    };
    this.dirtyEpisodes += 1;
  }

  updateParameters(parameters: Hyperparameters): void {
    this.parameters = parameters;
    this.replay.resize(parameters.replayBufferSize);
    this.replay.alpha = parameters.priorityAlpha;
    this.nStep.clear();
    this.nStep = new NStepAccumulator(parameters.nStep, parameters.discountFactor);
    this.optimizer.dispose();
    this.optimizer = tf.train.adam(parameters.learningRate);
  }

  async save(force = false): Promise<boolean> {
    if (this.readOnly) return false;
    const due = force || this.dirtyEpisodes >= 50 || Date.now() - this.lastSavedAt >= 15_000;
    if (!due) return false;
    await this.online.save(modelUrl(this.brain.modelKey), { includeOptimizer: false });
    this.dirtyEpisodes = 0;
    this.lastSavedAt = Date.now();
    return true;
  }

  exportBundle(): Promise<BrainBundle> {
    return createBrainBundle(this.online, this.brain, this.parameters);
  }

  async importBundle(value: unknown): Promise<void> {
    const { bundle, model } = await loadModelFromBundle(value, this.brain.snakeId);
    this.online.dispose();
    this.target.dispose();
    this.online = model;
    this.parameters = bundle.hyperparameters;
    this.optimizer.dispose();
    this.optimizer = tf.train.adam(this.parameters.learningRate);
    this.target = createDqnModel(this.parameters);
    copyWeights(this.online, this.target);
    this.brain = { ...bundle.brain, modelKey: this.brain.modelKey };
    this.replay = new PrioritizedReplayBuffer(this.parameters.replayBufferSize, this.parameters.priorityAlpha);
    this.nStep = new NStepAccumulator(this.parameters.nStep, this.parameters.discountFactor);
    this.dirtyEpisodes = 0;
    this.readOnly = false;
    await this.save(true);
  }

  async useRuntimeBundle(value: unknown): Promise<void> {
    const { model } = await loadModelFromBundle(value, this.brain.snakeId);
    this.online.dispose();
    this.target.dispose();
    this.optimizer.dispose();
    this.online = model;
    this.optimizer = tf.train.adam(this.parameters.learningRate);
    this.target = createDqnModel(this.parameters);
    copyWeights(this.online, this.target);
    this.replay.clear();
    this.nStep.clear();
    this.readOnly = true;
  }

  async reset(): Promise<void> {
    this.readOnly = false;
    this.online.dispose();
    this.target.dispose();
    await removeModelIfPresent(this.brain.modelKey);
    await removeModelIfPresent(backupKey(this.brain.modelKey));
    this.brain = { ...this.brain, ...{
      generation: 0, episodes: 0, epsilon: this.parameters.epsilonStart, bestScore: 0, totalFood: 0,
      wins: 0, environmentSteps: 0, learningSteps: 0, lastLoss: null,
      scenarioSteps: { survival: 0, powerup: 0, safezone: 0, hazard: 0, objective: 0, battle: 0, series: 0 },
      powerUpOpportunities: 0, powerUpsClaimed: 0, powerUpApproachMisses: 0,
      rareFoodClaims: 0, objectiveCaptures: 0, bountyKills: 0, hazardDeaths: 0, zoneDeaths: 0, lastBenchmark: null,
    } };
    this.online = createDqnModel(this.parameters);
    this.optimizer.dispose();
    this.optimizer = tf.train.adam(this.parameters.learningRate);
    this.target = createDqnModel(this.parameters);
    copyWeights(this.online, this.target);
    this.replay.clear();
    this.nStep.clear();
    await this.save(true);
  }

  async backup(): Promise<void> {
    await removeModelIfPresent(backupKey(this.brain.modelKey));
    await this.online.save(modelUrl(backupKey(this.brain.modelKey)), { includeOptimizer: false });
  }

  async restoreBackup(): Promise<void> {
    const restored = await tf.loadLayersModel(modelUrl(backupKey(this.brain.modelKey)));
    this.online.dispose();
    this.target.dispose();
    this.online = restored;
    this.optimizer.dispose();
    this.optimizer = tf.train.adam(this.parameters.learningRate);
    this.target = createDqnModel(this.parameters);
    copyWeights(this.online, this.target);
    await this.save(true);
  }

  dispose(): void { this.online.dispose(); this.target.dispose(); this.optimizer.dispose(); }
}

export async function averageAgents(agents: DqnAgent[]): Promise<void> {
  if (agents.length < 2) throw new Error("Hive Mind Sync requires at least two active snakes.");
  await Promise.all(agents.map((agent) => agent.backup()));
  const allWeights = agents.map((agent) => agent.online.getWeights());
  const averaged = averageWeightTensors(allWeights);
  for (const agent of agents) {
    agent.online.setWeights(averaged);
    copyWeights(agent.online, agent.target);
    await agent.save(true);
  }
  averaged.forEach((weight) => weight.dispose());
}

export async function undoAverage(agents: DqnAgent[]): Promise<void> {
  await Promise.all(agents.map((agent) => agent.restoreBackup()));
}
