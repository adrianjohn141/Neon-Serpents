import { readFile, writeFile, mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import * as tf from "@tensorflow/tfjs";
import { createBrain, DEFAULT_HYPERPARAMETERS } from "../game/ai";
import { createDqnModel } from "../game/dqn";
import { createBrainBundle } from "../game/model-bundle";

type InputTensor = { name: string; shape: number[]; values: number[] };
type InputBrain = {
  snakeId: string; environmentSteps: number; learningSteps: number; episodes?: number; epsilon: number; tensors: InputTensor[];
  trainingSpecVersion?: 2 | 3;
  scenarioSteps?: Record<string, number>; powerUpOpportunities?: number; powerUpsClaimed?: number; powerUpApproachMisses?: number;
  rareFoodClaims?: number; objectiveCaptures?: number; bountyKills?: number; hazardDeaths?: number; zoneDeaths?: number;
};

async function createLegacyBundle(model: tf.LayersModel, source: InputBrain) {
  let captured: tf.io.ModelArtifacts | null = null;
  await model.save(tf.io.withSaveHandler(async (artifacts) => {
    captured = artifacts;
    return { modelArtifactsInfo: tf.io.getModelArtifactsInfoForJSON(artifacts) };
  }), { includeOptimizer: false });
  if (!captured) throw new Error("TensorFlow.js did not return legacy model artifacts.");
  const artifacts = captured as tf.io.ModelArtifacts;
  const chunks = artifacts.weightData instanceof ArrayBuffer
    ? [Buffer.from(artifacts.weightData)]
    : (artifacts.weightData as ArrayBuffer[]).map((value) => Buffer.from(value));
  const data = Buffer.concat(chunks);
  const brain = {
    ...createBrain(source.snakeId),
    modelVersion: 2,
    trainingSpecVersion: 2,
    observationSize: 159,
    environmentSteps: source.environmentSteps,
    learningSteps: source.learningSteps,
    episodes: source.episodes ?? 0,
    epsilon: source.epsilon,
    scenarioSteps: { ...createBrain(source.snakeId).scenarioSteps, ...source.scenarioSteps },
  };
  return {
    format: "neon-serpents-brain",
    formatVersion: 1,
    exportedAt: Date.now(),
    snakeId: source.snakeId,
    modelVersion: 2,
    trainingSpecVersion: 2,
    observationSize: 159,
    observationSpecHash: "neon-serpents:v2:observation-159",
    brain,
    hyperparameters: DEFAULT_HYPERPARAMETERS,
    benchmark: null,
    model: {
      topology: artifacts.modelTopology,
      weightSpecs: artifacts.weightSpecs,
      weightDataBase64: data.toString("base64"),
      trainingConfig: artifacts.trainingConfig,
    },
  };
}

async function main(): Promise<void> {
  const [inputPath, outputPath] = process.argv.slice(2);
  if (!inputPath || !outputPath) throw new Error("Usage: bundle-exporter <input.json> <output-directory>");
  const input = JSON.parse(await readFile(inputPath, "utf8")) as { brains: InputBrain[] };
  await mkdir(outputPath, { recursive: true });
  for (const source of input.brains) {
    const legacy = source.trainingSpecVersion === 2;
    const model = createDqnModel(DEFAULT_HYPERPARAMETERS, legacy ? 159 : undefined);
    const byName = new Map(source.tensors.map((tensor) => [tensor.name, tensor]));
    const matrix = (name: string, transpose = false) => {
      const value = byName.get(name)!;
      const tensor = tf.tensor(value.values, value.shape);
      if (!transpose) return tensor;
      const result = tensor.transpose(); tensor.dispose(); return result;
    };
    const weights = [
      matrix("fc1.weight", true), matrix("fc1.bias"), matrix("fc2.weight", true),
      matrix("fc2.bias"), matrix("out.weight", true), matrix("out.bias"),
    ];
    model.setWeights(weights); weights.forEach((weight) => weight.dispose());
    const brain = {
      ...createBrain(source.snakeId),
      environmentSteps: source.environmentSteps,
      learningSteps: source.learningSteps,
      episodes: source.episodes ?? 0,
      generation: source.episodes ?? 0,
      epsilon: source.epsilon,
      scenarioSteps: { ...createBrain(source.snakeId).scenarioSteps, ...source.scenarioSteps },
      powerUpOpportunities: source.powerUpOpportunities ?? 0,
      powerUpsClaimed: source.powerUpsClaimed ?? 0,
      powerUpApproachMisses: source.powerUpApproachMisses ?? 0,
      rareFoodClaims: source.rareFoodClaims ?? 0,
      objectiveCaptures: source.objectiveCaptures ?? 0,
      bountyKills: source.bountyKills ?? 0,
      hazardDeaths: source.hazardDeaths ?? 0,
      zoneDeaths: source.zoneDeaths ?? 0,
    };
    const bundle = legacy ? await createLegacyBundle(model, source) : await createBrainBundle(model, brain, DEFAULT_HYPERPARAMETERS);
    await writeFile(resolve(outputPath, `${source.snakeId}-${legacy ? "v2" : "v3"}.nsbrain.json`), JSON.stringify(bundle), "utf8");
    model.dispose();
  }
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
