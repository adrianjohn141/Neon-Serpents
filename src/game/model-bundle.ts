import * as tf from "@tensorflow/tfjs";
import { OBSERVATION_SIZE } from "./observation";
import type { Brain, BrainBenchmark, Hyperparameters } from "./types";

export const TRAINING_SPEC_VERSION = 3 as const;
export const MODEL_VERSION = 3 as const;
export const OBSERVATION_SPEC_HASH = "neon-serpents:v3:observation-228";

export type BrainBundle = {
  format: "neon-serpents-brain";
  formatVersion: 1;
  exportedAt: number;
  snakeId: string;
  modelVersion: typeof MODEL_VERSION;
  trainingSpecVersion: typeof TRAINING_SPEC_VERSION;
  observationSize: number;
  observationSpecHash: string;
  brain: Brain;
  hyperparameters: Hyperparameters;
  benchmark: BrainBenchmark | null;
  model: {
    topology: {};
    weightSpecs: tf.io.WeightsManifestEntry[];
    weightDataBase64: string;
    trainingConfig?: tf.io.TrainingConfig;
  };
};

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
  }
  return btoa(binary);
}

function base64ToArrayBuffer(value: string): ArrayBuffer {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes.buffer;
}

function singleWeightBuffer(weightData: tf.io.WeightData | undefined): ArrayBuffer {
  if (weightData instanceof ArrayBuffer) return weightData;
  if (Array.isArray(weightData)) {
    const size = weightData.reduce((sum, value) => sum + value.byteLength, 0);
    const joined = new Uint8Array(size);
    let offset = 0;
    for (const value of weightData) { joined.set(new Uint8Array(value), offset); offset += value.byteLength; }
    return joined.buffer;
  }
  throw new Error("The model did not provide serializable weight data.");
}

function hasNonFiniteNumber(value: unknown): boolean {
  if (typeof value === "number") return !Number.isFinite(value);
  if (Array.isArray(value)) return value.some(hasNonFiniteNumber);
  if (value && typeof value === "object") return Object.values(value).some(hasNonFiniteNumber);
  return false;
}

export async function createBrainBundle(
  model: tf.LayersModel,
  brain: Brain,
  hyperparameters: Hyperparameters,
): Promise<BrainBundle> {
  let captured: tf.io.ModelArtifacts | null = null;
  await model.save(tf.io.withSaveHandler(async (artifacts) => {
    captured = artifacts;
    return { modelArtifactsInfo: tf.io.getModelArtifactsInfoForJSON(artifacts) };
  }), { includeOptimizer: false });
  if (!captured) throw new Error("TensorFlow.js did not return model artifacts.");
  const artifacts = captured as tf.io.ModelArtifacts;
  if (!artifacts.modelTopology || artifacts.modelTopology instanceof ArrayBuffer || !artifacts.weightSpecs) {
    throw new Error("The exported model topology is not a TensorFlow.js LayersModel.");
  }
  return {
    format: "neon-serpents-brain",
    formatVersion: 1,
    exportedAt: Date.now(),
    snakeId: brain.snakeId,
    modelVersion: MODEL_VERSION,
    trainingSpecVersion: TRAINING_SPEC_VERSION,
    observationSize: OBSERVATION_SIZE,
    observationSpecHash: OBSERVATION_SPEC_HASH,
    brain: structuredClone(brain),
    hyperparameters: structuredClone(hyperparameters),
    benchmark: brain.lastBenchmark,
    model: {
      topology: artifacts.modelTopology,
      weightSpecs: artifacts.weightSpecs,
      weightDataBase64: arrayBufferToBase64(singleWeightBuffer(artifacts.weightData)),
      trainingConfig: artifacts.trainingConfig,
    },
  };
}

export function validateBrainBundle(value: unknown, expectedSnakeId?: string): BrainBundle {
  if (!value || typeof value !== "object") throw new Error("Brain bundle must be a JSON object.");
  const bundle = value as Partial<BrainBundle>;
  if (bundle.format !== "neon-serpents-brain" || bundle.formatVersion !== 1) throw new Error("Unsupported brain bundle format.");
  if (bundle.modelVersion !== MODEL_VERSION || bundle.trainingSpecVersion !== TRAINING_SPEC_VERSION) throw new Error("Brain bundle training version is incompatible.");
  if (bundle.observationSize !== OBSERVATION_SIZE || bundle.observationSpecHash !== OBSERVATION_SPEC_HASH) throw new Error("Brain bundle observation contract is incompatible.");
  if (!bundle.snakeId || (expectedSnakeId && bundle.snakeId !== expectedSnakeId)) throw new Error("Brain bundle belongs to a different snake.");
  if (!bundle.brain || bundle.brain.snakeId !== bundle.snakeId || !bundle.hyperparameters) throw new Error("Brain bundle metadata is incomplete.");
  if (hasNonFiniteNumber(bundle.brain) || hasNonFiniteNumber(bundle.hyperparameters) || hasNonFiniteNumber(bundle.benchmark)) {
    throw new Error("Brain bundle metadata contains non-finite numbers.");
  }
  if (!bundle.model?.topology || !Array.isArray(bundle.model.weightSpecs) || typeof bundle.model.weightDataBase64 !== "string") throw new Error("Brain bundle model artifacts are incomplete.");
  const weightData = base64ToArrayBuffer(bundle.model.weightDataBase64);
  const expectedBytes = bundle.model.weightSpecs.reduce((sum, spec) => {
    const values = spec.shape.reduce((product, size) => product * size, 1);
    const bytes = spec.dtype === "complex64" ? 8 : spec.dtype === "float32" || spec.dtype === "int32" ? 4 : spec.dtype === "bool" ? 1 : 0;
    return sum + values * bytes;
  }, 0);
  if (!expectedBytes || weightData.byteLength !== expectedBytes) throw new Error("Brain bundle tensor data has an invalid size.");
  const floats = new Float32Array(weightData);
  if (bundle.model.weightSpecs.some((spec) => spec.dtype === "float32") && !floats.every(Number.isFinite)) throw new Error("Brain bundle contains non-finite weights.");
  return bundle as BrainBundle;
}

export async function loadModelFromBundle(bundleValue: unknown, expectedSnakeId?: string): Promise<{ bundle: BrainBundle; model: tf.LayersModel }> {
  const bundle = validateBrainBundle(bundleValue, expectedSnakeId);
  const model = await tf.loadLayersModel(tf.io.fromMemory({
    modelTopology: bundle.model.topology,
    weightSpecs: bundle.model.weightSpecs,
    weightData: base64ToArrayBuffer(bundle.model.weightDataBase64),
    trainingConfig: bundle.model.trainingConfig,
  }));
  if (model.inputs[0]?.shape.at(-1) !== OBSERVATION_SIZE || model.outputs[0]?.shape.at(-1) !== 3) {
    model.dispose();
    throw new Error("Brain bundle model shape is incompatible.");
  }
  return { bundle, model };
}
