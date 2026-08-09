import * as tf from "@tensorflow/tfjs";
import { beforeAll, describe, expect, it } from "vitest";
import { createBrain, DEFAULT_HYPERPARAMETERS } from "./ai";
import { createDqnModel } from "./dqn";
import { createBrainBundle, loadModelFromBundle, validateBrainBundle } from "./model-bundle";
import { OBSERVATION_SIZE } from "./observation";

describe("portable brain bundles", () => {
  beforeAll(async () => { await tf.setBackend("cpu"); await tf.ready(); });

  it("round-trips browser and desktop model artifacts without changing Q-values", async () => {
    const model = createDqnModel(DEFAULT_HYPERPARAMETERS);
    const input = tf.linspace(0, 1, OBSERVATION_SIZE).reshape([1, OBSERVATION_SIZE]);
    const expected = Array.from((model.predict(input) as tf.Tensor).dataSync());
    const bundle = await createBrainBundle(model, createBrain("nova"), DEFAULT_HYPERPARAMETERS);
    const loaded = await loadModelFromBundle(bundle, "nova");
    const actual = Array.from((loaded.model.predict(input) as tf.Tensor).dataSync());
    actual.forEach((value, index) => expect(value).toBeCloseTo(expected[index], 5));
    expect(() => validateBrainBundle(bundle, "ember")).toThrow(/different snake/);
    input.dispose(); model.dispose(); loaded.model.dispose();
  });

  it("rejects malformed weight payloads", async () => {
    const model = createDqnModel(DEFAULT_HYPERPARAMETERS);
    const bundle = await createBrainBundle(model, createBrain("nova"), DEFAULT_HYPERPARAMETERS);
    expect(() => validateBrainBundle({ ...bundle, model: { ...bundle.model, weightDataBase64: "AAAA" } })).toThrow(/invalid size/);
    expect(() => validateBrainBundle({ ...bundle, hyperparameters: { ...bundle.hyperparameters, learningRate: Number.NaN } })).toThrow(/non-finite/);
    model.dispose();
  });
});
