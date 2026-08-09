import * as tf from "@tensorflow/tfjs";
import { beforeAll, describe, expect, it } from "vitest";
import { createBrain, DEFAULT_HYPERPARAMETERS, OBSERVATION_SIZE } from "./ai";
import { averageWeightTensors, createDqnModel, DqnAgent, synchronizeTarget } from "./dqn";

describe("DQN model", () => {
  beforeAll(async () => { await tf.setBackend("cpu"); await tf.ready(); });

  it("uses the versioned 228-128-64-3 architecture", () => {
    const model = createDqnModel(DEFAULT_HYPERPARAMETERS);
    expect(model.inputs[0].shape.at(-1)).toBe(OBSERVATION_SIZE);
    expect(model.layers.map((layer) => layer.outputShape)).toEqual([[null, 128], [null, 64], [null, 3]]);
    model.dispose();
  });

  it("synchronizes target weights exactly", () => {
    const online = createDqnModel(DEFAULT_HYPERPARAMETERS);
    const target = createDqnModel(DEFAULT_HYPERPARAMETERS);
    const source = online.getWeights().map((weight) => tf.onesLike(weight));
    online.setWeights(source); source.forEach((weight) => weight.dispose());
    synchronizeTarget(online, target);
    expect(target.getWeights().every((weight) => [...weight.dataSync()].every((value) => value === 1))).toBe(true);
    expect(online.getWeights().every((weight) => [...weight.dataSync()].every((value) => value === 1))).toBe(true);
    online.dispose(); target.dispose();
  });

  it("averages corresponding model tensors", () => {
    const first = [tf.tensor1d([0, 2]), tf.scalar(4)];
    const second = [tf.tensor1d([2, 4]), tf.scalar(8)];
    const average = averageWeightTensors([first, second]);
    expect([...average[0].dataSync()]).toEqual([1, 3]);
    expect(average[1].dataSync()[0]).toBe(6);
    [...first, ...second, ...average].forEach((tensor) => tensor.dispose());
  });

  it("does not leak tensors during repeated inference", () => {
    const model = createDqnModel(DEFAULT_HYPERPARAMETERS);
    const baseline = tf.memory().numTensors;
    for (let index = 0; index < 100; index += 1) tf.tidy(() => {
      const output = model.predict(tf.zeros([1, OBSERVATION_SIZE])) as tf.Tensor;
      return output.dataSync()[0];
    });
    expect(tf.memory().numTensors).toBe(baseline);
    model.dispose();
  });

  it("performs a prioritized, importance-weighted training update", async () => {
    const parameters = { ...DEFAULT_HYPERPARAMETERS, batchSize: 16, warmupTransitions: 0 };
    const agent = DqnAgent.createFresh(createBrain("weighted", parameters), parameters);
    for (let index = 0; index < 16; index += 1) {
      agent.remember({
        state: new Float32Array(OBSERVATION_SIZE).fill(index / 16),
        action: (index % 3) as 0 | 1 | 2,
        reward: index % 2,
        nextState: new Float32Array(OBSERVATION_SIZE).fill((index + 1) / 16),
        terminal: true,
      });
    }
    const loss = await agent.train(true);
    expect(loss).not.toBeNull();
    expect(Number.isFinite(loss)).toBe(true);
    expect(agent.brain.learningSteps).toBe(1);
    agent.dispose();
  });
});
