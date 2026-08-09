import { describe, expect, it } from "vitest";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import * as tf from "@tensorflow/tfjs";
import { DEFAULT_HYPERPARAMETERS } from "../game/ai";
import { createDqnModel } from "../game/dqn";
import { loadEvaluationBundle, loadRoster } from "./evaluate-release";

async function legacyBundle() {
  const model = createDqnModel(DEFAULT_HYPERPARAMETERS, 159);
  let captured: tf.io.ModelArtifacts | null = null;
  await model.save(tf.io.withSaveHandler(async (artifacts) => {
    captured = artifacts;
    return { modelArtifactsInfo: tf.io.getModelArtifactsInfoForJSON(artifacts) };
  }));
  model.dispose();
  const artifacts = captured as unknown as tf.io.ModelArtifacts;
  const buffers = artifacts.weightData instanceof ArrayBuffer ? [artifacts.weightData] : artifacts.weightData as ArrayBuffer[];
  const bytes = new Uint8Array(buffers.reduce((sum, value) => sum + value.byteLength, 0));
  let offset = 0;
  for (const value of buffers) { bytes.set(new Uint8Array(value), offset); offset += value.byteLength; }
  return {
    format: "neon-serpents-brain", formatVersion: 1, snakeId: "nova",
    modelVersion: 2, trainingSpecVersion: 2, observationSize: 159,
    observationSpecHash: "neon-serpents:v2:observation-159",
    model: {
      topology: artifacts.modelTopology,
      weightSpecs: artifacts.weightSpecs,
      weightDataBase64: Buffer.from(bytes).toString("base64"),
    },
  };
}

describe("release evaluator compatibility", () => {
  it("runs a validated v2 policy against the preserved first 159 v3 observations", async () => {
    const bundle = await legacyBundle();
    const agent = await loadEvaluationBundle(bundle, "baseline-nova", "nova");
    expect(agent.observationSize).toBe(159);
    expect([0, 1, 2]).toContain(agent.chooseAction(new Float32Array(228), false, [0, 1, 2]));
    agent.dispose();
  });

  it("rejects a malformed legacy observation contract", async () => {
    const bundle = await legacyBundle();
    await expect(loadEvaluationBundle({ ...bundle, observationSpecHash: "wrong" }, "baseline-nova", "nova"))
      .rejects.toThrow("incompatible contract");
  });

  it("loads a future five-snake roster and rejects malformed metadata", async () => {
    const directory = await mkdtemp(join(tmpdir(), "neon-roster-"));
    try {
      const roster = [0, 1, 2, 3, 4].map((index) => ({ id: `snake-${index}`, name: `Snake ${index}`, color: "#68f7c1", accent: "#ffffff" }));
      await writeFile(join(directory, "roster.json"), JSON.stringify(roster));
      expect((await loadRoster(directory)).map((entry) => entry.id)).toHaveLength(5);
      await writeFile(join(directory, "roster.json"), "{bad");
      await expect(loadRoster(directory)).rejects.toThrow("Invalid roster metadata");
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });
});
