import { readFile } from "node:fs/promises";
import * as tf from "@tensorflow/tfjs";
import { loadModelFromBundle } from "../game/model-bundle";

async function main(): Promise<void> {
  const [bundlePath, observationPath] = process.argv.slice(2);
  if (!bundlePath || !observationPath) throw new Error("Usage: predict-bundle <bundle.json> <observation.json>");
  const bundle = JSON.parse(await readFile(bundlePath, "utf8"));
  const observation = JSON.parse(await readFile(observationPath, "utf8"));
  if (!Array.isArray(observation) || observation.length !== 228 || !observation.every(Number.isFinite)) {
    throw new Error("Prediction requires exactly 228 finite observation values.");
  }
  const { model } = await loadModelFromBundle(bundle, bundle.snakeId);
  const output = tf.tidy(() => model.predict(tf.tensor2d([observation], [1, 228])) as tf.Tensor);
  const values = Array.from(await output.data());
  output.dispose();
  model.dispose();
  process.stdout.write(JSON.stringify(values));
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
