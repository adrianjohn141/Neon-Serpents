import { readFile } from "node:fs/promises";
import * as tf from "@tensorflow/tfjs";
import { loadModelFromBundle } from "../game/model-bundle";
import { runEvaluation } from "../game/evaluation";
import type { Brain, RelativeAction, SnakeDefinition } from "../game/types";

function requiredArgument(name: string): string {
  const index = process.argv.indexOf(`--${name}`);
  if (index < 0 || !process.argv[index + 1]) throw new Error(`Missing --${name} <bundle> argument.`);
  return process.argv[index + 1];
}

async function loadAgent(path: string, id: string, name: string) {
  const parsed = JSON.parse(await readFile(path, "utf8")) as unknown;
  const { bundle, model } = await loadModelFromBundle(parsed);
  const brain: Brain = { ...bundle.brain, snakeId: id };
  return {
    brain,
    model,
    definition: { id, name, color: id === "candidate" ? "#68f7c1" : "#ff6b7a", accent: "#ffffff" } satisfies SnakeDefinition,
    chooseAction(observation: ArrayLike<number>, _explore: boolean, safeActions?: RelativeAction[]): RelativeAction {
      return tf.tidy(() => {
        const prediction = model.predict(tf.tensor2d([Array.from(observation)])) as tf.Tensor;
        const values = Array.from(prediction.dataSync());
        if (safeActions?.length) for (let index = 0; index < values.length; index += 1) if (!safeActions.includes(index as RelativeAction)) values[index] = -Infinity;
        return values.indexOf(Math.max(...values)) as RelativeAction;
      });
    },
  };
}

async function main(): Promise<void> {
  await import("@tensorflow/tfjs-node");
  const candidate = await loadAgent(requiredArgument("candidate"), "candidate", "Candidate");
  const baseline = await loadAgent(requiredArgument("baseline"), "baseline", "Baseline");
  const matchIndex = process.argv.indexOf("--matches");
  const matches = Math.max(1, Number(matchIndex >= 0 ? process.argv[matchIndex + 1] : 200) || 200);
  const report = await runEvaluation([candidate, baseline], [candidate.definition, baseline.definition], matches, 42);
  console.log(JSON.stringify(report, null, 2));
  candidate.model.dispose(); baseline.model.dispose();
}

main().catch((error) => {
  const reason = error instanceof Error ? error.message : String(error);
  console.error(`Desktop TensorFlow.js benchmark could not start: ${reason}`);
  console.error("Install a compatible @tensorflow/tfjs-node native binding for this Node.js/Windows runtime; browser evaluation remains available.");
  process.exitCode = 1;
});
