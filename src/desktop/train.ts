import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { createBrain, DEFAULT_HYPERPARAMETERS } from "../game/ai";
import { DEFAULT_CURRICULUM, SNAKES } from "../game/constants";
import {
  runBattleEpisode, runHazardEpisode, runObjectiveEpisode, runPowerUpEpisode, runSafeZoneEpisode,
  runSeriesEpisode, runSurvivalEpisode, selectTrainingScenario,
} from "../game/curriculum";
import { DqnAgent } from "../game/dqn";
import { ScriptedOpponent } from "../game/opponents";
import type { CurriculumAgent } from "../game/curriculum";

function argument(name: string, fallback: string): string {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 && process.argv[index + 1] ? process.argv[index + 1] : fallback;
}

async function main(): Promise<void> {
  await import("@tensorflow/tfjs-node");
  const stepBudget = Math.max(1, Number(argument("steps", "1000000")) || 1_000_000);
  const seed = Number(argument("seed", "42")) >>> 0;
  const output = resolve(argument("out", `artifacts/training/run-${Date.now()}`));
  const parameters = { ...DEFAULT_HYPERPARAMETERS, replayBufferSize: 50_000 };
  const agents = SNAKES.map((definition) => DqnAgent.createFresh(createBrain(definition.id, parameters), parameters));
  let totalSteps = 0;
  let episodes = 0;
  let cursor = 0;
  let battleEpisodes = 0;
  let randomState = seed || 1;
  const random = () => {
    randomState = (Math.imul(randomState, 1_664_525) + 1_013_904_223) >>> 0;
    return randomState / 0x1_0000_0000;
  };
  const opponentPool = new Map<string, DqnAgent[]>();
  const startedAt = Date.now();
  while (totalSteps < stepBudget) {
    const primary = agents[cursor % agents.length];
    const scenario = selectTrainingScenario(primary.brain, DEFAULT_CURRICULUM, random);
    const episodeSeed = (seed + episodes * 7919) >>> 0;
    if (["survival", "powerup", "safezone", "hazard", "objective"].includes(scenario)) {
      const definition = SNAKES.find((entry) => entry.id === primary.brain.snakeId)!;
      const labs = { survival: runSurvivalEpisode, powerup: runPowerUpEpisode, safezone: runSafeZoneEpisode, hazard: runHazardEpisode, objective: runObjectiveEpisode } as const;
      totalSteps += (await labs[scenario as keyof typeof labs](primary, definition, episodeSeed)).ticks;
      cursor += 1;
    } else {
      const current = Array.from({ length: Math.min(DEFAULT_CURRICULUM.battleSize, agents.length) }, (_, offset) => agents[(cursor + offset) % agents.length]);
      const participants: CurriculumAgent[] = [current[0]];
      const learningIds = new Set([current[0].brain.snakeId]);
      for (const opponent of current.slice(1)) {
        const roll = random();
        if (roll < DEFAULT_CURRICULUM.currentOpponentRatio) {
          participants.push(opponent);
          learningIds.add(opponent.brain.snakeId);
        } else if (roll < DEFAULT_CURRICULUM.currentOpponentRatio + DEFAULT_CURRICULUM.historicalOpponentRatio) {
          const snapshots = opponentPool.get(opponent.brain.snakeId) ?? [];
          participants.push(snapshots[Math.floor(random() * snapshots.length)] ?? opponent);
          if (!snapshots.length) learningIds.add(opponent.brain.snakeId);
        } else {
          const styles = ["aggressive", "defensive", "food", "powerup", "trap", "leaderHunter"] as const;
          participants.push(new ScriptedOpponent(opponent.brain.snakeId, styles[battleEpisodes % styles.length]));
        }
      }
      const definitions = participants.map((agent) => SNAKES.find((entry) => entry.id === agent.brain.snakeId)!);
      totalSteps += (scenario === "series"
        ? await runSeriesEpisode(participants, definitions, episodeSeed, learningIds)
        : await runBattleEpisode(participants, definitions, episodeSeed, 2_000, learningIds)).ticks;
      battleEpisodes += 1;
      if (battleEpisodes % DEFAULT_CURRICULUM.checkpointIntervalEpisodes === 0) {
        for (const agent of agents) {
          const snapshots = opponentPool.get(agent.brain.snakeId) ?? [];
          snapshots.push(agent.cloneFrozen());
          while (snapshots.length > DEFAULT_CURRICULUM.checkpointRetention) snapshots.shift()?.dispose();
          opponentPool.set(agent.brain.snakeId, snapshots);
        }
      }
      cursor += 1;
    }
    episodes += 1;
    if (episodes % 25 === 0) process.stdout.write(`\r${totalSteps.toLocaleString()} / ${stepBudget.toLocaleString()} steps · ${episodes.toLocaleString()} episodes`);
  }
  await mkdir(output, { recursive: true });
  const bundles = await Promise.all(agents.map((agent) => agent.exportBundle()));
  await Promise.all(bundles.map((bundle) => writeFile(resolve(output, `${bundle.snakeId}-v3.nsbrain.json`), JSON.stringify(bundle), "utf8")));
  await writeFile(resolve(output, "run.json"), JSON.stringify({ seed, stepBudget, totalSteps, episodes, startedAt, finishedAt: Date.now(), parameters, curriculum: DEFAULT_CURRICULUM }, null, 2), "utf8");
  agents.forEach((agent) => agent.dispose());
  opponentPool.forEach((snapshots) => snapshots.forEach((agent) => agent.dispose()));
  process.stdout.write(`\nSaved ${bundles.length} validated brain bundles to ${output}\n`);
}

main().catch((error) => {
  const reason = error instanceof Error ? error.message : String(error);
  console.error(`Desktop TensorFlow.js training could not start: ${reason}`);
  console.error("Install a compatible @tensorflow/tfjs-node native binding for this Node.js/Windows runtime; browser training remains available.");
  process.exitCode = 1;
});
