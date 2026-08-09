/// <reference lib="webworker" />
import { effectiveHyperparameters } from "../game/ai";
import { DEFAULT_CURRICULUM } from "../game/constants";
import {
  runBattleEpisode, runHazardEpisode, runObjectiveEpisode, runPowerUpEpisode, runSafeZoneEpisode,
  runSeriesEpisode, runSurvivalEpisode, selectTrainingScenario, type CurriculumAgent,
} from "../game/curriculum";
import { averageAgents, DqnAgent, initializeTensorFlow, undoAverage } from "../game/dqn";
import { runEvaluation } from "../game/evaluation";
import { ScriptedOpponent } from "../game/opponents";
import type { CurriculumConfig, Hyperparameters, PersistedProfile, RelativeAction, SnakeDefinition } from "../game/types";
import type { WorkerRequest, WorkerResponse } from "../game/worker-protocol";

const scope = self as DedicatedWorkerGlobalScope;
const agents = new Map<string, DqnAgent>();
let profiles: PersistedProfile[] = [];
let globalParameters: Hyperparameters;
let globalCurriculum: CurriculumConfig = { ...DEFAULT_CURRICULUM };
let fastRunning = false;
let totalEpisodes = 0;
let totalSteps = 0;
let fastStartedAt = 0;
let operation = Promise.resolve();
let fastPromise: Promise<void> | null = null;
let battleEpisodes = 0;
const opponentPool = new Map<string, DqnAgent[]>();

const post = (message: WorkerResponse) => scope.postMessage(message);
const brains = () => [...agents.values()].map((agent) => agent.brain);
const fail = (error: unknown) => post({ type: "error", message: error instanceof Error ? error.message : "AI worker operation failed." });
const definitionOf = (profile: PersistedProfile): SnakeDefinition => ({ id: profile.snakeId, name: profile.name, color: profile.color, accent: profile.accent });
const activeDefinitions = () => profiles.filter((entry) => entry.active).map(definitionOf);

async function initialize(request: Extract<WorkerRequest, { type: "init" }>): Promise<void> {
  fastRunning = false;
  if (fastPromise) await fastPromise;
  agents.forEach((agent) => agent.dispose());
  opponentPool.forEach((snapshots) => snapshots.forEach((agent) => agent.dispose()));
  opponentPool.clear();
  agents.clear();
  battleEpisodes = 0;
  profiles = request.profiles;
  globalParameters = request.hyperparameters;
  globalCurriculum = request.curriculum;
  await initializeTensorFlow();
  for (const profile of profiles.filter((entry) => entry.active)) {
    const parameters = effectiveHyperparameters(globalParameters, profile.hyperparameterOverrides);
    agents.set(profile.snakeId, await DqnAgent.load(profile.brain, parameters));
  }
  post({ type: "ready", brains: brains(), backend: "cpu" });
}

async function checkpoint(): Promise<void> {
  await Promise.all([...agents.values()].map((agent) => agent.save(true)));
  post({ type: "saved", brains: brains() });
}

async function runFast(snakes: SnakeDefinition[]): Promise<void> {
  fastStartedAt = performance.now();
  totalEpisodes = 0;
  totalSteps = 0;
  const definitions = new Map(snakes.map((snake) => [snake.id, snake]));
  const list = [...agents.values()];
  if (!list.length) { post({ type: "stopped", brains: brains() }); return; }
  const battleSize = Math.max(2, Math.min(globalCurriculum.battleSize, list.length));
  let cursor = 0;
  fastRunning = true;
  while (fastRunning) {
    const primary = list[cursor % list.length];
    const scenario = selectTrainingScenario(primary.brain, globalCurriculum);
    if (!["battle", "series"].includes(scenario)) {
      cursor += 1;
      const definition = definitions.get(primary.brain.snakeId);
      if (!definition) continue;
      const episodeSeed = (Date.now() + totalEpisodes * 7919) >>> 0;
      const labs = { survival: runSurvivalEpisode, powerup: runPowerUpEpisode, safezone: runSafeZoneEpisode, hazard: runHazardEpisode, objective: runObjectiveEpisode } as const;
      const result = await labs[scenario as keyof typeof labs](primary, definition, episodeSeed);
      totalEpisodes += 1;
      totalSteps += result.ticks;
      await primary.save();
    } else {
      const currentParticipants: DqnAgent[] = [];
      for (let offset = 0; offset < battleSize; offset += 1) currentParticipants.push(list[(cursor + offset) % list.length]);
      cursor += 1;
      const matchupRoll = Math.random();
      const participants: CurriculumAgent[] = [currentParticipants[0]];
      const learningIds = new Set<string>([currentParticipants[0].brain.snakeId]);
      for (const current of currentParticipants.slice(1)) {
        if (matchupRoll < globalCurriculum.currentOpponentRatio) {
          participants.push(current); learningIds.add(current.brain.snakeId);
        } else if (matchupRoll < globalCurriculum.currentOpponentRatio + globalCurriculum.historicalOpponentRatio) {
          const snapshots = opponentPool.get(current.brain.snakeId) ?? [];
          participants.push(snapshots[Math.floor(Math.random() * snapshots.length)] ?? current);
          if (!snapshots.length) learningIds.add(current.brain.snakeId);
        } else {
          const styles = ["aggressive", "defensive", "food", "powerup", "trap", "leaderHunter"] as const;
          participants.push(new ScriptedOpponent(current.brain.snakeId, styles[battleEpisodes % styles.length]));
        }
      }
      const matchDefs = participants.map((agent) => definitions.get(agent.brain.snakeId)).filter((def): def is SnakeDefinition => Boolean(def));
      if (matchDefs.length < 2) continue;
      const result = scenario === "series"
        ? await runSeriesEpisode(participants, matchDefs, (Date.now() + totalEpisodes * 7919 + 13) >>> 0, learningIds)
        : await runBattleEpisode(participants, matchDefs, (Date.now() + totalEpisodes * 7919 + 13) >>> 0, 2000, learningIds);
      battleEpisodes += 1;
      totalEpisodes += 1;
      totalSteps += result.ticks;
      await Promise.all(currentParticipants.filter((agent) => learningIds.has(agent.brain.snakeId)).map((agent) => agent.save()));
      if (battleEpisodes % globalCurriculum.checkpointIntervalEpisodes === 0) {
        for (const current of list) {
          const snapshots = opponentPool.get(current.brain.snakeId) ?? [];
          snapshots.push(current.cloneFrozen());
          while (snapshots.length > globalCurriculum.checkpointRetention) snapshots.shift()?.dispose();
          opponentPool.set(current.brain.snakeId, snapshots);
        }
      }
    }
    if (totalEpisodes % Math.max(1, list.length) === 0) {
      post({ type: "progress", scenario, brains: brains(), totalEpisodes, totalSteps, elapsedMs: performance.now() - fastStartedAt });
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }
  await Promise.all([...agents.values()].map((agent) => agent.save(true)));
  post({ type: "stopped", brains: brains() });
}
async function handle(request: WorkerRequest): Promise<void> {
  switch (request.type) {
    case "init": await initialize(request); break;
    case "act": {
      const actions: Record<string, RelativeAction> = {};
      for (const [snakeId, payload] of Object.entries(request.observations)) {
        const agent = agents.get(snakeId);
        if (agent) actions[snakeId] = agent.chooseAction(payload.data, request.explore, payload.safeActions);
      }
      post({ type: "actions", requestId: request.requestId, actions });
      break;
    }
    case "observe":
      for (const [snakeId, experience] of Object.entries(request.experiences)) {
        const agent = agents.get(snakeId);
        const context = request.contexts?.[snakeId];
        if (agent) {
          agent.remember(experience, context?.scenario ?? "battle", context);
          await agent.train();
        }
      }
      break;
    case "episode":
      for (const [snakeId, result] of Object.entries(request.outcomes)) {
        const agent = agents.get(snakeId);
        if (agent) { agent.finishEpisode(result.score, result.food, result.won); await agent.save(); }
      }
      post({ type: "brains", brains: brains() });
      break;
    case "startFast":
      if (!fastRunning) {
        fastPromise = runFast(request.snakes).catch((error) => {
          fastRunning = false;
          fail(error);
          post({ type: "stopped", brains: brains() });
        }).finally(() => { fastPromise = null; });
      }
      break;
    case "evaluate":
      if (fastPromise) await fastPromise;
      {
        const report = await runEvaluation([...agents.values()], activeDefinitions(), request.runs, Date.now() >>> 0);
        for (const stats of report.snakes) agents.get(stats.id)?.recordBenchmark(stats, report);
        await Promise.all([...agents.values()].map((agent) => agent.save(true)));
        post({ type: "evalReport", report });
        post({ type: "saved", brains: brains() });
      }
      break;
    case "stop": fastRunning = false; break;
    case "checkpoint":
      if (fastPromise) await fastPromise;
      await checkpoint();
      break;
    case "updateParameters":
      if (fastPromise) await fastPromise;
      profiles = request.profiles;
      globalParameters = request.hyperparameters;
      for (const profile of profiles.filter((entry) => entry.active)) {
        agents.get(profile.snakeId)?.updateParameters(effectiveHyperparameters(globalParameters, profile.hyperparameterOverrides));
      }
      await checkpoint();
      break;
    case "reset":
      if (fastPromise) await fastPromise;
      for (const id of request.snakeIds) await agents.get(id)?.reset();
      post({ type: "brains", brains: brains() });
      break;
    case "hiveSync":
      if (fastPromise) await fastPromise;
      await averageAgents([...agents.values()]); post({ type: "hiveComplete", brains: brains(), canUndo: true });
      break;
    case "undoHive":
      if (fastPromise) await fastPromise;
      await undoAverage([...agents.values()]); post({ type: "hiveComplete", brains: brains(), canUndo: false });
      break;
    case "trainAfterMatch":
      if (fastPromise) await fastPromise;
      for (let step = 0; step < request.steps; step += 1) for (const agent of agents.values()) await agent.train(true);
      await checkpoint();
      break;
    case "exportBundle": {
      try {
        const agent = agents.get(request.snakeId);
        if (!agent) throw new Error("Active snake brain not found.");
        post({ type: "bundle", requestId: request.requestId, bundle: await agent.exportBundle() });
      } catch (error) {
        post({ type: "error", requestId: request.requestId, message: error instanceof Error ? error.message : "Brain export failed." });
      }
      break;
    }
    case "importBundle": {
      try {
        const agent = agents.get(request.snakeId);
        if (!agent) throw new Error("Active snake brain not found.");
        await agent.importBundle(request.bundle);
        post({ type: "bundleImported", requestId: request.requestId, brains: brains() });
      } catch (error) {
        post({ type: "error", requestId: request.requestId, message: error instanceof Error ? error.message : "Brain import failed." });
      }
      break;
    }
    case "useProductionBundles": {
      try {
        for (const [snakeId, bundle] of Object.entries(request.bundles)) {
          const agent = agents.get(snakeId);
          if (agent) await agent.useRuntimeBundle(bundle);
        }
        post({ type: "productionBundlesLoaded", requestId: request.requestId });
      } catch (error) {
        post({ type: "error", requestId: request.requestId, message: error instanceof Error ? error.message : "Production brain loading failed." });
      }
      break;
    }
  }
}

scope.onmessage = (event: MessageEvent<WorkerRequest>) => {
  if (event.data.type === "stop") { fastRunning = false; return; }
  operation = operation.then(() => handle(event.data)).catch(fail);
};
