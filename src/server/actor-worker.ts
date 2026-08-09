import { parentPort, workerData } from "node:worker_threads";
import { createBrain, DEFAULT_HYPERPARAMETERS, explorationRate, completeEpisode } from "../game/ai";
import { DEFAULT_CURRICULUM, SNAKES } from "../game/constants";
import {
  runBattleEpisode, runHazardEpisode, runObjectiveEpisode, runPowerUpEpisode, runSafeZoneEpisode,
  runSeriesEpisode, runSurvivalEpisode, selectTrainingScenario, type CurriculumAgent,
} from "../game/curriculum";
import { ScriptedOpponent } from "../game/opponents";
import type { Brain, Experience, PowerUpKind, RelativeAction, RewardBreakdown, SnakeDefinition, TrainingDeathCause, TrainingScenario } from "../game/types";
import { DensePolicy, type WirePolicy } from "./policy";
import { selectParticipantRoster } from "./training-participants";

type StartMessage = { type: "start"; experimentId: string; seed: number; policies: WirePolicy[]; trainingSpecVersion?: number; observationSize?: number; roster?: SnakeDefinition[]; battleSize?: number };
type PolicyMessage = { type: "policy"; policy: WirePolicy };
type FlowMessage = { type: "flow"; enabled: boolean };
type StopMessage = { type: "stop" };
type Incoming = StartMessage | PolicyMessage | FlowMessage | StopMessage;
type TransitionWire = {
  snakeId: string; scenario: TrainingScenario; state: Uint8Array; action: number; reward: number; nextState: Uint8Array;
  terminal: boolean; opportunity: boolean; claimed: boolean; approachMiss: boolean; episodeId: string;
  powerUpKind: PowerUpKind | ""; rewardComponents?: RewardBreakdown;
  rareFoodClaimed: boolean; objectiveCaptured: boolean; bountyKill: boolean; deathCause: TrainingDeathCause | "";
};

if (!parentPort) throw new Error("Actor worker requires a parent port.");

const actorIndex = Number(workerData.actorIndex);
const flowSignal = new Int32Array(workerData.flowBuffer as SharedArrayBuffer);
let running = false;
let experimentId = "";
let episodeCounter = 0;
let randomState = 1;
let trainingSpecVersion: 2 | 3 = 3;
let observationSize: 159 | 228 = 228;
let activeRoster: SnakeDefinition[] = SNAKES;
let battleSize = DEFAULT_CURRICULUM.battleSize;
const currentPolicies = new Map<string, DensePolicy>();
const histories = new Map<string, DensePolicy[]>();
const transitions: TransitionWire[] = [];

const random = () => {
  randomState = (Math.imul(randomState, 1_664_525) + 1_013_904_223) >>> 0;
  return randomState / 0x1_0000_0000;
};

function bytes(values: ArrayLike<number>): Uint8Array {
  const source = values instanceof Float32Array ? values : Float32Array.from(values);
  const array = source.length === observationSize ? source : source.slice(0, observationSize);
  return new Uint8Array(array.buffer.slice(array.byteOffset, array.byteOffset + array.byteLength));
}

function flush(): void {
  if (Atomics.load(flowSignal, 0) === 0 || !transitions.length) return;
  parentPort!.postMessage({ type: "batch", actorId: `actor-${actorIndex}`, transitions: transitions.splice(0, 128) });
}

function waitForFlowSync(): void {
  while (running && Atomics.load(flowSignal, 0) === 0) Atomics.wait(flowSignal, 0, 0, 1_000);
}

class RemoteAgent implements CurriculumAgent {
  brain: Brain;
  constructor(readonly id: string, private policy: DensePolicy, private readonly collect: boolean) {
    this.brain = createBrain(id);
  }
  setPolicy(policy: DensePolicy): void {
    this.policy = policy;
    this.brain = {
      ...this.brain,
      epsilon: policy.epsilon,
      environmentSteps: policy.wire.environmentSteps,
      scenarioSteps: { ...this.brain.scenarioSteps, ...policy.wire.scenarioSteps },
    };
  }
  chooseAction(observation: ArrayLike<number>, explore: boolean, safeActions?: RelativeAction[]): RelativeAction {
    const contracted = observation.length === observationSize ? observation : Array.from(observation).slice(0, observationSize);
    return this.policy.action(contracted, safeActions, explore, random);
  }
  remember(experience: Experience, scenario: TrainingScenario, metrics: { opportunity?: boolean; claimed?: boolean; approachMiss?: boolean; powerUpKind?: PowerUpKind } = {}): void {
    const environmentSteps = this.brain.environmentSteps + 1;
    this.brain = {
      ...this.brain,
      environmentSteps,
      epsilon: explorationRate(environmentSteps, DEFAULT_HYPERPARAMETERS),
      scenarioSteps: { ...this.brain.scenarioSteps, [scenario]: this.brain.scenarioSteps[scenario] + 1 },
    };
    if (!this.collect) return;
    if (transitions.length >= 128) {
      waitForFlowSync();
      flush();
    }
    transitions.push({
      snakeId: this.id,
      scenario,
      state: bytes(experience.state),
      action: experience.action,
      reward: experience.reward,
      nextState: bytes(experience.nextState),
      terminal: experience.terminal,
      opportunity: Boolean(metrics.opportunity),
      claimed: Boolean(metrics.claimed),
      approachMiss: Boolean(metrics.approachMiss),
      episodeId: `${experimentId}:${actorIndex}:${episodeCounter}`,
      powerUpKind: metrics.powerUpKind ?? "",
      rewardComponents: experience.rewardBreakdown,
      rareFoodClaimed: Boolean(experience.rewardBreakdown?.rareFoodClaim),
      objectiveCaptured: Boolean(experience.rewardBreakdown?.objectiveCapture),
      bountyKill: Boolean(experience.rewardBreakdown?.bountyKill),
      deathCause: experience.deathCause ?? "",
    });
    if (transitions.length >= 128) flush();
  }
  async train(): Promise<null> { return null; }
  finishEpisode(score: number, food: number, won = false): void {
    this.brain = completeEpisode(this.brain, score, food, won, DEFAULT_HYPERPARAMETERS);
    flush();
  }
  async save(): Promise<boolean> { return false; }
}

const agents = new Map<string, RemoteAgent>();

function installPolicy(wire: WirePolicy): void {
  const previous = currentPolicies.get(wire.snakeId);
  if (previous) {
    const history = histories.get(wire.snakeId) ?? [];
    history.push(previous);
    while (history.length > DEFAULT_CURRICULUM.checkpointRetention) history.shift();
    histories.set(wire.snakeId, history);
  }
  const policy = new DensePolicy(wire);
  currentPolicies.set(wire.snakeId, policy);
  agents.get(wire.snakeId)?.setPolicy(policy);
}

async function waitForFlow(): Promise<void> {
  while (running && Atomics.load(flowSignal, 0) === 0) await new Promise((resolve) => setTimeout(resolve, 10));
}

function selectLegacyScenario(agent: RemoteAgent): TrainingScenario {
  if (agent.brain.environmentSteps < 50_000) return "survival";
  if (agent.brain.environmentSteps < 150_000) return "powerup";
  const roll = random();
  if (roll < .15) return "survival";
  if (roll < .40) return "powerup";
  return "battle";
}

async function loop(): Promise<void> {
  const definitions = new Map(activeRoster.map((entry) => [entry.id, entry]));
  while (running) {
    await waitForFlow();
    if (!running) break;
    const list = [...agents.values()];
    const primary = list[episodeCounter % list.length];
    const scenario: TrainingScenario = trainingSpecVersion === 2
      ? selectLegacyScenario(primary)
      : selectTrainingScenario(primary.brain, DEFAULT_CURRICULUM, random);
    const seed = (randomState + episodeCounter * 7_919) >>> 0;
    episodeCounter += 1;
    if (["survival", "powerup", "safezone", "hazard", "objective"].includes(scenario)) {
      const labs = { survival: runSurvivalEpisode, powerup: runPowerUpEpisode, safezone: runSafeZoneEpisode, hazard: runHazardEpisode, objective: runObjectiveEpisode } as const;
      await labs[scenario as keyof typeof labs](primary, definitions.get(primary.id)!, seed);
    } else {
      const participants: CurriculumAgent[] = [primary];
      const participantDefinitions = [definitions.get(primary.id)!];
      const learningIds = new Set([primary.id]);
      const roster = selectParticipantRoster(list, primary, battleSize);
      for (const current of roster.slice(1)) {
        const roll = random();
        if (roll < DEFAULT_CURRICULUM.currentOpponentRatio) {
          participants.push(current); learningIds.add(current.id);
        } else if (roll < DEFAULT_CURRICULUM.currentOpponentRatio + DEFAULT_CURRICULUM.historicalOpponentRatio) {
          const history = histories.get(current.id) ?? [];
          const policy = history[Math.floor(random() * history.length)];
          if (policy) participants.push(new RemoteAgent(current.id, policy, false));
          else { participants.push(current); learningIds.add(current.id); }
        } else {
          const styles = trainingSpecVersion === 2
            ? (["defensive", "food", "powerup"] as const)
            : (["aggressive", "defensive", "food", "powerup", "trap", "leaderHunter"] as const);
          participants.push(new ScriptedOpponent(current.id, styles[episodeCounter % styles.length]));
        }
        participantDefinitions.push(definitions.get(current.id)!);
      }
      if (scenario === "series") await runSeriesEpisode(participants, participantDefinitions, seed, learningIds);
      else await runBattleEpisode(participants, participantDefinitions, seed, 2_000, learningIds, { adaptiveArena: trainingSpecVersion === 3 });
    }
    flush();
    await new Promise((resolve) => setImmediate(resolve));
  }
}

parentPort.on("message", (message: Incoming) => {
  if (message.type === "flow") return;
  if (message.type === "stop") { running = false; flush(); return; }
  if (message.type === "policy") { installPolicy(message.policy); return; }
  experimentId = message.experimentId;
  trainingSpecVersion = message.trainingSpecVersion === 2 ? 2 : 3;
  observationSize = trainingSpecVersion === 2 ? 159 : 228;
  if (message.observationSize && message.observationSize !== observationSize) throw new Error("Learner observation contract does not match the requested training specification.");
  randomState = (message.seed + actorIndex * 104_729) >>> 0 || 1;
  const candidateRoster = (message.roster ?? []).filter((entry) => entry?.id);
  activeRoster = candidateRoster.length >= 2 ? candidateRoster : SNAKES;
  if (new Set(activeRoster.map((entry) => entry.id)).size !== activeRoster.length || activeRoster.length > 8) {
    throw new Error("Actor roster must contain 2-8 unique snake IDs.");
  }
  battleSize = Math.max(2, Math.min(activeRoster.length, Number(message.battleSize ?? DEFAULT_CURRICULUM.battleSize)));
  currentPolicies.clear();
  histories.clear();
  agents.clear();
  message.policies.forEach(installPolicy);
  for (const definition of activeRoster) {
    const policy = currentPolicies.get(definition.id);
    if (!policy) throw new Error(`Initial policy for ${definition.id} is missing.`);
    const agent = new RemoteAgent(definition.id, policy, true);
    agent.setPolicy(policy);
    agents.set(definition.id, agent);
  }
  running = true;
  void loop().catch((error) => parentPort!.postMessage({ type: "error", message: error instanceof Error ? error.message : String(error) }));
});
