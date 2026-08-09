import { createGame, isDangerAt, pointKey, powerUpDistanceField, relativeDirection, stepGame } from "./engine";
import { encodeObservation } from "./observation";
import type {
  Brain, CurriculumConfig, Experience, GameMode, GameState, OpponentProfile, PowerUpKind,
  RelativeAction, SnakeDefinition, TrainingDeathCause, TrainingScenario,
} from "./types";
import { defaultOpponentProfile, mapForSeed } from "./adaptive-arena";

// Agent adapter: DqnAgent satisfies it structurally, and tests can inject a fake.
export type CurriculumAgent = {
  brain: Brain;
  chooseAction(observation: ArrayLike<number>, explore: boolean, safeActions?: RelativeAction[]): RelativeAction;
  remember(
    experience: Experience,
    scenario?: TrainingScenario,
    metrics?: { opportunity?: boolean; claimed?: boolean; approachMiss?: boolean; powerUpKind?: PowerUpKind },
  ): void;
  train(force?: boolean): Promise<number | null>;
  finishEpisode(score: number, food: number, won?: boolean): void;
};

export type CurriculumEpisodeResult = {
  ticks: number;
  winnerId: string | null;
  outcomes: Record<string, { score: number; food: number; won: boolean }>;
  rounds?: Array<{ round: 1 | 2 | 3; winnerId: string | null }>;
  profiles?: Record<string, OpponentProfile>;
};

export type BattleEpisodeOptions = { heldOut?: boolean; adaptiveArena?: boolean };

const RELATIVE_ACTIONS = [0, 1, 2] as readonly RelativeAction[];

function trainingDeathCause(reason?: string): TrainingDeathCause | undefined {
  if (!reason) return undefined;
  if (reason.includes("zone")) return "zone";
  if (reason.includes("hazard")) return "hazard";
  if (reason.includes("wall")) return "wall";
  if (reason.includes("obstacle")) return "obstacle";
  if (reason.includes("head-on")) return "headOn";
  if (reason.includes("snake")) return "snakeBody";
  return "other";
}

function reachablePowerUp(state: GameState, snakeId: string): boolean {
  const snake = state.snakes.find((entry) => entry.id === snakeId && entry.alive);
  if (!snake || !state.powerUp) return false;
  const distance = powerUpDistanceField(state)?.get(pointKey(snake.segments[0]));
  return distance !== undefined && distance <= state.powerUp.expiresAt - state.tick;
}

export function safeActionsFor(state: GameState, snake: GameState["snakes"][number]): RelativeAction[] {
  const lookahead = snake.buffs.hasteUntil > state.tick || snake.buffs.visionUntil > state.tick ? 2 : 1;
  return RELATIVE_ACTIONS.filter((action) => !isDangerAt(state, snake, relativeDirection(snake.direction, action), lookahead));
}

export function selectTrainingScenario(
  brain: Brain,
  config: CurriculumConfig,
  random = Math.random,
): TrainingScenario {
  if (brain.scenarioSteps.survival < config.navigationWarmupSteps) return "survival";
  if (brain.scenarioSteps.powerup < config.powerUpWarmupSteps) return "powerup";
  if (brain.scenarioSteps.safezone < config.safeZoneWarmupSteps) return "safezone";
  if (brain.scenarioSteps.hazard < config.hazardWarmupSteps) return "hazard";
  if (brain.scenarioSteps.objective < config.objectiveWarmupSteps) return "objective";
  const roll = random();
  const weighted: Array<[TrainingScenario, number]> = [
    ["survival", config.survivalRatio], ["powerup", config.powerUpRatio], ["safezone", config.safeZoneRatio],
    ["hazard", config.hazardRatio], ["objective", config.objectiveRatio], ["battle", config.battleRatio], ["series", config.seriesRatio],
  ];
  let cumulative = 0;
  for (const [scenario, weight] of weighted) { cumulative += weight; if (roll < cumulative) return scenario; }
  return "series";
}

type StepOutcome = { state: GameState; experiences: Record<string, Experience[]>; actions: Record<string, RelativeAction> };

function stepAgents(
  state: GameState,
  choose: (snakeId: string, observation: number[], safeActions: RelativeAction[]) => RelativeAction,
): StepOutcome {
  const observations = new Map<string, { data: number[]; safe: RelativeAction[] }>();
  const aliveBefore = state.snakes.filter((snake) => snake.alive);
  for (const snake of aliveBefore) {
    observations.set(snake.id, {
      data: Array.from(encodeObservation(state, snake)),
      safe: safeActionsFor(state, snake),
    });
  }
  const actions: Record<string, RelativeAction> = {};
  for (const [id, payload] of observations) actions[id] = choose(id, payload.data, payload.safe);
  const result = stepGame(state, actions);
  const experiences: Record<string, Experience[]> = {};
  for (const snake of aliveBefore) {
    const next = result.state.snakes.find((entry) => entry.id === snake.id);
    if (!next) continue;
    (experiences[snake.id] ??= []).push({
      state: observations.get(snake.id)!.data,
      action: actions[snake.id] ?? 0,
      reward: result.rewards[snake.id] ?? 0,
      nextState: Array.from(encodeObservation(result.state, next)),
      terminal: !next.alive || result.state.status === "finished",
      rewardBreakdown: result.rewardBreakdowns[snake.id],
      deathCause: !next.alive ? trainingDeathCause(next.deathReason) : undefined,
    });
  }
  return { state: result.state, experiences, actions };
}

/**
 * Runs a single-snake survival/food episode in "training" mode. This is cheap,
 * high-throughput phase-A training that teaches movement, food-seeking, and
 * wall-avoidance before the agent enters contest with rivals.
 */
export async function runSurvivalEpisode(
  agent: CurriculumAgent,
  definition: SnakeDefinition,
  seed: number,
): Promise<CurriculumEpisodeResult> {
  let state = createGame({ mode: "training", snakes: [definition], seed, mapArchetype: mapForSeed(seed, false) });
  while (state.status === "running") {
    const snakeId = state.snakes[0].id;
    const outcome = stepAgents(
      state,
      (_id, observation, safeActions) => agent.chooseAction(observation, true, safeActions),
    );
    state = outcome.state;
    for (const experience of outcome.experiences[snakeId] ?? []) { agent.remember(experience, "survival"); await agent.train(); }
  }
  const snake = state.snakes[0];
  agent.finishEpisode(snake.score, snake.foodEaten);
  return {
    ticks: state.tick,
    winnerId: state.winnerId,
    outcomes: { [snake.id]: { score: snake.score, food: snake.foodEaten, won: false } },
  };
}

/** Single-agent lab with early, repeated power-up spawns for dense exposure. */
export async function runPowerUpEpisode(
  agent: CurriculumAgent,
  definition: SnakeDefinition,
  seed: number,
): Promise<CurriculumEpisodeResult> {
  let state = createGame({ mode: "powerup", snakes: [definition], seed, mapArchetype: mapForSeed(seed, false) });
  while (state.status === "running") {
    const before = state;
    const snakeId = state.snakes[0].id;
    const outcome = stepAgents(
      state,
      (_id, observation, safeActions) => agent.chooseAction(observation, true, safeActions),
    );
    state = outcome.state;
    const beforeSnake = before.snakes[0];
    const afterSnake = state.snakes[0];
    const opportunity = !before.powerUp && Boolean(state.powerUp) && reachablePowerUp(state, snakeId);
    const claimed = afterSnake.powerUps > beforeSnake.powerUps;
    const approachMiss = Boolean(before.powerUp) && !state.powerUp && !claimed;
    for (const experience of outcome.experiences[snakeId] ?? []) {
      agent.remember(experience, "powerup", { opportunity, claimed, approachMiss: approachMiss && (experience.rewardBreakdown?.powerUpApproach ?? 0) > 0, powerUpKind: (state.powerUp ?? before.powerUp)?.kind });
      await agent.train();
    }
  }
  const snake = state.snakes[0];
  agent.finishEpisode(snake.score, snake.foodEaten);
  return {
    ticks: state.tick,
    winnerId: null,
    outcomes: { [snake.id]: { score: snake.score, food: snake.foodEaten, won: false } },
  };
}

async function runAdaptiveLabEpisode(
  agent: CurriculumAgent,
  definition: SnakeDefinition,
  seed: number,
  mode: Extract<GameMode, "safezone" | "hazard" | "objective">,
): Promise<CurriculumEpisodeResult> {
  let state = createGame({ mode, snakes: [definition], seed, mapArchetype: mapForSeed(seed, false) });
  while (state.status === "running") {
    const snakeId = state.snakes[0].id;
    const outcome = stepAgents(state, (_id, observation, safe) => agent.chooseAction(observation, true, safe));
    state = outcome.state;
    for (const experience of outcome.experiences[snakeId] ?? []) { agent.remember(experience, mode); await agent.train(); }
  }
  const snake = state.snakes[0];
  agent.finishEpisode(snake.score, snake.foodEaten);
  return { ticks: state.tick, winnerId: null, outcomes: { [snake.id]: { score: snake.score, food: snake.foodEaten, won: false } } };
}

export const runSafeZoneEpisode = (agent: CurriculumAgent, definition: SnakeDefinition, seed: number) =>
  runAdaptiveLabEpisode(agent, definition, seed, "safezone");
export const runHazardEpisode = (agent: CurriculumAgent, definition: SnakeDefinition, seed: number) =>
  runAdaptiveLabEpisode(agent, definition, seed, "hazard");
export const runObjectiveEpisode = (agent: CurriculumAgent, definition: SnakeDefinition, seed: number) =>
  runAdaptiveLabEpisode(agent, definition, seed, "objective");

/**
 * Runs a multi-snake battle self-play episode in "battle" mode. All active
 * agents share one board, experiencing contention, head-on clashes, power-ups,
 * kills, and survival pressure. Each agent trains on its own collected
 * experiences as the match unfolds, then finishes its episode with the outcome.
 */
export async function runBattleEpisode(
  agents: CurriculumAgent[],
  definitions: SnakeDefinition[],
  seed: number,
  maxTicks = 2000,
  learningIds = new Set(agents.map((agent) => agent.brain.snakeId)),
  options: BattleEpisodeOptions = {},
): Promise<CurriculumEpisodeResult> {
  const byId = new Map(agents.map((agent) => [agent.brain.snakeId, agent]));
  let state = createGame({
    mode: "battle", snakes: definitions, seed,
    mapArchetype: mapForSeed(seed, options.heldOut ?? false),
    adaptiveArena: options.adaptiveArena ?? true,
  });
  while (state.status === "running") {
    const before = state;
    const outcome = stepAgents(
      state,
      (snakeId, observation, safeActions) => byId.get(snakeId)?.chooseAction(observation, true, safeActions) ?? 0,
    );
    state = outcome.state;
    for (const agent of agents) {
      if (!learningIds.has(agent.brain.snakeId)) continue;
      const mine = outcome.experiences[agent.brain.snakeId] ?? [];
      const beforeSnake = before.snakes.find((entry) => entry.id === agent.brain.snakeId);
      const afterSnake = state.snakes.find((entry) => entry.id === agent.brain.snakeId);
      const opportunity = !before.powerUp && Boolean(state.powerUp) && reachablePowerUp(state, agent.brain.snakeId);
      const claimed = Boolean(beforeSnake && afterSnake && afterSnake.powerUps > beforeSnake.powerUps);
      const approachMiss = Boolean(before.powerUp) && !state.powerUp && !claimed;
      for (const experience of mine) {
        agent.remember(experience, "battle", { opportunity, claimed, approachMiss: approachMiss && (experience.rewardBreakdown?.powerUpApproach ?? 0) > 0, powerUpKind: (state.powerUp ?? before.powerUp)?.kind });
        await agent.train();
      }
    }
    if (state.tick >= maxTicks && state.status === "running") {
      // Safety cap: resolve a stalled match without declaring a winner.
      state = { ...state, status: "finished", winnerId: null };
      break;
    }
  }
  const outcomes: CurriculumEpisodeResult["outcomes"] = {};
  for (const agent of agents) {
    const snake = state.snakes.find((entry) => entry.id === agent.brain.snakeId);
    const score = snake?.score ?? 0;
    const food = snake?.foodEaten ?? 0;
    const won = state.winnerId === agent.brain.snakeId;
    if (learningIds.has(agent.brain.snakeId)) agent.finishEpisode(score, food, won);
    outcomes[agent.brain.snakeId] = { score, food, won };
  }
  return { ticks: state.tick, winnerId: state.winnerId, outcomes };
}

function seriesProfiles(
  prior: Record<string, OpponentProfile>,
  outcome: CurriculumEpisodeResult,
  state: GameState,
  turns: Record<string, { left: number; right: number; total: number }>,
): Record<string, OpponentProfile> {
  return Object.fromEntries(Object.entries(outcome.outcomes).map(([id, row]) => {
    const previous = prior[id] ?? defaultOpponentProfile();
    const snake = state.snakes.find((entry) => entry.id === id);
    const action = turns[id] ?? { left: 0, right: 0, total: 0 };
    const samples = previous.samples + 1;
    const commonTarget = snake?.objectiveCaptures ? "objective" : snake && snake.powerUps > snake.foodEaten ? "powerup" : row.won ? "leader" : row.food >= 4 ? "food" : "survival";
    const reason = snake?.deathReason ?? "";
    const typicalDeathCause = reason.includes("wall") ? "wall" : reason.includes("obstacle") ? "obstacle" : reason.includes("head-on") ? "headOn" : reason.includes("snake") ? "snakeBody" : "other";
    return [id, {
      turnBias: (previous.turnBias * previous.samples + (action.right - action.left) / Math.max(1, action.total)) / samples,
      aggressionRate: (previous.aggressionRate * previous.samples + Math.min(1, (snake?.kills ?? 0) + (snake?.bountyKills ?? 0))) / samples,
      commonTarget,
      powerUpRate: (previous.powerUpRate * previous.samples + Math.min(1, (snake?.powerUps ?? 0) / Math.max(1, state.tick / 100))) / samples,
      typicalDeathCause,
      samples,
    }];
  }));
}

/** A deterministic best-of-three rematch. Later rounds expose compact profiles from prior rounds. */
export async function runSeriesEpisode(
  agents: CurriculumAgent[],
  definitions: SnakeDefinition[],
  seed: number,
  learningIds = new Set(agents.map((agent) => agent.brain.snakeId)),
  options: BattleEpisodeOptions = {},
): Promise<CurriculumEpisodeResult> {
  const totals: CurriculumEpisodeResult = { ticks: 0, winnerId: null, outcomes: {}, rounds: [] };
  let profiles: Record<string, OpponentProfile> = {};
  const wins = new Map<string, number>();
  for (let round = 1 as 1 | 2 | 3; round <= 3; round = (round + 1) as 1 | 2 | 3) {
    const byId = new Map(agents.map((agent) => [agent.brain.snakeId, agent]));
    const turns: Record<string, { left: number; right: number; total: number }> = {};
    const roundSeed = seed + round * 7_919;
    let state = createGame({
      mode: "battle", snakes: definitions, seed: roundSeed,
      mapArchetype: mapForSeed(roundSeed, options.heldOut ?? false),
      adaptiveArena: options.adaptiveArena ?? true,
      seriesRound: round,
      opponentProfiles: profiles,
    });
    while (state.status === "running" && state.tick < 2_000) {
      const before = state;
      const outcome = stepAgents(state, (id, observation, safe) => byId.get(id)?.chooseAction(observation, true, safe) ?? 0);
      state = outcome.state;
      for (const [id, action] of Object.entries(outcome.actions)) {
        const row = turns[id] ??= { left: 0, right: 0, total: 0 };
        row.total += 1; row.left += Number(action === 1); row.right += Number(action === 2);
      }
      for (const agent of agents) if (learningIds.has(agent.brain.snakeId)) {
        const beforeSnake = before.snakes.find((entry) => entry.id === agent.brain.snakeId);
        const afterSnake = state.snakes.find((entry) => entry.id === agent.brain.snakeId);
        const opportunity = !before.powerUp && Boolean(state.powerUp) && reachablePowerUp(state, agent.brain.snakeId);
        const claimed = Boolean(beforeSnake && afterSnake && afterSnake.powerUps > beforeSnake.powerUps);
        const disappeared = Boolean(before.powerUp) && !state.powerUp && !claimed;
        for (const experience of outcome.experiences[agent.brain.snakeId] ?? []) {
          agent.remember(experience, "series", { opportunity, claimed, approachMiss: disappeared && (experience.rewardBreakdown?.powerUpApproach ?? 0) > 0, powerUpKind: (state.powerUp ?? before.powerUp)?.kind });
          await agent.train();
        }
      }
    }
    const roundResult: CurriculumEpisodeResult = { ticks: state.tick, winnerId: state.winnerId, outcomes: {} };
    for (const agent of agents) {
      const snake = state.snakes.find((entry) => entry.id === agent.brain.snakeId);
      const won = state.winnerId === agent.brain.snakeId;
      roundResult.outcomes[agent.brain.snakeId] = { score: snake?.score ?? 0, food: snake?.foodEaten ?? 0, won };
      const total = totals.outcomes[agent.brain.snakeId] ?? { score: 0, food: 0, won: false };
      totals.outcomes[agent.brain.snakeId] = { score: total.score + (snake?.score ?? 0), food: total.food + (snake?.foodEaten ?? 0), won: total.won || won };
      if (won) wins.set(agent.brain.snakeId, (wins.get(agent.brain.snakeId) ?? 0) + 1);
    }
    totals.ticks += state.tick;
    totals.rounds!.push({ round, winnerId: state.winnerId });
    profiles = { ...profiles, ...seriesProfiles(profiles, roundResult, state, turns) };
  }
  totals.winnerId = [...wins.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? null;
  for (const agent of agents) if (learningIds.has(agent.brain.snakeId)) {
    const row = totals.outcomes[agent.brain.snakeId];
    agent.finishEpisode(row.score, row.food, totals.winnerId === agent.brain.snakeId);
  }
  totals.profiles = profiles;
  return totals;
}
