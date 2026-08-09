import { createGame, isDangerAt, relativeDirection, stepGame } from "./engine";
import { encodeObservation } from "./observation";
import { mapForSeed } from "./adaptive-arena";
import {
  AdaptiveBehaviorTracker, emptyAdaptiveBehavior, emptyDeathCauses, emptyPowerUpBehavior,
  finalizePowerUpBehavior, normalizeDeathCause, PowerUpBehaviorTracker,
} from "./evaluation-intelligence";
import type {
  AdaptiveBehaviorStats, Brain, DeathCause, EvalReport, EvalSnakeStats, GameMode, GameState,
  MapArchetype, OpponentProfile, PowerUpKind, RelativeAction, RewardBreakdown, SnakeDefinition,
} from "./types";
import { MAP_ARCHETYPES } from "./constants";

export interface EvalAgent {
  brain: Brain;
  chooseAction(observation: ArrayLike<number>, explore: boolean, safeActions?: RelativeAction[]): RelativeAction;
}

export type RunEvaluationOptions = {
  heldOut?: boolean;
  adaptiveArena?: boolean;
  safeActionMask?: boolean;
  seriesRound?: 1 | 2 | 3;
  opponentProfiles?: Record<string, OpponentProfile>;
  mode?: GameMode;
};

const RELATIVE_ACTIONS = [0, 1, 2] as readonly RelativeAction[];

function safeActions(state: GameState, snake: GameState["snakes"][number]): RelativeAction[] {
  const lookahead = snake.buffs.hasteUntil > state.tick || snake.buffs.visionUntil > state.tick ? 2 : 1;
  return RELATIVE_ACTIONS.filter((action) => !isDangerAt(state, snake, relativeDirection(snake.direction, action), lookahead));
}

/**
 * Runs `runs` greedy head-to-head playoffs against the active roster and
 * aggregates per-snake statistics. Evaluation is done with `explore=false` so it
 * measures the trained policy rather than exploration noise.
 */
export async function runEvaluation(
  agents: EvalAgent[],
  definitions: SnakeDefinition[],
  runs: number,
  seedOffset = 0,
  maxTicks = 2000,
  options: RunEvaluationOptions = {},
): Promise<EvalReport> {
  const startedAt = Date.now();
  const map = new Map(agents.map((agent) => [agent.brain.snakeId, agent]));
  const blankReward = (): RewardBreakdown => ({
    step: 0, survival: 0, foodApproach: 0, foodClaim: 0, powerUpApproach: 0, powerUpClaim: 0,
    zonePositioning: 0, objectiveApproach: 0, objectiveCapture: 0, bountyKill: 0, rareFoodClaim: 0,
    kill: 0, death: 0, win: 0,
  });
  const emptyMaps = () => Object.fromEntries(MAP_ARCHETYPES.map((map) => [map, { matches: 0, wins: 0 }])) as Record<MapArchetype, { matches: number; wins: number }>;
  const aggregate: Record<string, { score: number; food: number; powerUps: number; opportunities: number; misses: number; deaths: number; survival: number; wins: number; rareFood: number; objectives: number; bounties: number; hazardDeaths: number; zoneDeaths: number; phaseSurvival: Record<"opening" | "midgame" | "endgame", number>; phaseResults: Record<"opening" | "midgame" | "endgame", { matches: number; wins: number }>; mapResults: Record<MapArchetype, { matches: number; wins: number }>; rewards: RewardBreakdown; deathCauses: Record<DeathCause, number>; powerUpsByKind: ReturnType<typeof emptyPowerUpBehavior>; adaptive: AdaptiveBehaviorStats }> = {};
  for (const def of definitions) aggregate[def.id] = { score: 0, food: 0, powerUps: 0, opportunities: 0, misses: 0, deaths: 0, survival: 0, wins: 0, rareFood: 0, objectives: 0, bounties: 0, hazardDeaths: 0, zoneDeaths: 0, phaseSurvival: { opening: 0, midgame: 0, endgame: 0 }, phaseResults: { opening: { matches: 0, wins: 0 }, midgame: { matches: 0, wins: 0 }, endgame: { matches: 0, wins: 0 } }, mapResults: emptyMaps(), rewards: blankReward(), deathCauses: emptyDeathCauses(), powerUpsByKind: emptyPowerUpBehavior(), adaptive: emptyAdaptiveBehavior() };

  for (let run = 0; run < runs; run += 1) {
    const evaluationSeed = seedOffset + run * 7919;
    let state = createGame({
      mode: options.mode ?? "battle", snakes: definitions, seed: evaluationSeed,
      mapArchetype: mapForSeed(evaluationSeed, options.heldOut ?? true),
      adaptiveArena: options.adaptiveArena ?? true,
      seriesRound: options.seriesRound,
      opponentProfiles: options.opponentProfiles,
    });
    const tracker = new PowerUpBehaviorTracker(Object.fromEntries(Object.entries(aggregate).map(([id, row]) => [id, row.powerUpsByKind])) as Record<string, Record<PowerUpKind, ReturnType<typeof emptyPowerUpBehavior>[PowerUpKind]>>);
    const adaptiveTracker = new AdaptiveBehaviorTracker(Object.fromEntries(Object.entries(aggregate).map(([id, row]) => [id, row.adaptive])));
    while (state.status === "running") {
      for (const snake of state.snakes.filter((entry) => entry.alive)) {
        aggregate[snake.id].survival += 1;
        aggregate[snake.id].phaseSurvival[state.arena.phase] += 1;
      }
      const observations: Record<string, { data: number[]; safe: RelativeAction[] }> = {};
      for (const snake of state.snakes.filter((entry) => entry.alive)) {
        observations[snake.id] = { data: Array.from(encodeObservation(state, snake)), safe: options.safeActionMask === false ? [] : safeActions(state, snake) };
      }
      const actions: Record<string, RelativeAction> = {};
      for (const [id, payload] of Object.entries(observations)) actions[id] = map.get(id)?.chooseAction(payload.data, false, payload.safe) ?? 0;
      const before = state;
      const result = stepGame(state, actions);
      state = result.state;
      tracker.observe(before, state);
      adaptiveTracker.observe(before, state);
      const spawned = !before.powerUp && Boolean(state.powerUp);
      const disappeared = Boolean(before.powerUp) && !state.powerUp;
      for (const snake of before.snakes.filter((entry) => entry.alive)) {
        const nextSnake = state.snakes.find((entry) => entry.id === snake.id);
        if (!nextSnake) continue;
        if (spawned) aggregate[snake.id].opportunities += 1;
        const claimed = nextSnake.powerUps > snake.powerUps;
        if (disappeared && !claimed) aggregate[snake.id].misses += 1;
        if (!nextSnake.alive) {
          aggregate[snake.id].deaths += 1;
          aggregate[snake.id].deathCauses[normalizeDeathCause(nextSnake.deathReason)] += 1;
          aggregate[snake.id].hazardDeaths += Number(nextSnake.deathReason?.includes("hazard"));
          aggregate[snake.id].zoneDeaths += Number(nextSnake.deathReason?.includes("zone"));
        }
        const breakdown = result.rewardBreakdowns[snake.id];
        if (breakdown) for (const key of Object.keys(breakdown) as Array<keyof RewardBreakdown>) aggregate[snake.id].rewards[key] += breakdown[key];
      }
      if (maxTicks > 0 && state.tick >= maxTicks && state.status === "running") state = { ...state, status: "finished", winnerId: null };
      if (state.tick % 64 === 0) await new Promise((resolve) => setTimeout(resolve, 0));
    }
    tracker.finish();
    adaptiveTracker.finish(state);
    for (const snake of state.snakes) {
      const row = aggregate[snake.id];
      if (!row) continue;
      row.score += snake.score;
      row.food += snake.foodEaten;
      row.powerUps += snake.powerUps;
      row.rareFood += snake.rareFoodEaten;
      row.objectives += snake.objectiveCaptures;
      row.bounties += snake.bountyKills;
      row.mapResults[state.arena.mapArchetype].matches += 1;
      row.mapResults[state.arena.mapArchetype].wins += Number(state.winnerId === snake.id);
      row.phaseResults[state.arena.phase].matches += 1;
      row.phaseResults[state.arena.phase].wins += Number(state.winnerId === snake.id);
      if (state.winnerId === snake.id) row.wins += 1;
    }
  }

  const snakes: EvalSnakeStats[] = definitions.map((def) => {
    const row = aggregate[def.id];
    const powerUpBehavior = finalizePowerUpBehavior(row.powerUpsByKind);
    const opportunities = Object.values(powerUpBehavior).reduce((sum, value) => sum + value.reachable, 0);
    const approachWithoutClaims = Object.values(powerUpBehavior).reduce((sum, value) => sum + value.pursuitWithoutClaim, 0);
    return {
      id: def.id,
      wins: row.wins,
      avgScore: row.score / runs,
      powerUpsClaimed: row.powerUps,
      foodEaten: row.food,
      avgSurvivalTicks: row.survival / runs,
      powerUpOpportunities: opportunities,
      powerUpClaimRate: opportunities ? row.powerUps / opportunities : 0,
      approachWithoutClaimRate: opportunities ? approachWithoutClaims / opportunities : 0,
      approachWithoutClaims,
      deaths: row.deaths,
      deathsPerThousandTicks: row.survival ? row.deaths / row.survival * 1000 : 0,
      rareFoodClaims: row.adaptive.rareFoodClaims,
      objectiveCaptures: row.adaptive.objectiveCaptures,
      bountyKills: row.adaptive.bountyKills,
      hazardDeaths: row.adaptive.hazardDeaths,
      zoneDeaths: row.adaptive.zoneDeaths,
      phaseSurvival: row.phaseSurvival,
      phaseResults: row.phaseResults,
      mapResults: row.mapResults,
      rewardBreakdown: row.rewards,
      deathCauses: row.deathCauses,
      powerUpBehavior,
      adaptive: row.adaptive,
    };
  });

  return { runs, snakes, startedAt, finishedAt: Date.now() };
}

export function summarize(report: EvalReport): string {
  return report.snakes
    .slice()
    .sort((a, b) => b.wins - a.wins || b.avgScore - a.avgScore)
    .map((snake) => `${snake.id}: ${snake.wins}W, avg ${snake.avgScore.toFixed(1)}pts, ${snake.powerUpsClaimed} power-ups, ${snake.foodEaten} food`)
    .join(" | ");
}
