import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import * as tf from "@tensorflow/tfjs";
import { loadModelFromBundle } from "../game/model-bundle";
import { runEvaluation, type EvalAgent } from "../game/evaluation";
import { createBrain } from "../game/ai";
import { runSeriesEpisode } from "../game/curriculum";
import { ScriptedOpponent } from "../game/opponents";
import { MAP_ARCHETYPES, POWER_UP_KINDS, SNAKES } from "../game/constants";
import type { Brain, EvalSnakeStats, Experience, RelativeAction, SnakeDefinition, TrainingScenario } from "../game/types";

class ModelAgent implements EvalAgent {
  constructor(public brain: Brain, private model: tf.LayersModel, readonly observationSize = 228, readonly trainedEnvironmentSteps = brain.environmentSteps) {}
  chooseAction(observation: ArrayLike<number>, _explore: boolean, safeActions?: RelativeAction[]): RelativeAction {
    return tf.tidy(() => {
      const output = this.model.predict(tf.tensor2d([Array.from(observation).slice(0, this.observationSize)])) as tf.Tensor;
      const values = Array.from(output.dataSync());
      if (safeActions?.length) for (let index = 0; index < 3; index += 1) if (!safeActions.includes(index as RelativeAction)) values[index] = -Infinity;
      return values.indexOf(Math.max(...values)) as RelativeAction;
    });
  }
  remember(_experience: Experience, _scenario?: TrainingScenario): void {}
  async train(): Promise<null> { return null; }
  finishEpisode(_score: number, _food: number, _won?: boolean): void {}
  dispose(): void { this.model.dispose(); }
}

async function loadAgent(directory: string, snakeId: string, id: string): Promise<ModelAgent> {
  let parsed: any = null;
  for (const version of ["v3", "v2"] as const) {
    try { parsed = JSON.parse(await readFile(resolve(directory, `${snakeId}-${version}.nsbrain.json`), "utf8")); break; }
    catch { /* Try the other supported evaluation format. */ }
  }
  if (!parsed) throw new Error(`No compatible bundle exists for ${snakeId}.`);
  return loadEvaluationBundle(parsed, id, snakeId);
}

export async function loadEvaluationBundle(parsed: any, id: string, snakeId = id): Promise<ModelAgent> {
  if (parsed.modelVersion === 3) {
    const loaded = await loadModelFromBundle(parsed);
    return new ModelAgent({ ...loaded.bundle.brain, snakeId: id }, loaded.model, 228);
  }
  if (parsed.format !== "neon-serpents-brain" || parsed.formatVersion !== 1
    || parsed.modelVersion !== 2 || parsed.trainingSpecVersion !== 2
    || parsed.observationSize !== 159 || parsed.observationSpecHash !== "neon-serpents:v2:observation-159"
    || parsed.snakeId !== snakeId || !parsed.model?.topology || !Array.isArray(parsed.model?.weightSpecs)) {
    throw new Error(`Legacy baseline bundle for ${snakeId} has an incompatible contract.`);
  }
  const bytes = Buffer.from(parsed.model?.weightDataBase64 ?? "", "base64");
  const weightData = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const expectedBytes = parsed.model.weightSpecs.reduce((sum: number, spec: { shape: number[]; dtype: string }) => {
    const count = spec.shape.reduce((product, size) => product * size, 1);
    return sum + count * (spec.dtype === "float32" || spec.dtype === "int32" ? 4 : spec.dtype === "bool" ? 1 : 0);
  }, 0);
  if (!expectedBytes || bytes.byteLength !== expectedBytes || !new Float32Array(weightData).every(Number.isFinite)) {
    throw new Error(`Legacy baseline bundle for ${snakeId} has invalid weights.`);
  }
  const model = await tf.loadLayersModel(tf.io.fromMemory({
    modelTopology: parsed.model?.topology,
    weightSpecs: parsed.model?.weightSpecs,
    weightData,
    trainingConfig: parsed.model?.trainingConfig,
  }));
  if (model.inputs[0]?.shape.at(-1) !== 159 || model.outputs[0]?.shape.at(-1) !== 3) {
    model.dispose();
    throw new Error(`Legacy baseline bundle for ${snakeId} has invalid tensor shapes.`);
  }
  return new ModelAgent(createBrain(id), model, 159, Number(parsed.brain?.environmentSteps ?? 0));
}

class ObservationBlindAgent implements EvalAgent {
  brain: Brain;
  constructor(private readonly source: ModelAgent) { this.brain = source.brain; }
  chooseAction(observation: ArrayLike<number>): RelativeAction {
    const masked = Array.from(observation);
    masked.fill(0, 159);
    return this.source.chooseAction(masked, false, undefined);
  }
}

const definition = (id: string, name: string, color: string): SnakeDefinition => ({ id, name, color, accent: "#ffffff" });

export async function loadRoster(directory: string): Promise<SnakeDefinition[]> {
  let raw: string;
  try { raw = await readFile(resolve(directory, "roster.json"), "utf8"); }
  catch { return SNAKES; }
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed) || parsed.length < 2 || parsed.length > 8) throw new Error("invalid roster length");
    const roster = parsed.map((entry: any) => ({ id: String(entry.id), name: String(entry.name), color: String(entry.color), accent: String(entry.accent) }));
    if (roster.some((entry) => !entry.id || !entry.name || !/^#[0-9a-f]{6}$/i.test(entry.color) || !/^#[0-9a-f]{6}$/i.test(entry.accent)) || new Set(roster.map((entry) => entry.id)).size !== roster.length) throw new Error("invalid roster entries");
    return roster;
  } catch (error) { throw new Error(`Invalid roster metadata in ${directory}: ${error instanceof Error ? error.message : String(error)}`); }
}

async function headToHead(candidate: ModelAgent, baseline: ModelAgent, matches: number, seedBase: number, heldOut = true) {
  const outcomes: number[] = [];
  let candidateStats: EvalSnakeStats | null = null;
  let baselineStats: EvalSnakeStats | null = null;
  for (let index = 0; index < matches; index += 1) {
    const report = await runEvaluation(
      [candidate, baseline],
      [definition(candidate.brain.snakeId, "Candidate", "#68f7c1"), definition(baseline.brain.snakeId, "Baseline", "#ff6b7a")],
      1, seedBase + index * 7_919, 2_000, { heldOut },
    );
    const candidateRow = report.snakes[0];
    const baselineRow = report.snakes[1];
    outcomes.push(candidateRow.wins - baselineRow.wins);
    candidateStats = merge(candidateStats, candidateRow);
    baselineStats = merge(baselineStats, baselineRow);
  }
  return { outcomes, candidate: normalize(candidateStats!, matches), baseline: normalize(baselineStats!, matches) };
}

function merge(current: EvalSnakeStats | null, value: EvalSnakeStats): EvalSnakeStats {
  if (!current) return structuredClone(value);
  const result = { ...current };
  for (const key of ["wins", "powerUpsClaimed", "foodEaten", "powerUpOpportunities", "approachWithoutClaims", "deaths", "rareFoodClaims", "objectiveCaptures", "bountyKills", "hazardDeaths", "zoneDeaths"] as const) result[key] += value[key];
  result.avgScore += value.avgScore;
  result.avgSurvivalTicks += value.avgSurvivalTicks;
  for (const key of Object.keys(result.rewardBreakdown) as Array<keyof typeof result.rewardBreakdown>) result.rewardBreakdown[key] += value.rewardBreakdown[key];
  for (const key of Object.keys(result.deathCauses) as Array<keyof typeof result.deathCauses>) result.deathCauses[key] += value.deathCauses[key];
  for (const phase of ["opening", "midgame", "endgame"] as const) result.phaseSurvival[phase] += value.phaseSurvival[phase];
  for (const phase of ["opening", "midgame", "endgame"] as const) {
    result.phaseResults[phase].matches += value.phaseResults[phase].matches;
    result.phaseResults[phase].wins += value.phaseResults[phase].wins;
  }
  for (const map of Object.keys(result.mapResults) as Array<keyof typeof result.mapResults>) {
    result.mapResults[map].matches += value.mapResults[map].matches;
    result.mapResults[map].wins += value.mapResults[map].wins;
  }
  for (const kind of POWER_UP_KINDS) {
    const currentKind = result.powerUpBehavior[kind];
    const incoming = value.powerUpBehavior[kind];
    const seen = currentKind.seen + incoming.seen;
    const claimed = currentKind.claimed + incoming.claimed;
    const weighted = (left: number | null, leftCount: number, right: number | null, rightCount: number) => {
      const count = leftCount + rightCount;
      return count ? ((left ?? 0) * leftCount + (right ?? 0) * rightCount) / count : null;
    };
    result.powerUpBehavior[kind] = {
      seen,
      reachable: currentKind.reachable + incoming.reachable,
      pursued: currentKind.pursued + incoming.pursued,
      claimed,
      ignored: currentKind.ignored + incoming.ignored,
      pursuitWithoutClaim: currentKind.pursuitWithoutClaim + incoming.pursuitWithoutClaim,
      pursuitDeaths: currentKind.pursuitDeaths + incoming.pursuitDeaths,
      avgInitialDistance: weighted(currentKind.avgInitialDistance, currentKind.seen, incoming.avgInitialDistance, incoming.seen),
      avgClosestDistance: weighted(currentKind.avgClosestDistance, currentKind.seen, incoming.avgClosestDistance, incoming.seen),
      avgClaimTicks: weighted(currentKind.avgClaimTicks, currentKind.claimed, incoming.avgClaimTicks, incoming.claimed),
    };
    result.adaptive.powerUpCounters[kind].opportunities += value.adaptive.powerUpCounters[kind].opportunities;
    result.adaptive.powerUpCounters[kind].successes += value.adaptive.powerUpCounters[kind].successes;
  }
  for (const key of Object.keys(result.adaptive) as Array<keyof typeof result.adaptive>) {
    if (key !== "powerUpCounters") (result.adaptive[key] as number) += value.adaptive[key] as number;
  }
  return result;
}

function normalize(value: EvalSnakeStats, matches: number): EvalSnakeStats {
  const opportunities = value.powerUpOpportunities;
  return {
    ...value,
    avgScore: value.avgScore / matches,
    avgSurvivalTicks: value.avgSurvivalTicks / matches,
    powerUpClaimRate: opportunities ? value.powerUpsClaimed / opportunities : 0,
    approachWithoutClaimRate: opportunities ? value.approachWithoutClaims / opportunities : 0,
    deathsPerThousandTicks: value.avgSurvivalTicks ? value.deaths / value.avgSurvivalTicks * 1000 : 0,
  };
}

function bootstrapInterval(values: number[], seed = 42): [number, number] {
  let state = seed >>> 0 || 1;
  const random = () => { state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0; return state / 0x1_0000_0000; };
  const estimates = Array.from({ length: 10_000 }, () => {
    let sum = 0;
    for (let index = 0; index < values.length; index += 1) sum += values[Math.floor(random() * values.length)];
    return sum / values.length;
  }).sort((a, b) => a - b);
  return [estimates[Math.floor(estimates.length * .025)], estimates[Math.floor(estimates.length * .975)]];
}

const ratio = (numerator: number, denominator: number) => denominator ? numerator / denominator : 0;

async function soloLab(
  agent: EvalAgent,
  snake: SnakeDefinition,
  mode: "safezone" | "hazard" | "objective",
  runs: number,
  seed: number,
  safeActionMask = true,
): Promise<EvalSnakeStats> {
  const report = await runEvaluation([agent], [snake], runs, seed, 900, { mode, heldOut: true, safeActionMask });
  return report.snakes[0];
}

async function evaluateAdaptiveSeries(candidate: ModelAgent, snake: SnakeDefinition, matches: number, seedBase: number) {
  const roundWins = { 1: 0, 2: 0, 3: 0 };
  const roundMatches = { 1: 0, 2: 0, 3: 0 };
  const byStyle: Record<string, { wins: number; matches: number }> = {};
  const styles = ["aggressive", "defensive", "food", "powerup", "trap", "leaderHunter"] as const;
  const seriesPerStyle = Math.max(1, Math.floor(matches / (styles.length * 3)));
  for (const [styleIndex, style] of styles.entries()) {
    const row = byStyle[style] = { wins: 0, matches: 0 };
    for (let series = 0; series < seriesPerStyle; series += 1) {
      const scripted = new ScriptedOpponent(`series-${style}`, style);
      const result = await runSeriesEpisode(
        [candidate, scripted],
        [definition(candidate.brain.snakeId, "Candidate", snake.color), definition(scripted.brain.snakeId, `Adaptive ${style}`, "#ffffff")],
        seedBase + styleIndex * 100_000 + series * 23_357,
        new Set<string>(),
        { heldOut: true },
      );
      for (const round of result.rounds ?? []) {
        roundMatches[round.round] += 1;
        row.matches += 1;
        if (round.winnerId === candidate.brain.snakeId) {
          roundWins[round.round] += 1;
          row.wins += 1;
        }
      }
    }
  }
  return {
    roundWinRates: {
      1: ratio(roundWins[1], roundMatches[1]),
      2: ratio(roundWins[2], roundMatches[2]),
      3: ratio(roundWins[3], roundMatches[3]),
    },
    roundWins,
    roundMatches,
    byStyle,
  };
}

async function benchmarkSimulationThroughput(agent: EvalAgent, snake: SnakeDefinition, seed: number): Promise<number> {
  const started = performance.now();
  const report = await runEvaluation([agent], [snake], 20, seed, 900, {
    mode: "training", adaptiveArena: false, heldOut: false, safeActionMask: true,
  });
  const elapsedSeconds = Math.max(.001, (performance.now() - started) / 1_000);
  return report.snakes[0].avgSurvivalTicks * report.runs / elapsedSeconds;
}

async function main(): Promise<void> {
  const candidateDir = process.argv[2];
  const baselineDir = process.argv[3] === "-" ? null : process.argv[3];
  const output = process.argv[4];
  const matches = Math.max(1, Number(process.argv[5] ?? 200));
  if (!candidateDir || !output) throw new Error("Usage: evaluate-release <candidate-dir> <baseline-dir|-> <output.json> [matches]");
  try { await import("@tensorflow/tfjs-node"); } catch { await tf.setBackend("cpu"); }
  await tf.ready();
  const roster = await loadRoster(candidateDir);
  if (baselineDir) {
    const baselineRoster = await loadRoster(baselineDir);
    if (baselineRoster.map((entry) => entry.id).join(",") !== roster.map((entry) => entry.id).join(",")) {
      throw new Error("Candidate and baseline releases have incompatible snake rosters.");
    }
  }
  const perSnake: any[] = [];
  const paired: number[] = [];
  for (let snakeIndex = 0; snakeIndex < roster.length; snakeIndex += 1) {
    const snake = roster[snakeIndex];
    const candidate = await loadAgent(candidateDir, snake.id, `candidate-${snake.id}`);
    let baseline: ModelAgent | null = null;
    if (baselineDir) {
      baseline = await loadAgent(baselineDir, snake.id, `baseline-${snake.id}`);
      const validation = await headToHead(candidate, baseline, Math.min(40, matches), 1_000_000 + snakeIndex * 100_000);
      const test = await headToHead(candidate, baseline, matches, 2_000_000 + snakeIndex * 100_000);
      const trainingMaps = await headToHead(candidate, baseline, Math.max(12, Math.floor(matches / 4)), 2_500_000 + snakeIndex * 100_000, false);
      paired.push(...test.outcomes);
      perSnake.push({
        snakeId: snake.id,
        validationWinRate: validation.candidate.wins / Math.min(40, matches),
        candidate: test.candidate,
        baseline: test.baseline,
        candidateTrainingSteps: candidate.trainedEnvironmentSteps,
        baselineTrainingSteps: baseline.trainedEnvironmentSteps,
        trainingMapCandidate: trainingMaps.candidate,
        trainingMapBaseline: trainingMaps.baseline,
      });
    }
    let scriptedWins = 0;
    let scriptedMatches = 0;
    let scriptedCandidate: EvalSnakeStats | null = null;
    const scriptedStyles: Record<string, { wins: number; matches: number }> = {};
    const styles = ["aggressive", "defensive", "food", "powerup", "trap", "leaderHunter"] as const;
    for (const [styleIndex, style] of styles.entries()) {
        const scripted = new ScriptedOpponent(`scripted-${style}`, style);
        const styleMatches = Math.max(1, Math.floor(matches / styles.length));
        const report = await runEvaluation(
          [candidate, scripted],
          [definition(candidate.brain.snakeId, "Candidate", snake.color), definition(scripted.brain.snakeId, `Scripted ${style}`, "#ffffff")],
          styleMatches, 3_000_000 + snakeIndex * 100_000 + styleIndex * 10_000, 2_000,
        );
        scriptedWins += report.snakes[0].wins;
        scriptedMatches += styleMatches;
        scriptedCandidate = merge(scriptedCandidate, report.snakes[0]);
        scriptedStyles[style] = { wins: report.snakes[0].wins, matches: styleMatches };
    }
    const row = perSnake.find((entry) => entry.snakeId === snake.id);
    if (row) Object.assign(row, { scriptedWinRate: scriptedWins / scriptedMatches, scriptedStyles });
    else perSnake.push({ snakeId: snake.id, validationWinRate: scriptedWins / scriptedMatches, scriptedWinRate: scriptedWins / scriptedMatches, scriptedStyles, bootstrap: true, candidate: normalize(scriptedCandidate!, scriptedMatches), baseline: null });
    const selectedRow = perSnake.find((entry) => entry.snakeId === snake.id)!;
    const labRuns = Math.max(8, Math.min(40, Math.floor(matches / 5)));
    const labDefinition = definition(candidate.brain.snakeId, "Candidate", snake.color);
    const candidateZone = await soloLab(candidate, labDefinition, "safezone", labRuns, 4_000_000 + snakeIndex * 100_000);
    const candidateHazard = await soloLab(candidate, labDefinition, "hazard", labRuns, 4_100_000 + snakeIndex * 100_000);
    const candidateObjective = await soloLab(candidate, labDefinition, "objective", labRuns, 4_200_000 + snakeIndex * 100_000);
    const blindHazard = await soloLab(new ObservationBlindAgent(candidate), labDefinition, "hazard", labRuns, 4_100_000 + snakeIndex * 100_000, false);
    const adaptiveSeries = await evaluateAdaptiveSeries(candidate, snake, matches, 5_000_000 + snakeIndex * 1_000_000);
    const candidateThroughput = await benchmarkSimulationThroughput(candidate, labDefinition, 6_000_000 + snakeIndex * 100_000);
    Object.assign(selectedRow, { candidateLabs: { zone: candidateZone, hazard: candidateHazard, objective: candidateObjective }, blindHazard, adaptiveSeries, candidateThroughput });
    if (baseline) {
      const baselineDefinition = definition(baseline.brain.snakeId, "Baseline", "#ff6b7a");
      Object.assign(selectedRow, {
        baselineLabs: {
          zone: await soloLab(baseline, baselineDefinition, "safezone", labRuns, 4_000_000 + snakeIndex * 100_000),
          hazard: await soloLab(baseline, baselineDefinition, "hazard", labRuns, 4_100_000 + snakeIndex * 100_000),
          objective: await soloLab(baseline, baselineDefinition, "objective", labRuns, 4_200_000 + snakeIndex * 100_000),
        },
        baselineThroughput: await benchmarkSimulationThroughput(baseline, baselineDefinition, 6_000_000 + snakeIndex * 100_000),
      });
      baseline.dispose();
    }
    candidate.dispose();
  }
  const interval = paired.length ? bootstrapInterval(paired) : [0, 0];
  const powerCandidate = perSnake.reduce((sum, row) => sum + (row.candidate?.powerUpsClaimed ?? 0), 0);
  const powerBaseline = perSnake.reduce((sum, row) => sum + (row.baseline?.powerUpsClaimed ?? 0), 0);
  const opportunitiesCandidate = perSnake.reduce((sum, row) => sum + (row.candidate?.powerUpOpportunities ?? 0), 0);
  const opportunitiesBaseline = perSnake.reduce((sum, row) => sum + (row.baseline?.powerUpOpportunities ?? 0), 0);
  const powerClaimRateCandidate = opportunitiesCandidate ? powerCandidate / opportunitiesCandidate : 0;
  const powerClaimRateBaseline = opportunitiesBaseline ? powerBaseline / opportunitiesBaseline : 0;
  const survivalCandidate = perSnake.reduce((sum, row) => sum + (row.candidate?.avgSurvivalTicks ?? 0), 0) / perSnake.length;
  const survivalBaseline = perSnake.reduce((sum, row) => sum + (row.baseline?.avgSurvivalTicks ?? 0), 0) / perSnake.length;
  const deathsCandidate = perSnake.reduce((sum, row) => sum + (row.candidate?.deathsPerThousandTicks ?? 0), 0) / perSnake.length;
  const deathsBaseline = perSnake.reduce((sum, row) => sum + (row.baseline?.deathsPerThousandTicks ?? 0), 0) / perSnake.length;
  const totalMatches = matches * perSnake.length;
  const rareCandidate = perSnake.reduce((sum, row) => sum + (row.candidate?.rareFoodClaims ?? 0), 0) / Math.max(1, totalMatches);
  const rareBaseline = perSnake.reduce((sum, row) => sum + (row.baseline?.rareFoodClaims ?? 0), 0) / Math.max(1, totalMatches);
  const objectiveCandidate = perSnake.reduce((sum, row) => sum + (row.candidate?.objectiveCaptures ?? 0), 0) / Math.max(1, totalMatches);
  const objectiveBaseline = perSnake.reduce((sum, row) => sum + (row.baseline?.objectiveCaptures ?? 0), 0) / Math.max(1, totalMatches);
  const hazardDeathsCandidate = perSnake.reduce((sum, row) => sum + (row.candidate?.hazardDeaths ?? 0), 0) / Math.max(1, totalMatches);
  const hazardDeathsBaseline = perSnake.reduce((sum, row) => sum + (row.baseline?.hazardDeaths ?? 0), 0) / Math.max(1, totalMatches);
  const zoneDeathsCandidate = perSnake.reduce((sum, row) => sum + (row.candidate?.zoneDeaths ?? 0), 0) / Math.max(1, totalMatches);
  const zoneDeathsBaseline = perSnake.reduce((sum, row) => sum + (row.baseline?.zoneDeaths ?? 0), 0) / Math.max(1, totalMatches);
  const adaptiveTotal = (side: "candidate" | "baseline", key: string, lab?: "zone" | "hazard" | "objective") => perSnake.reduce((sum, row) => {
    const source = lab ? row[`${side}Labs`]?.[lab]?.adaptive : row[side]?.adaptive;
    return sum + Number(source?.[key] ?? 0);
  }, 0);
  const zoneWarningsCandidate = adaptiveTotal("candidate", "zoneWarnings", "zone");
  const zoneWarningsBaseline = adaptiveTotal("baseline", "zoneWarnings", "zone");
  const zoneRepositionsCandidate = adaptiveTotal("candidate", "zoneRepositions", "zone");
  const zoneRepositionsBaseline = adaptiveTotal("baseline", "zoneRepositions", "zone");
  const zoneRepositionRateCandidate = ratio(zoneRepositionsCandidate, zoneWarningsCandidate);
  const zoneRepositionRateBaseline = ratio(zoneRepositionsBaseline, zoneWarningsBaseline);
  const hazardEncountersCandidate = adaptiveTotal("candidate", "hazardEncounters", "hazard");
  const hazardEvasionsCandidate = adaptiveTotal("candidate", "hazardEvasions", "hazard");
  const hazardEncountersBaseline = adaptiveTotal("baseline", "hazardEncounters", "hazard");
  const hazardEvasionsBaseline = adaptiveTotal("baseline", "hazardEvasions", "hazard");
  const blindHazardEncounters = perSnake.reduce((sum, row) => sum + Number(row.blindHazard?.adaptive?.hazardEncounters ?? 0), 0);
  const blindHazardEvasions = perSnake.reduce((sum, row) => sum + Number(row.blindHazard?.adaptive?.hazardEvasions ?? 0), 0);
  const hazardAvoidanceRateCandidate = ratio(hazardEvasionsCandidate, hazardEncountersCandidate);
  const hazardAvoidanceRateBaseline = ratio(hazardEvasionsBaseline, hazardEncountersBaseline);
  const hazardAvoidanceRateBlind = ratio(blindHazardEvasions, blindHazardEncounters);
  const objectiveOpportunitiesCandidate = adaptiveTotal("candidate", "objectiveOpportunities", "objective");
  const objectiveOpportunitiesBaseline = adaptiveTotal("baseline", "objectiveOpportunities", "objective");
  const objectiveCapturesLabCandidate = adaptiveTotal("candidate", "objectiveCaptures", "objective");
  const objectiveCapturesLabBaseline = adaptiveTotal("baseline", "objectiveCaptures", "objective");
  const objectivePursuitDeathsCandidate = adaptiveTotal("candidate", "objectivePursuitDeaths", "objective");
  const objectivePursuitDeathsBaseline = adaptiveTotal("baseline", "objectivePursuitDeaths", "objective");
  const objectiveCaptureRateCandidate = ratio(objectiveCapturesLabCandidate, objectiveOpportunitiesCandidate);
  const objectiveCaptureRateBaseline = ratio(objectiveCapturesLabBaseline, objectiveOpportunitiesBaseline);
  const objectivePursuitDeathRateCandidate = ratio(objectivePursuitDeathsCandidate, objectiveOpportunitiesCandidate);
  const objectivePursuitDeathRateBaseline = ratio(objectivePursuitDeathsBaseline, objectiveOpportunitiesBaseline);
  const counterTotals = (side: "candidate" | "baseline") => perSnake.reduce((totals, row) => {
    for (const kind of POWER_UP_KINDS) {
      const value = row[side]?.adaptive?.powerUpCounters?.[kind];
      totals.opportunities += Number(value?.opportunities ?? 0);
      totals.successes += Number(value?.successes ?? 0);
    }
    return totals;
  }, { opportunities: 0, successes: 0 });
  const countersCandidate = counterTotals("candidate");
  const countersBaseline = counterTotals("baseline");
  const counterSuccessRateCandidate = ratio(countersCandidate.successes, countersCandidate.opportunities);
  const counterSuccessRateBaseline = ratio(countersBaseline.successes, countersBaseline.opportunities);
  const powerUpCounters = Object.fromEntries(POWER_UP_KINDS.map((kind) => {
    const candidate = perSnake.reduce((sum, row) => sum + Number(row.candidate?.adaptive?.powerUpCounters?.[kind]?.successes ?? 0), 0);
    const candidateOpportunities = perSnake.reduce((sum, row) => sum + Number(row.candidate?.adaptive?.powerUpCounters?.[kind]?.opportunities ?? 0), 0);
    const baseline = perSnake.reduce((sum, row) => sum + Number(row.baseline?.adaptive?.powerUpCounters?.[kind]?.successes ?? 0), 0);
    const baselineOpportunities = perSnake.reduce((sum, row) => sum + Number(row.baseline?.adaptive?.powerUpCounters?.[kind]?.opportunities ?? 0), 0);
    return [kind, { candidate, candidateOpportunities, candidateRate: ratio(candidate, candidateOpportunities), baseline, baselineOpportunities, baselineRate: ratio(baseline, baselineOpportunities) }];
  }));
  const roundTotals = (round: 1 | 2 | 3, field: "roundWins" | "roundMatches") => perSnake.reduce((sum, row) => sum + Number(row.adaptiveSeries?.[field]?.[round] ?? 0), 0);
  const roundOneWinRate = ratio(roundTotals(1, "roundWins"), roundTotals(1, "roundMatches"));
  const roundThreeWinRate = ratio(roundTotals(3, "roundWins"), roundTotals(3, "roundMatches"));
  const candidateThroughput = perSnake.reduce((sum, row) => sum + Number(row.candidateThroughput ?? 0), 0) / perSnake.length;
  const baselineThroughput = perSnake.reduce((sum, row) => sum + Number(row.baselineThroughput ?? 0), 0) / perSnake.length;
  const throughputRatio = baselineThroughput ? candidateThroughput / baselineThroughput : 1;
  const candidateTrainingSteps = perSnake.reduce((sum, row) => sum + Number(row.candidateTrainingSteps ?? 0), 0);
  const baselineTrainingSteps = perSnake.reduce((sum, row) => sum + Number(row.baselineTrainingSteps ?? 0), 0);
  const trainingBudgetRatio = candidateTrainingSteps ? baselineTrainingSteps / candidateTrainingSteps : 0;
  const improvement = (candidate: number, baseline: number, factor: number) => baseline === 0 ? candidate > 0 : candidate >= baseline * factor;
  const stableLower = (candidate: number, baseline: number) => baseline === 0 ? candidate === 0 : candidate <= baseline * 1.05;
  const phaseWinRates = Object.fromEntries((["opening", "midgame", "endgame"] as const).map((phase) => {
    const candidate = perSnake.reduce((sum, row) => sum + (row.candidate?.phaseResults?.[phase]?.wins ?? 0), 0);
    const candidateMatches = perSnake.reduce((sum, row) => sum + (row.candidate?.phaseResults?.[phase]?.matches ?? 0), 0);
    const baseline = perSnake.reduce((sum, row) => sum + (row.baseline?.phaseResults?.[phase]?.wins ?? 0), 0);
    const baselineMatches = perSnake.reduce((sum, row) => sum + (row.baseline?.phaseResults?.[phase]?.matches ?? 0), 0);
    return [phase, { candidate: candidate / Math.max(1, candidateMatches), baseline: baseline / Math.max(1, baselineMatches), candidateMatches, baselineMatches }];
  }));
  const mapWinRates = Object.fromEntries(MAP_ARCHETYPES.map((map) => {
    const candidate = perSnake.reduce((sum, row) => sum + (row.candidate?.mapResults?.[map]?.wins ?? 0), 0);
    const candidateMatches = perSnake.reduce((sum, row) => sum + (row.candidate?.mapResults?.[map]?.matches ?? 0), 0);
    const baseline = perSnake.reduce((sum, row) => sum + (row.baseline?.mapResults?.[map]?.wins ?? 0), 0);
    const baselineMatches = perSnake.reduce((sum, row) => sum + (row.baseline?.mapResults?.[map]?.matches ?? 0), 0);
    return [map, { candidate: candidate / Math.max(1, candidateMatches), baseline: baseline / Math.max(1, baselineMatches), candidateMatches, baselineMatches }];
  }));
  const trainingMapWinRates = Object.fromEntries(MAP_ARCHETYPES.map((map) => {
    const candidate = perSnake.reduce((sum, row) => sum + (row.trainingMapCandidate?.mapResults?.[map]?.wins ?? 0), 0);
    const candidateMatches = perSnake.reduce((sum, row) => sum + (row.trainingMapCandidate?.mapResults?.[map]?.matches ?? 0), 0);
    const baseline = perSnake.reduce((sum, row) => sum + (row.trainingMapBaseline?.mapResults?.[map]?.wins ?? 0), 0);
    const baselineMatches = perSnake.reduce((sum, row) => sum + (row.trainingMapBaseline?.mapResults?.[map]?.matches ?? 0), 0);
    return [map, { candidate: ratio(candidate, candidateMatches), baseline: ratio(baseline, baselineMatches), candidateMatches, baselineMatches }];
  }));
  const heldOutCandidateWins = Object.values(mapWinRates).reduce((sum, row) => sum + row.candidate * row.candidateMatches, 0);
  const heldOutCandidateMatches = Object.values(mapWinRates).reduce((sum, row) => sum + row.candidateMatches, 0);
  const heldOutBaselineWins = Object.values(mapWinRates).reduce((sum, row) => sum + row.baseline * row.baselineMatches, 0);
  const heldOutBaselineMatches = Object.values(mapWinRates).reduce((sum, row) => sum + row.baselineMatches, 0);
  const trainingCandidateWins = Object.values(trainingMapWinRates).reduce((sum, row) => sum + row.candidate * row.candidateMatches, 0);
  const trainingCandidateMatches = Object.values(trainingMapWinRates).reduce((sum, row) => sum + row.candidateMatches, 0);
  const trainingBaselineWins = Object.values(trainingMapWinRates).reduce((sum, row) => sum + row.baseline * row.baselineMatches, 0);
  const trainingBaselineMatches = Object.values(trainingMapWinRates).reduce((sum, row) => sum + row.baselineMatches, 0);
  const heldOutVsTraining = {
    candidate: { heldOut: ratio(heldOutCandidateWins, heldOutCandidateMatches), training: ratio(trainingCandidateWins, trainingCandidateMatches) },
    baseline: { heldOut: ratio(heldOutBaselineWins, heldOutBaselineMatches), training: ratio(trainingBaselineWins, trainingBaselineMatches) },
  };
  const scriptedWinRate = perSnake.reduce((sum, row) => sum + row.scriptedWinRate, 0) / perSnake.length;
  const stable = survivalCandidate >= survivalBaseline * 0.95 && (deathsBaseline === 0 ? deathsCandidate === 0 : deathsCandidate <= deathsBaseline * 1.05);
  const powerImproved = powerClaimRateBaseline === 0 ? powerClaimRateCandidate > 0 : powerClaimRateCandidate >= powerClaimRateBaseline * 1.25;
  const hazardAvoidanceImproved = hazardAvoidanceRateBlind === 0 ? hazardAvoidanceRateCandidate > 0 : hazardAvoidanceRateCandidate >= hazardAvoidanceRateBlind * 1.25;
  const objectiveImproved = improvement(objectiveCaptureRateCandidate, objectiveCaptureRateBaseline, 1.01)
    && (objectivePursuitDeathRateBaseline === 0 ? objectivePursuitDeathRateCandidate === 0 : objectivePursuitDeathRateCandidate <= objectivePursuitDeathRateBaseline);
  const counterImproved = counterSuccessRateBaseline === 0 ? counterSuccessRateCandidate > 0 : counterSuccessRateCandidate >= counterSuccessRateBaseline * 1.25;
  const heldOutImproved = heldOutVsTraining.candidate.heldOut > heldOutVsTraining.baseline.heldOut;
  const noMapRegression = Object.values(mapWinRates).every((row) => row.baselineMatches === 0 || row.candidate >= row.baseline - .05);
  const gates = {
    baselineCompatible: Boolean(baselineDir),
    baselineBudgetComparable: Boolean(baselineDir) && trainingBudgetRatio >= .99 && trainingBudgetRatio <= 1.01,
    positivePairedInterval: Boolean(baselineDir) && interval[0] > 0,
    powerUpImprovement: Boolean(baselineDir) && powerImproved,
    survivalStable: Boolean(baselineDir) && survivalCandidate >= survivalBaseline * 0.95,
    deathsStable: Boolean(baselineDir) && (deathsBaseline === 0 ? deathsCandidate === 0 : deathsCandidate <= deathsBaseline * 1.05),
    zoneRepositioning: zoneRepositionRateCandidate >= .85,
    hazardAvoidanceImprovement: hazardAvoidanceImproved,
    objectiveImprovement: Boolean(baselineDir) && objectiveImproved,
    powerUpCounterImprovement: Boolean(baselineDir) && counterImproved,
    roundThreeAdaptation: roundThreeWinRate >= roundOneWinRate + .05,
    heldOutMapImprovement: Boolean(baselineDir) && heldOutImproved,
    noMapRegression,
    throughputStable: throughputRatio >= .85,
    frozenOpponentWinRate: Boolean(baselineDir) && paired.reduce((sum, value) => sum + Number(value > 0), 0) / Math.max(1, paired.length) >= .5,
    scriptedWinRate: scriptedWinRate >= 0.5,
  };
  const eligible = Object.values(gates).every(Boolean);
  await writeFile(output, JSON.stringify({
    metricsSchemaVersion: 3,
    coverage: "full",
    matches,
    perSnake,
    pairedWinInterval: interval,
    pairedOutcomes: paired,
    powerCandidate,
    powerBaseline,
    powerClaimRateCandidate,
    powerClaimRateBaseline,
    survivalCandidate,
    survivalBaseline,
    deathsCandidate,
    deathsBaseline,
    rareCandidate,
    rareBaseline,
    objectiveCandidate,
    objectiveBaseline,
    hazardDeathsCandidate,
    hazardDeathsBaseline,
    zoneDeathsCandidate,
    zoneDeathsBaseline,
    phaseWinRates,
    mapWinRates,
    trainingMapWinRates,
    heldOutVsTraining,
    powerUpCounters,
    adaptive: {
      zoneWarningsCandidate, zoneWarningsBaseline, zoneRepositionsCandidate, zoneRepositionsBaseline,
      zoneRepositionRateCandidate, zoneRepositionRateBaseline,
      hazardEncountersCandidate, hazardEvasionsCandidate, hazardEncountersBaseline, hazardEvasionsBaseline,
      hazardAvoidanceRateCandidate, hazardAvoidanceRateBaseline, hazardAvoidanceRateBlind,
      objectiveOpportunitiesCandidate, objectiveOpportunitiesBaseline,
      objectiveCapturesCandidate: objectiveCapturesLabCandidate, objectiveCapturesBaseline: objectiveCapturesLabBaseline,
      objectiveCaptureRateCandidate, objectiveCaptureRateBaseline,
      objectivePursuitDeathsCandidate, objectivePursuitDeathsBaseline,
      objectivePursuitDeathRateCandidate, objectivePursuitDeathRateBaseline,
      counterSuccessRateCandidate, counterSuccessRateBaseline,
      roundOneWinRate, roundThreeWinRate,
      candidateThroughput, baselineThroughput, throughputRatio,
      candidateTrainingSteps, baselineTrainingSteps, trainingBudgetRatio,
    },
    scriptedWinRate,
    stable,
    powerImproved,
    gates,
    eligible,
  }, null, 2), "utf8");
}

if (require.main === module) main().catch((error) => { console.error(error); process.exitCode = 1; });
