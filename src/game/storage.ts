import { createStore, get, set } from "idb-keyval";
import { createBrain, DEFAULT_HYPERPARAMETERS, effectiveHyperparameters } from "./ai";
import { DEFAULT_CURRICULUM, SNAKES } from "./constants";
import type {
  Brain, CurriculumConfig, GameState, Hyperparameters, LegacyTrainingSummary, PersistedProfile,
  PersistedRoster, SnakeDefinition,
} from "./types";

const LEGACY_STORAGE_KEY = "neon-serpents:profiles:v1";
const V2_STORAGE_KEY = "neon-serpents:roster:v2";
const OLD_APP_KEY = "app-state:v3";
const APP_KEY = "app-state:v4";
const storage = () => typeof indexedDB === "undefined" ? undefined : createStore("neon-serpents", "profiles");

export const MIN_ACTIVE_SNAKES = 2;
export const MAX_ACTIVE_SNAKES = 8;
const HEX_COLOR = /^#[0-9a-f]{6}$/i;

export type RosterMutationResult = { profiles: PersistedProfile[]; error?: string };
export type LoadResult = { roster: PersistedRoster; migrated: boolean; persistent: boolean; error?: string };

export function defaultProfiles(parameters = DEFAULT_HYPERPARAMETERS): PersistedProfile[] {
  return SNAKES.map((snake, index) => ({
    snakeId: snake.id,
    name: snake.name,
    color: snake.color,
    accent: snake.accent,
    active: true,
    createdAt: index,
    highScore: 0,
    wins: 0,
    matches: 0,
    brain: createBrain(snake.id, parameters),
  }));
}

export function defaultRoster(): PersistedRoster {
  return {
    version: 6,
    profiles: defaultProfiles(),
    hyperparameters: { ...DEFAULT_HYPERPARAMETERS },
    curriculum: { ...DEFAULT_CURRICULUM },
  };
}

function normalizeCurriculum(value: unknown): CurriculumConfig {
  const candidate = value && typeof value === "object" ? value as Partial<CurriculumConfig> & {
    survivalEpisodes?: number; battleEpisodes?: number; mixedRatio?: number;
  } : {};
  const n = (key: keyof CurriculumConfig, min: number, max: number) =>
    Math.max(min, Math.min(max, finiteNumber(candidate[key], DEFAULT_CURRICULUM[key])));
  const rawRatios = [
    n("survivalRatio", 0, 1), n("powerUpRatio", 0, 1), n("safeZoneRatio", 0, 1),
    n("hazardRatio", 0, 1), n("objectiveRatio", 0, 1), n("battleRatio", 0, 1), n("seriesRatio", 0, 1),
  ];
  const ratioTotal = rawRatios.reduce((sum, value) => sum + value, 0) || 1;
  const opponentRatios = [
    n("currentOpponentRatio", 0, 1), n("historicalOpponentRatio", 0, 1), n("scriptedOpponentRatio", 0, 1),
  ];
  const opponentTotal = opponentRatios.reduce((sum, value) => sum + value, 0) || 1;
  return {
    navigationWarmupSteps: Math.round(n("navigationWarmupSteps", 1_000, 10_000_000)),
    powerUpWarmupSteps: Math.round(n("powerUpWarmupSteps", 1_000, 10_000_000)),
    safeZoneWarmupSteps: Math.round(n("safeZoneWarmupSteps", 1_000, 10_000_000)),
    hazardWarmupSteps: Math.round(n("hazardWarmupSteps", 1_000, 10_000_000)),
    objectiveWarmupSteps: Math.round(n("objectiveWarmupSteps", 1_000, 10_000_000)),
    survivalRatio: rawRatios[0] / ratioTotal,
    powerUpRatio: rawRatios[1] / ratioTotal,
    safeZoneRatio: rawRatios[2] / ratioTotal,
    hazardRatio: rawRatios[3] / ratioTotal,
    objectiveRatio: rawRatios[4] / ratioTotal,
    battleRatio: rawRatios[5] / ratioTotal,
    seriesRatio: rawRatios[6] / ratioTotal,
    battleSize: Math.round(n("battleSize", 2, 8)),
    checkpointIntervalEpisodes: Math.round(n("checkpointIntervalEpisodes", 10, 100_000)),
    checkpointRetention: Math.round(n("checkpointRetention", 1, 20)),
    currentOpponentRatio: opponentRatios[0] / opponentTotal,
    historicalOpponentRatio: opponentRatios[1] / opponentTotal,
    scriptedOpponentRatio: opponentRatios[2] / opponentTotal,
  };
}

function finiteNumber(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

export function normalizeHyperparameters(value: unknown): Hyperparameters {
  const candidate = value && typeof value === "object" ? value as Partial<Hyperparameters> : {};
  const number = (key: keyof Hyperparameters, min: number, max: number) =>
    Math.max(min, Math.min(max, finiteNumber(candidate[key], DEFAULT_HYPERPARAMETERS[key])));
  return {
    learningRate: number("learningRate", .00001, .01),
    discountFactor: number("discountFactor", .8, .999),
    epsilonStart: number("epsilonStart", .1, 1),
    epsilonMin: number("epsilonMin", .001, .5),
    epsilonDecaySteps: Math.round(number("epsilonDecaySteps", 1_000, 10_000_000)),
    batchSize: Math.round(number("batchSize", 16, 128)),
    replayBufferSize: Math.round(number("replayBufferSize", 1_000, 50_000)),
    warmupTransitions: Math.round(number("warmupTransitions", 0, 10_000)),
    trainEverySteps: Math.round(number("trainEverySteps", 1, 32)),
    targetSyncSteps: Math.round(number("targetSyncSteps", 100, 20_000)),
    nStep: Math.round(number("nStep", 1, 10)),
    priorityAlpha: number("priorityAlpha", 0, 1),
    priorityBetaStart: number("priorityBetaStart", 0, 1),
  };
}

function validBrain(value: unknown, snakeId: string, parameters = DEFAULT_HYPERPARAMETERS): Brain {
  const fresh = createBrain(snakeId, parameters);
  if (!value || typeof value !== "object") return fresh;
  const brain = value as Partial<Brain> & { modelVersion?: number; trainingSpecVersion?: number };
  if (brain.modelVersion !== 3 || brain.trainingSpecVersion !== 3 || brain.observationSize !== fresh.observationSize) return fresh;
  return {
    ...fresh,
    modelKey: typeof brain.modelKey === "string" && brain.modelKey ? brain.modelKey : fresh.modelKey,
    generation: finiteNumber(brain.generation),
    episodes: finiteNumber(brain.episodes),
    epsilon: finiteNumber(brain.epsilon, fresh.epsilon),
    bestScore: finiteNumber(brain.bestScore),
    totalFood: finiteNumber(brain.totalFood),
    wins: finiteNumber(brain.wins),
    environmentSteps: finiteNumber(brain.environmentSteps),
    learningSteps: finiteNumber(brain.learningSteps),
    lastLoss: brain.lastLoss === null ? null : finiteNumber(brain.lastLoss, 0),
    scenarioSteps: {
      survival: finiteNumber(brain.scenarioSteps?.survival),
      powerup: finiteNumber(brain.scenarioSteps?.powerup),
      safezone: finiteNumber(brain.scenarioSteps?.safezone),
      hazard: finiteNumber(brain.scenarioSteps?.hazard),
      objective: finiteNumber(brain.scenarioSteps?.objective),
      battle: finiteNumber(brain.scenarioSteps?.battle),
      series: finiteNumber(brain.scenarioSteps?.series),
    },
    powerUpOpportunities: finiteNumber(brain.powerUpOpportunities),
    powerUpsClaimed: finiteNumber(brain.powerUpsClaimed),
    powerUpApproachMisses: finiteNumber(brain.powerUpApproachMisses),
    rareFoodClaims: finiteNumber(brain.rareFoodClaims),
    objectiveCaptures: finiteNumber(brain.objectiveCaptures),
    bountyKills: finiteNumber(brain.bountyKills),
    hazardDeaths: finiteNumber(brain.hazardDeaths),
    zoneDeaths: finiteNumber(brain.zoneDeaths),
    lastBenchmark: brain.lastBenchmark ?? null,
  };
}

function legacySummary(brain: unknown): LegacyTrainingSummary | undefined {
  if (!brain || typeof brain !== "object") return undefined;
  const value = brain as Record<string, unknown>;
  const q = value.q && typeof value.q === "object" ? value.q as object : {};
  return {
    generation: finiteNumber(value.generation),
    episodes: finiteNumber(value.episodes),
    epsilon: finiteNumber(value.epsilon, .82),
    bestScore: finiteNumber(value.bestScore),
    totalFood: finiteNumber(value.totalFood),
    wins: finiteNumber(value.wins),
    learnedStates: Object.keys(q).length,
  };
}

function normalizeProfile(value: unknown, index: number, parameters: Hyperparameters): PersistedProfile | null {
  if (!value || typeof value !== "object") return null;
  const profile = value as Partial<PersistedProfile>;
  if (typeof profile.snakeId !== "string" || !profile.snakeId || typeof profile.name !== "string" || !profile.name.trim()) return null;
  const fallback = SNAKES.find((snake) => snake.id === profile.snakeId);
  const color = typeof profile.color === "string" && HEX_COLOR.test(profile.color) ? profile.color : fallback?.color ?? "#68f7c1";
  const accent = typeof profile.accent === "string" && HEX_COLOR.test(profile.accent) ? profile.accent : fallback?.accent ?? lightenColor(color);
  const overrides = profile.hyperparameterOverrides && typeof profile.hyperparameterOverrides === "object"
    ? profile.hyperparameterOverrides : undefined;
  return {
    snakeId: profile.snakeId,
    name: profile.name.trim().slice(0, 24),
    color,
    accent,
    active: profile.active !== false,
    createdAt: finiteNumber(profile.createdAt, index),
    highScore: finiteNumber(profile.highScore),
    wins: finiteNumber(profile.wins),
    matches: finiteNumber(profile.matches),
    brain: validBrain(profile.brain, profile.snakeId, effectiveHyperparameters(parameters, overrides)),
    legacyTraining: profile.legacyTraining ?? (profile.brain && (profile.brain as { modelVersion?: number }).modelVersion !== 3 ? legacySummary(profile.brain) : undefined),
    hyperparameterOverrides: overrides,
  };
}

function ensureActiveBounds(input: PersistedProfile[], parameters = DEFAULT_HYPERPARAMETERS): PersistedProfile[] {
  const deduplicated = input.filter((profile, index, profiles) => profiles.findIndex((item) => item.snakeId === profile.snakeId) === index);
  let profiles = deduplicated.length ? deduplicated : defaultProfiles(parameters);
  let activeCount = profiles.filter((profile) => profile.active).length;
  if (activeCount > MAX_ACTIVE_SNAKES) profiles = profiles.map((profile) => {
    if (!profile.active || activeCount <= MAX_ACTIVE_SNAKES) return profile;
    activeCount -= 1;
    return { ...profile, active: false };
  });
  if (activeCount < MIN_ACTIVE_SNAKES) profiles = profiles.map((profile) => {
    if (profile.active || activeCount >= MIN_ACTIVE_SNAKES) return profile;
    activeCount += 1;
    return { ...profile, active: true };
  });
  for (const fallback of defaultProfiles(parameters)) {
    if (activeCount >= MIN_ACTIVE_SNAKES) break;
    if (!profiles.some((profile) => profile.snakeId === fallback.snakeId)) { profiles.push(fallback); activeCount += 1; }
  }
  return profiles;
}

function parseJson(raw: string | null): unknown {
  if (!raw) return null;
  try { return JSON.parse(raw); } catch { return null; }
}

export function migrateLegacyProfiles(value: unknown, parameters = DEFAULT_HYPERPARAMETERS): PersistedProfile[] {
  const entries = Array.isArray(value) ? value as Array<Record<string, unknown>> : [];
  if (!entries.length) return defaultProfiles(parameters);
  const source = entries;
  const profiles = source.map((raw, index): PersistedProfile | null => {
    const profile = normalizeProfile(raw, index, parameters);
    if (!profile) return null;
    const legacyTraining = legacySummary(raw.brain);
    return { ...profile, brain: createBrain(profile.snakeId, parameters), ...(legacyTraining ? { legacyTraining } : {}) };
  }).filter((profile): profile is PersistedProfile => Boolean(profile));
  const existingIds = new Set(profiles.map((profile) => profile.snakeId));
  for (const fallback of defaultProfiles(parameters)) if (!existingIds.has(fallback.snakeId) && entries.length < 4) profiles.push(fallback);
  return ensureActiveBounds(profiles, parameters);
}

export function deserializeRoster(v2Raw: string | null, legacyRaw: string | null): PersistedRoster {
  const raw = parseJson(v2Raw) as { version?: number; profiles?: unknown[] } | null;
  const source = raw?.version === 2 && Array.isArray(raw.profiles) ? raw.profiles : parseJson(legacyRaw);
  return {
    version: 6,
    profiles: migrateLegacyProfiles(source),
    hyperparameters: { ...DEFAULT_HYPERPARAMETERS },
    curriculum: { ...DEFAULT_CURRICULUM },
    migratedAt: Date.now(),
  };
}

export function normalizeRoster(value: unknown): PersistedRoster | null {
  if (!value || typeof value !== "object") return null;
  const roster = value as { version?: unknown; profiles?: unknown; hyperparameters?: unknown; curriculum?: unknown; migratedAt?: unknown };
  if (![3, 4, 5, 6].includes(Number(roster.version)) || !Array.isArray(roster.profiles)) return null;
  const hyperparameters = normalizeHyperparameters(roster.hyperparameters);
  const profiles = roster.profiles.map((profile, index) => normalizeProfile(profile, index, hyperparameters))
    .filter((profile): profile is PersistedProfile => Boolean(profile));
  return {
    version: 6,
    profiles: ensureActiveBounds(profiles, hyperparameters),
    hyperparameters,
    curriculum: normalizeCurriculum(roster.curriculum),
    migratedAt: typeof roster.migratedAt === "number" ? roster.migratedAt : undefined,
  };
}

export async function loadAppData(): Promise<LoadResult> {
  const store = storage();
  if (!store || typeof window === "undefined") return { roster: defaultRoster(), migrated: false, persistent: false };
  try {
    const existing = normalizeRoster(await get(APP_KEY, store));
    if (existing) return { roster: existing, migrated: false, persistent: true };
    const previous = normalizeRoster(await get(OLD_APP_KEY, store));
    if (previous) return { roster: previous, migrated: true, persistent: true };
    const roster = deserializeRoster(window.localStorage.getItem(V2_STORAGE_KEY), window.localStorage.getItem(LEGACY_STORAGE_KEY));
    return { roster, migrated: true, persistent: true };
  } catch (error) {
    return { roster: defaultRoster(), migrated: false, persistent: false, error: error instanceof Error ? error.message : "IndexedDB is unavailable." };
  }
}

export async function saveRoster(roster: PersistedRoster): Promise<void> {
  const store = storage();
  if (!store) throw new Error("IndexedDB is unavailable in this browser.");
  await set(APP_KEY, normalizeRoster(roster) ?? defaultRoster(), store);
}

export async function saveProfiles(profiles: PersistedProfile[], hyperparameters = DEFAULT_HYPERPARAMETERS, curriculum = DEFAULT_CURRICULUM): Promise<void> {
  await saveRoster({ version: 6, profiles, hyperparameters, curriculum });
}

export function activeProfiles(profiles: PersistedProfile[]): PersistedProfile[] {
  return profiles.filter((profile) => profile.active).sort((a, b) => a.createdAt - b.createdAt);
}
export function archivedProfiles(profiles: PersistedProfile[]): PersistedProfile[] {
  return profiles.filter((profile) => !profile.active).sort((a, b) => a.createdAt - b.createdAt);
}
export function profileToDefinition(profile: PersistedProfile): SnakeDefinition {
  return { id: profile.snakeId, name: profile.name, color: profile.color, accent: profile.accent };
}

export function lightenColor(color: string, amount = .68): string {
  const value = HEX_COLOR.test(color) ? color.slice(1) : "68f7c1";
  const channels = [0, 2, 4].map((offset) => parseInt(value.slice(offset, offset + 2), 16));
  return `#${channels.map((channel) => Math.round(channel + (255 - channel) * amount).toString(16).padStart(2, "0")).join("")}`;
}

export function validateSnakeName(name: string, profiles: PersistedProfile[]): string | null {
  const trimmed = name.trim();
  if (trimmed.length < 2 || trimmed.length > 24) return "Use a name between 2 and 24 characters.";
  if (profiles.some((profile) => profile.name.toLocaleLowerCase() === trimmed.toLocaleLowerCase())) return "That snake name already exists.";
  return null;
}

function createSnakeId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return `snake-${crypto.randomUUID()}`;
  return `snake-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 9)}`;
}

export function addSnake(profiles: PersistedProfile[], input: { name: string; color: string; id?: string; createdAt?: number }, parameters = DEFAULT_HYPERPARAMETERS): RosterMutationResult {
  if (activeProfiles(profiles).length >= MAX_ACTIVE_SNAKES) return { profiles, error: `The active roster is limited to ${MAX_ACTIVE_SNAKES} snakes.` };
  const error = validateSnakeName(input.name, profiles);
  if (error) return { profiles, error };
  if (!HEX_COLOR.test(input.color)) return { profiles, error: "Choose a valid six-digit color." };
  const snakeId = input.id ?? createSnakeId();
  if (profiles.some((profile) => profile.snakeId === snakeId)) return { profiles, error: "That snake identifier already exists." };
  return { profiles: [...profiles, {
    snakeId, name: input.name.trim(), color: input.color.toLowerCase(), accent: lightenColor(input.color), active: true,
    createdAt: input.createdAt ?? Date.now(), highScore: 0, wins: 0, matches: 0, brain: createBrain(snakeId, parameters),
  }] };
}

export function archiveSnake(profiles: PersistedProfile[], snakeId: string): RosterMutationResult {
  const target = profiles.find((profile) => profile.snakeId === snakeId);
  if (!target?.active) return { profiles, error: "Active snake not found." };
  if (activeProfiles(profiles).length <= MIN_ACTIVE_SNAKES) return { profiles, error: `Keep at least ${MIN_ACTIVE_SNAKES} active snakes.` };
  return { profiles: profiles.map((profile) => profile.snakeId === snakeId ? { ...profile, active: false } : profile) };
}
export function restoreSnake(profiles: PersistedProfile[], snakeId: string): RosterMutationResult {
  const target = profiles.find((profile) => profile.snakeId === snakeId);
  if (!target || target.active) return { profiles, error: "Archived snake not found." };
  if (activeProfiles(profiles).length >= MAX_ACTIVE_SNAKES) return { profiles, error: `The active roster is limited to ${MAX_ACTIVE_SNAKES} snakes.` };
  return { profiles: profiles.map((profile) => profile.snakeId === snakeId ? { ...profile, active: true } : profile) };
}

export function mergeBrains(profiles: PersistedProfile[], brains: Brain[]): PersistedProfile[] {
  const byId = new Map(brains.map((brain) => [brain.snakeId, brain]));
  return profiles.map((profile) => ({ ...profile, brain: byId.get(profile.snakeId) ?? profile.brain }));
}

export function recordMatch(profiles: PersistedProfile[], state: GameState): PersistedProfile[] {
  return profiles.map((profile) => {
    const snake = state.snakes.find((entry) => entry.id === profile.snakeId);
    if (!snake) return profile;
    return { ...profile, highScore: Math.max(profile.highScore, snake.score), matches: profile.matches + 1, wins: profile.wins + Number(state.winnerId === snake.id) };
  });
}
