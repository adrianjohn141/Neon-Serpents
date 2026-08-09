export type Point = { x: number; y: number };
export type Direction = "up" | "right" | "down" | "left";
export type RelativeAction = 0 | 1 | 2;
export type GameMode = "battle" | "training" | "powerup" | "safezone" | "hazard" | "objective";
export type GameStatus = "running" | "finished";
export type ArenaPhase = "opening" | "midgame" | "endgame";
export type MapArchetype = "open" | "corridors" | "islands" | "fortress" | "quadrants" | "hazard";

export type PowerUpKind =
  | "shield" | "phase" | "haste" | "double" | "magnet" | "growth"
  | "trim" | "secondChance" | "warp" | "freeze" | "crown" | "vision";

export type PowerUp = { id: number; kind: PowerUpKind; position: Point; expiresAt: number };
export type ArenaFood = { id: number; position: Point; kind: "normal" | "rare"; value: 1 | 3; spawnedAt: number; expiresAt: number };

export type ArenaBounds = { inset: number; pendingInset: number; telegraphAt: number; closesAt: number; minimumWidth: number; minimumHeight: number };
export type HazardKind = "laser" | "sweeper" | "blocks";
export type ArenaHazard = {
  id: number;
  kind: HazardKind;
  origin: Point;
  direction: Direction;
  length: number;
  telegraphAt: number;
  activatesAt: number;
  activeUntil: number;
};
export type EnergyCore = {
  id: number;
  position: Point;
  radius: number;
  expiresAt: number;
  captureRequired: number;
  progress: Record<string, number>;
  contested: boolean;
};
export type OpponentTarget = "survival" | "food" | "powerup" | "objective" | "leader";
export type OpponentProfile = {
  turnBias: number;
  aggressionRate: number;
  commonTarget: OpponentTarget;
  powerUpRate: number;
  typicalDeathCause: DeathCause;
  samples: number;
};
export type ArenaState = {
  enabled: boolean;
  phase: ArenaPhase;
  phaseEndsAt: number;
  mapArchetype: MapArchetype;
  safeZone: ArenaBounds;
  objective: EnergyCore | null;
  nextObjectiveAt: number;
  hazards: ArenaHazard[];
  nextHazardAt: number;
  dormantObstacles: Point[];
  obstaclesActivateAt: number;
  seriesRound: 1 | 2 | 3;
  opponentProfiles: Record<string, OpponentProfile>;
  leaderId: string | null;
};

export type Buffs = {
  shield: number;
  phaseUntil: number;
  hasteUntil: number;
  doubleUntil: number;
  magnetUntil: number;
  secondChance: number;
  frozenUntil: number;
  crownUntil: number;
  visionUntil: number;
};

export type Snake = {
  id: string;
  name: string;
  color: string;
  accent: string;
  segments: Point[];
  direction: Direction;
  alive: boolean;
  score: number;
  foodEaten: number;
  powerUps: number;
  rareFoodEaten: number;
  objectiveCaptures: number;
  bountyKills: number;
  kills: number;
  deathReason?: string;
  buffs: Buffs;
};

export type GameEvent = { id: number; tick: number; text: string; tone: "info" | "power" | "danger" | "victory" };

export type GameState = {
  width: number;
  height: number;
  seed: number;
  tick: number;
  mode: GameMode;
  status: GameStatus;
  winnerId: string | null;
  snakes: Snake[];
  food: ArenaFood[];
  obstacles: Point[];
  powerUp: PowerUp | null;
  powerUpSpawnCount: number;
  nextPowerUpAt: number;
  arena: ArenaState;
  events: GameEvent[];
};

export type SnakeDefinition = Pick<Snake, "id" | "name" | "color" | "accent">;
export type GameConfig = {
  seed?: number;
  mode?: GameMode;
  snakes?: SnakeDefinition[];
  width?: number;
  height?: number;
  adaptiveArena?: boolean;
  mapArchetype?: MapArchetype;
  seriesRound?: 1 | 2 | 3;
  opponentProfiles?: Record<string, OpponentProfile>;
};
export type RewardBreakdown = {
  step: number;
  survival: number;
  foodApproach: number;
  foodClaim: number;
  powerUpApproach: number;
  powerUpClaim: number;
  zonePositioning: number;
  objectiveApproach: number;
  objectiveCapture: number;
  bountyKill: number;
  rareFoodClaim: number;
  kill: number;
  death: number;
  win: number;
};

export type StepResult = {
  state: GameState;
  rewards: Record<string, number>;
  rewardBreakdowns: Record<string, RewardBreakdown>;
};

export type Hyperparameters = {
  learningRate: number;
  discountFactor: number;
  epsilonStart: number;
  epsilonMin: number;
  epsilonDecaySteps: number;
  batchSize: number;
  replayBufferSize: number;
  warmupTransitions: number;
  trainEverySteps: number;
  targetSyncSteps: number;
  nStep: number;
  priorityAlpha: number;
  priorityBetaStart: number;
};

export type TrainingScenario = "survival" | "powerup" | "safezone" | "hazard" | "objective" | "battle" | "series";

export type CurriculumConfig = {
  navigationWarmupSteps: number;
  powerUpWarmupSteps: number;
  safeZoneWarmupSteps: number;
  hazardWarmupSteps: number;
  objectiveWarmupSteps: number;
  survivalRatio: number;
  powerUpRatio: number;
  safeZoneRatio: number;
  hazardRatio: number;
  objectiveRatio: number;
  battleRatio: number;
  seriesRatio: number;
  battleSize: number;
  checkpointIntervalEpisodes: number;
  checkpointRetention: number;
  currentOpponentRatio: number;
  historicalOpponentRatio: number;
  scriptedOpponentRatio: number;
};

export type LegacyTrainingSummary = {
  generation: number;
  episodes: number;
  epsilon: number;
  bestScore: number;
  totalFood: number;
  wins: number;
  learnedStates: number;
};

export type Brain = {
  snakeId: string;
  modelKey: string;
  modelVersion: 3;
  trainingSpecVersion: 3;
  observationSize: number;
  generation: number;
  episodes: number;
  epsilon: number;
  bestScore: number;
  totalFood: number;
  wins: number;
  environmentSteps: number;
  learningSteps: number;
  lastLoss: number | null;
  scenarioSteps: Record<TrainingScenario, number>;
  powerUpOpportunities: number;
  powerUpsClaimed: number;
  powerUpApproachMisses: number;
  rareFoodClaims: number;
  objectiveCaptures: number;
  bountyKills: number;
  hazardDeaths: number;
  zoneDeaths: number;
  lastBenchmark: BrainBenchmark | null;
};

export type BrainBenchmark = {
  evaluatedAt: number;
  matches: number;
  winRate: number;
  avgSurvivalTicks: number;
  foodPerMatch: number;
  powerUpClaimRate: number;
  deathsPerThousandTicks: number;
};

export type PersistedProfile = {
  snakeId: string;
  name: string;
  color: string;
  accent: string;
  active: boolean;
  createdAt: number;
  highScore: number;
  wins: number;
  matches: number;
  brain: Brain;
  legacyTraining?: LegacyTrainingSummary;
  hyperparameterOverrides?: Partial<Hyperparameters>;
};

export type PersistedRoster = {
  version: 6;
  profiles: PersistedProfile[];
  hyperparameters: Hyperparameters;
  curriculum: CurriculumConfig;
  migratedAt?: number;
};

export type EvalSnakeStats = {
  id: string;
  wins: number;
  avgScore: number;
  powerUpsClaimed: number;
  foodEaten: number;
  avgSurvivalTicks: number;
  powerUpOpportunities: number;
  powerUpClaimRate: number;
  approachWithoutClaimRate: number;
  approachWithoutClaims: number;
  deaths: number;
  deathsPerThousandTicks: number;
  rareFoodClaims: number;
  objectiveCaptures: number;
  bountyKills: number;
  hazardDeaths: number;
  zoneDeaths: number;
  phaseSurvival: Record<ArenaPhase, number>;
  phaseResults: Record<ArenaPhase, { matches: number; wins: number }>;
  mapResults: Record<MapArchetype, { matches: number; wins: number }>;
  rewardBreakdown: RewardBreakdown;
  deathCauses: Record<DeathCause, number>;
  powerUpBehavior: Record<PowerUpKind, PowerUpBehaviorStats>;
  adaptive: AdaptiveBehaviorStats;
};

export type DeathCause = "wall" | "obstacle" | "headOn" | "snakeBody" | "other";
export type TrainingDeathCause = DeathCause | "hazard" | "zone";

export type PowerUpBehaviorStats = {
  seen: number;
  reachable: number;
  pursued: number;
  claimed: number;
  ignored: number;
  pursuitWithoutClaim: number;
  pursuitDeaths: number;
  avgInitialDistance: number | null;
  avgClosestDistance: number | null;
  avgClaimTicks: number | null;
};

export type PowerUpCounterStats = { opportunities: number; successes: number };

export type AdaptiveBehaviorStats = {
  zoneWarnings: number;
  zoneRepositions: number;
  zoneDeaths: number;
  hazardEncounters: number;
  hazardEvasions: number;
  hazardShieldBlocks: number;
  hazardDeaths: number;
  objectiveOpportunities: number;
  objectivePursuits: number;
  objectiveContests: number;
  objectiveCaptures: number;
  objectivePursuitDeaths: number;
  leaderTicks: number;
  bountyKills: number;
  leaderDeaths: number;
  crownClaims: number;
  crownActiveTicks: number;
  crownDeaths: number;
  normalFoodClaims: number;
  rareFoodClaims: number;
  foodExpirationsObserved: number;
  powerUpCounters: Record<PowerUpKind, PowerUpCounterStats>;
};

export type EvalReport = {
  runs: number;
  snakes: EvalSnakeStats[];
  startedAt: number;
  finishedAt: number;
};

export type RayEntity = "wall" | "obstacle" | "self" | "enemy" | "food" | "powerUp";
export type RayHit = { distance: number; entity: RayEntity; point: Point };

export type Experience = {
  state: number[] | Float32Array;
  action: RelativeAction;
  reward: number;
  nextState: number[] | Float32Array;
  terminal: boolean;
  nSteps?: number;
  rewardBreakdown?: RewardBreakdown;
  deathCause?: TrainingDeathCause;
};
