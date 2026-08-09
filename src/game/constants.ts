import type { ArenaPhase, CurriculumConfig, Direction, MapArchetype, Point, PowerUpKind, SnakeDefinition } from "./types";

export const BOARD_WIDTH = 72;
export const BOARD_HEIGHT = 44;
export const FOOD_COUNT = 5;
export const ADAPTIVE_PHASE_ENDS: Record<ArenaPhase, number> = { opening: 500, midgame: 1_200, endgame: 2_000 };
export const PHASE_FOOD_COUNT: Record<ArenaPhase, number> = { opening: 7, midgame: 4, endgame: 2 };
export const PHASE_FOOD_LIFETIME: Record<ArenaPhase, number> = { opening: 300, midgame: 180, endgame: 90 };
export const SAFE_ZONE_TELEGRAPH_TICKS = 45;
export const SAFE_ZONE_INTERVAL_TICKS = 90;
export const OBJECTIVE_CAPTURE_TICKS = 45;
export const OBJECTIVE_LIFETIME_TICKS = 180;
export const OBJECTIVE_COOLDOWN_TICKS = 45;
export const MAP_ARCHETYPES: MapArchetype[] = ["open", "corridors", "islands", "fortress", "quadrants", "hazard"];

export const DEFAULT_CURRICULUM: CurriculumConfig = {
  navigationWarmupSteps: 50_000,
  powerUpWarmupSteps: 50_000,
  safeZoneWarmupSteps: 75_000,
  hazardWarmupSteps: 75_000,
  objectiveWarmupSteps: 100_000,
  survivalRatio: .10,
  powerUpRatio: .15,
  safeZoneRatio: .15,
  hazardRatio: .10,
  objectiveRatio: .15,
  battleRatio: .20,
  seriesRatio: .15,
  battleSize: 4,
  checkpointIntervalEpisodes: 250,
  checkpointRetention: 5,
  currentOpponentRatio: .50,
  historicalOpponentRatio: .30,
  scriptedOpponentRatio: .20,
};
export const MAX_TRAINING_TICKS = 900;
export const MAX_POWERUP_TRAINING_TICKS = 360;

export const SNAKES: SnakeDefinition[] = [
  { id: "nova", name: "Nova Viper", color: "#68f7c1", accent: "#d7fff1" },
  { id: "ember", name: "Ember Fang", color: "#ff6b7a", accent: "#ffd7dc" },
  { id: "volt", name: "Volt Coil", color: "#ffd166", accent: "#fff1bd" },
  { id: "echo", name: "Echo Wyrm", color: "#9d83ff", accent: "#e5dcff" },
];

export const POWER_UP_KINDS: PowerUpKind[] = [
  "shield", "phase", "haste", "double", "magnet", "growth",
  "trim", "secondChance", "warp", "freeze", "crown", "vision",
];

export const POWER_UP_META: Record<PowerUpKind, { name: string; icon: string; color: string; description: string }> = {
  shield: { name: "Aegis", icon: "◇", color: "#62e6ff", description: "Absorbs the next collision." },
  phase: { name: "Ghost Drive", icon: "◌", color: "#c49bff", description: "Pass through bodies and obstacles." },
  haste: { name: "Overclock", icon: "»", color: "#ffe066", description: "Move twice per battle tick." },
  double: { name: "Score Surge", icon: "×2", color: "#ff9f6e", description: "Double food and survival points." },
  magnet: { name: "Food Magnet", icon: "∪", color: "#ff6fb5", description: "Collect nearby food from a distance." },
  growth: { name: "Mega Meal", icon: "+", color: "#7dff8a", description: "Gain four segments and bonus score." },
  trim: { name: "Tail Trim", icon: "✂", color: "#8ee8ff", description: "Shorten to a safer, nimble body." },
  secondChance: { name: "Reboot", icon: "↻", color: "#ffffff", description: "Escape one fatal collision." },
  warp: { name: "Safe Warp", icon: "◎", color: "#81a7ff", description: "Teleport to the safest open cell." },
  freeze: { name: "Cryo Pulse", icon: "✦", color: "#9cf5ff", description: "Briefly freeze every rival." },
  crown: { name: "Kingmaker", icon: "♛", color: "#ffd45b", description: "Earn score while surviving." },
  vision: { name: "Oracle", icon: "◉", color: "#b6ff9e", description: "See danger two cells ahead." },
};

export const DIRECTION_VECTOR: Record<Direction, Point> = {
  up: { x: 0, y: -1 },
  right: { x: 1, y: 0 },
  down: { x: 0, y: 1 },
  left: { x: -1, y: 0 },
};

export const DIRECTIONS: Direction[] = ["up", "right", "down", "left"];
