import type { Brain, CurriculumConfig, EvalReport, Experience, Hyperparameters, PersistedProfile, RelativeAction, SnakeDefinition, TrainingScenario } from "./types";
import type { BrainBundle } from "./model-bundle";

export type WorkerRequest =
  | { type: "init"; profiles: PersistedProfile[]; hyperparameters: Hyperparameters; curriculum: CurriculumConfig }
  | { type: "act"; requestId: number; observations: Record<string, { data: number[], safeActions: RelativeAction[] }>; explore: boolean }
  | { type: "observe"; experiences: Record<string, Experience>; contexts?: Record<string, { scenario: TrainingScenario; opportunity?: boolean; claimed?: boolean; approachMiss?: boolean }> }
  | { type: "episode"; outcomes: Record<string, { score: number; food: number; won: boolean }> }
  | { type: "startFast"; snakes: SnakeDefinition[] }
  | { type: "evaluate"; runs: number }
  | { type: "stop" }
  | { type: "checkpoint" }
  | { type: "updateParameters"; profiles: PersistedProfile[]; hyperparameters: Hyperparameters }
  | { type: "reset"; snakeIds: string[] }
  | { type: "hiveSync" }
  | { type: "undoHive" }
  | { type: "trainAfterMatch"; steps: number }
  | { type: "exportBundle"; requestId: number; snakeId: string }
  | { type: "importBundle"; requestId: number; snakeId: string; bundle: BrainBundle }
  | { type: "useProductionBundles"; requestId: number; bundles: Record<string, BrainBundle> };

export type WorkerResponse =
  | { type: "ready"; brains: Brain[]; backend: string }
  | { type: "actions"; requestId: number; actions: Record<string, RelativeAction> }
  | { type: "progress"; scenario: TrainingScenario; brains: Brain[]; totalEpisodes: number; totalSteps: number; elapsedMs: number }
  | { type: "brains"; brains: Brain[] }
  | { type: "saved"; brains: Brain[] }
  | { type: "stopped"; brains: Brain[] }
  | { type: "hiveComplete"; brains: Brain[]; canUndo: boolean }
  | { type: "evalReport"; report: EvalReport }
  | { type: "bundle"; requestId: number; bundle: BrainBundle }
  | { type: "bundleImported"; requestId: number; brains: Brain[] }
  | { type: "productionBundlesLoaded"; requestId: number }
  | { type: "error"; message: string; requestId?: number };
