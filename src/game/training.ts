import { createGame } from "./engine";
import type { GameState, SnakeDefinition } from "./types";

export function createVisualTrainingState(snake: SnakeDefinition, seed: number): GameState {
  return createGame({ mode: "training", snakes: [snake], seed });
}
