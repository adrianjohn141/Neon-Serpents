import { createBrain } from "./ai";
import { OBSERVATION_LAYOUT } from "./observation";
import type { CurriculumAgent } from "./curriculum";
import type { Experience, RelativeAction } from "./types";

export type ScriptedOpponentStyle = "aggressive" | "defensive" | "food" | "powerup" | "trap" | "leaderHunter";

export class ScriptedOpponent implements CurriculumAgent {
  brain;

  constructor(snakeId: string, private style: ScriptedOpponentStyle) {
    this.brain = createBrain(snakeId);
  }

  chooseAction(observation: ArrayLike<number>, _explore: boolean, safeActions: RelativeAction[] = [0, 1, 2]): RelativeAction {
    const safe: RelativeAction[] = safeActions.length ? safeActions : [0, 1, 2];
    if (this.style === "defensive") return safe.includes(0) ? 0 : safe[0];
    let vectorStart: number = OBSERVATION_LAYOUT.food.start;
    if (this.style === "powerup" && observation[OBSERVATION_LAYOUT.powerUp.start] > .5) vectorStart = OBSERVATION_LAYOUT.powerUp.start + 14;
    if (this.style === "aggressive") vectorStart = OBSERVATION_LAYOUT.enemies.start;
    if (this.style === "leaderHunter" && observation[OBSERVATION_LAYOUT.bounty.start] > .5) vectorStart = OBSERVATION_LAYOUT.bounty.start + 1;
    const lateral = observation[vectorStart] ?? 0;
    const forward = observation[vectorStart + 1] ?? 0;
    let preferred: RelativeAction = forward >= Math.abs(lateral) * .5 ? 0 : lateral < 0 ? 1 : 2;
    if (this.style === "trap") {
      const rank = observation[OBSERVATION_LAYOUT.self.start + 1] ?? .5;
      preferred = rank > .5 ? 1 : 2;
    }
    if (safe.includes(preferred)) return preferred;
    return safe.includes(0) ? 0 : safe[0];
  }

  remember(_experience: Experience): void {}
  async train(): Promise<number | null> { return null; }
  finishEpisode(): void {}
}
