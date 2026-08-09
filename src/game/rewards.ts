// Central reward constants and shaping helpers. Kept as small pure functions so
// the engine and the tests stay in sync. Potential-based shaping rewards a snake
// when it moves *toward* a target without biasing the optimal policy.
export const REWARD = {
  step: -0.04,        // per-move efficiency penalty
  survival: 0.02,     // per-tick survival pressure in battle phase
  food: 35,           // per food eaten (before multiplier)
  foodApproach: 0.2,  // potential step when net closer to food
  foodAway: -0.2,     // potential step when net farther from food
  foodStall: -0.05,   // no progress toward food
  powerUpClaim: 30,   // per power-up claimed
  powerUpApproach: 5, // scale for discounted, normalized potential shaping
  kill: 20,           // attacker reward for eliminating a rival
  death: -120,        // fatal collision
  win: 80,            // last snake standing
} as const;

export const stepPenalty = (): number => REWARD.step;
export const survivalReward = (): number => REWARD.survival;
export const foodEatReward = (multiplier: number): number => REWARD.food * multiplier;
export const powerUpClaimReward = (): number => REWARD.powerUpClaim;
export const killReward = (): number => REWARD.kill;
export const deathReward = (): number => REWARD.death;
export const winReward = (): number => REWARD.win;

/** Distance-delta shaping for food: positive when net closer, negative when farther. */
export function foodDistanceDeltaReward(before: number, after: number): number {
  if (after < before) return REWARD.foodApproach;
  if (after > before) return REWARD.foodAway;
  return REWARD.foodStall;
}

/**
 * Potential-based power-up approaching reward. Only meaningful while a single
 * power-up is live on the board. Returns a positive reward for reducing the
 * Manhattan distance to it, negative for moving away, zero when none is present.
 */
export function powerUpApproachReward(
  before: number,
  after: number,
  present: boolean,
  discount = .95,
  claimed = false,
): number {
  if (!present) return 0;
  const beforePotential = -Math.max(0, Math.min(1, before));
  const afterPotential = claimed ? 0 : -Math.max(0, Math.min(1, after));
  return REWARD.powerUpApproach * (discount * afterPotential - beforePotential);
}
