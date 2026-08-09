import { describe, expect, it } from "vitest";
import {
  deathReward, foodDistanceDeltaReward, foodEatReward, killReward, powerUpApproachReward,
  powerUpClaimReward, REWARD, stepPenalty, survivalReward, winReward,
} from "./rewards";

describe("reward shaping", () => {
  it("keeps the moving and terminal reward magnitudes stable", () => {
    expect(stepPenalty()).toBe(REWARD.step);
    expect(survivalReward()).toBe(REWARD.survival);
    expect(deathReward()).toBe(REWARD.death);
    expect(winReward()).toBe(REWARD.win);
    expect(foodEatReward(1)).toBe(REWARD.food);
    expect(foodEatReward(2)).toBe(REWARD.food * 2);
    expect(powerUpClaimReward()).toBe(REWARD.powerUpClaim);
    expect(killReward()).toBe(REWARD.kill);
  });

  it("rewards net progress toward food and punishes drift away", () => {
    expect(foodDistanceDeltaReward(10, 8)).toBe(REWARD.foodApproach);
    expect(foodDistanceDeltaReward(8, 11)).toBe(REWARD.foodAway);
    expect(foodDistanceDeltaReward(5, 5)).toBe(REWARD.foodStall);
  });

  it("potential-shapes power-up approach only while a power-up is present", () => {
    expect(powerUpApproachReward(.8, .5, true)).toBeGreaterThan(0);
    expect(powerUpApproachReward(.5, .8, true)).toBeLessThan(0);
    expect(powerUpApproachReward(7, 4, false)).toBe(0);
  });
});
