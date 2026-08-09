import { describe, expect, it } from "vitest";
import { DensePolicy, type WirePolicy } from "./policy";

function tensor(name: string, shape: number[], values: number[]) { return { name, shape, values }; }

describe("server policy inference", () => {
  it("uses PyTorch row-major dense weights and safe-action masking", () => {
    const policy: WirePolicy = {
      snakeId: "nova", version: 1, epsilon: 0, environmentSteps: 12,
      scenarioSteps: { survival: 4, powerup: 4, battle: 4 },
      tensors: [
        tensor("fc1.weight", [2, 2], [1, 0, 0, 1]), tensor("fc1.bias", [2], [0, 0]),
        tensor("fc2.weight", [2, 2], [1, 0, 0, 1]), tensor("fc2.bias", [2], [0, 0]),
        tensor("out.weight", [3, 2], [1, 0, 0, 1, 1, 1]), tensor("out.bias", [3], [0, 0, 0]),
      ],
    };
    const dense = new DensePolicy(policy);
    expect(Array.from(dense.values([2, 1]))).toEqual([2, 1, 3]);
    expect(dense.action([2, 1], [0, 1], false, () => 0.5)).toBe(0);
  });
});
