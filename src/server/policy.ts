import type { RelativeAction, TrainingScenario } from "../game/types";

export type WireTensor = { name: string; shape: number[]; values: number[] };
export type WirePolicy = {
  snakeId: string; version: number; epsilon: number; environmentSteps: number;
  scenarioSteps: Pick<Record<TrainingScenario, number>, "survival" | "powerup" | "battle"> & Partial<Record<TrainingScenario, number>>;
  tensors: WireTensor[];
};

function tensor(policy: WirePolicy, name: string): WireTensor {
  const found = policy.tensors.find((entry) => entry.name === name);
  if (!found) throw new Error(`Policy tensor ${name} is missing.`);
  return found;
}

export class DensePolicy {
  readonly version: number;
  readonly epsilon: number;
  private readonly w1: WireTensor;
  private readonly b1: WireTensor;
  private readonly w2: WireTensor;
  private readonly b2: WireTensor;
  private readonly w3: WireTensor;
  private readonly b3: WireTensor;

  constructor(readonly wire: WirePolicy) {
    this.version = wire.version;
    this.epsilon = wire.epsilon;
    this.w1 = tensor(wire, "fc1.weight");
    this.b1 = tensor(wire, "fc1.bias");
    this.w2 = tensor(wire, "fc2.weight");
    this.b2 = tensor(wire, "fc2.bias");
    this.w3 = tensor(wire, "out.weight");
    this.b3 = tensor(wire, "out.bias");
  }

  private dense(input: Float32Array, weights: WireTensor, bias: WireTensor, relu: boolean): Float32Array {
    const [outputs, inputs] = weights.shape;
    if (input.length !== inputs || bias.values.length !== outputs) throw new Error("Policy tensor shape is incompatible.");
    const result = new Float32Array(outputs);
    for (let output = 0; output < outputs; output += 1) {
      let value = bias.values[output];
      const offset = output * inputs;
      for (let index = 0; index < inputs; index += 1) value += weights.values[offset + index] * input[index];
      result[output] = relu ? Math.max(0, value) : value;
    }
    return result;
  }

  values(observation: ArrayLike<number>): Float32Array {
    const input = observation instanceof Float32Array ? observation : Float32Array.from(observation);
    return this.dense(this.dense(this.dense(input, this.w1, this.b1, true), this.w2, this.b2, true), this.w3, this.b3, false);
  }

  action(observation: ArrayLike<number>, safeActions: RelativeAction[] | undefined, explore: boolean, random: () => number): RelativeAction {
    if (explore && random() < this.epsilon) {
      const options = safeActions?.length ? safeActions : [0, 1, 2] as RelativeAction[];
      return options[Math.floor(random() * options.length)];
    }
    const values = Array.from(this.values(observation));
    if (safeActions?.length) for (let index = 0; index < values.length; index += 1) if (!safeActions.includes(index as RelativeAction)) values[index] = -Infinity;
    return values.indexOf(Math.max(...values)) as RelativeAction;
  }
}
