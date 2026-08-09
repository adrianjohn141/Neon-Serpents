import type { Experience } from "./types";

export type PrioritizedSample = {
  experiences: Experience[];
  indices: number[];
  weights: number[];
};

/** Sum-tree prioritized replay with proportional sampling and bias correction. */
export class PrioritizedReplayBuffer {
  private tree: Float64Array;
  private items: Array<Experience | undefined>;
  private cursor = 0;
  private count = 0;
  private maximumPriority = 1;

  constructor(public capacity: number, public alpha = .6) {
    this.tree = new Float64Array(capacity * 2);
    this.items = Array(capacity);
  }

  get size(): number { return this.count; }
  get totalPriority(): number { return this.tree[1] ?? 0; }

  add(experience: Experience, priority = this.maximumPriority): void {
    const index = this.cursor;
    this.items[index] = experience;
    this.update(index, priority);
    this.cursor = (this.cursor + 1) % this.capacity;
    this.count = Math.min(this.capacity, this.count + 1);
  }

  update(index: number, tdError: number): void {
    const raw = Math.max(1e-6, Math.abs(tdError));
    this.maximumPriority = Math.max(this.maximumPriority, raw);
    const priority = Math.pow(raw, this.alpha);
    let treeIndex = index + this.capacity;
    const delta = priority - this.tree[treeIndex];
    while (treeIndex >= 1) {
      this.tree[treeIndex] += delta;
      treeIndex = Math.floor(treeIndex / 2);
    }
  }

  updateMany(indices: number[], tdErrors: number[]): void {
    indices.forEach((index, offset) => this.update(index, tdErrors[offset] ?? this.maximumPriority));
  }

  sample(count: number, beta: number, random = Math.random): PrioritizedSample {
    const take = Math.min(count, this.count);
    if (!take || this.totalPriority <= 0) return { experiences: [], indices: [], weights: [] };
    const experiences: Experience[] = [];
    const indices: number[] = [];
    const probabilities: number[] = [];
    const segment = this.totalPriority / take;
    for (let sampleIndex = 0; sampleIndex < take; sampleIndex += 1) {
      let mass = (sampleIndex + random()) * segment;
      let treeIndex = 1;
      while (treeIndex < this.capacity) {
        const left = treeIndex * 2;
        if (mass <= this.tree[left]) treeIndex = left;
        else {
          mass -= this.tree[left];
          treeIndex = left + 1;
        }
      }
      const dataIndex = treeIndex - this.capacity;
      const experience = this.items[dataIndex];
      if (!experience) continue;
      indices.push(dataIndex);
      experiences.push(experience);
      probabilities.push(this.tree[treeIndex] / this.totalPriority);
    }
    const rawWeights = probabilities.map((probability) => Math.pow(this.count * probability, -beta));
    const maximumWeight = Math.max(1e-12, ...rawWeights);
    return { experiences, indices, weights: rawWeights.map((weight) => weight / maximumWeight) };
  }

  resize(capacity: number): void {
    if (capacity === this.capacity) return;
    const retained: Experience[] = [];
    for (let offset = 0; offset < this.count; offset += 1) {
      const index = (this.cursor - this.count + offset + this.capacity) % this.capacity;
      const item = this.items[index];
      if (item) retained.push(item);
    }
    this.capacity = capacity;
    this.tree = new Float64Array(capacity * 2);
    this.items = Array(capacity);
    this.cursor = 0;
    this.count = 0;
    retained.slice(-capacity).forEach((item) => this.add(item));
  }

  clear(): void {
    this.tree.fill(0);
    this.items.fill(undefined);
    this.cursor = 0;
    this.count = 0;
    this.maximumPriority = 1;
  }
}
