import type { Experience } from "./types";

export class NStepAccumulator {
  private queue: Experience[] = [];

  constructor(private steps: number, private discount: number) {}

  add(experience: Experience): Experience[] {
    this.queue.push(experience);
    const ready: Experience[] = [];
    if (experience.terminal) {
      while (this.queue.length) ready.push(this.build(Math.min(this.steps, this.queue.length)));
    } else if (this.queue.length >= this.steps) {
      ready.push(this.build(this.steps));
    }
    return ready;
  }

  clear(): void { this.queue = []; }

  flush(terminal = true): Experience[] {
    const ready: Experience[] = [];
    while (this.queue.length) {
      const experience = this.build(Math.min(this.steps, this.queue.length));
      ready.push(terminal ? { ...experience, terminal: true } : experience);
    }
    return ready;
  }

  private build(count: number): Experience {
    const window = this.queue.slice(0, count);
    const last = window[window.length - 1];
    const reward = window.reduce((sum, entry, index) => sum + Math.pow(this.discount, index) * entry.reward, 0);
    const result: Experience = {
      state: this.queue[0].state,
      action: this.queue[0].action,
      reward,
      nextState: last.nextState,
      terminal: last.terminal,
      nSteps: count,
    };
    this.queue.shift();
    return result;
  }
}
