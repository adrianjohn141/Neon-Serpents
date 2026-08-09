import { POWER_UP_KINDS } from "./constants";
import { isDangerAt, pointKey, powerUpDistanceField, relativeDirection } from "./engine";
import { hazardCells, insideSafeBounds } from "./adaptive-arena";
import type {
  AdaptiveBehaviorStats, ArenaHazard, DeathCause, GameState, PowerUpBehaviorStats, PowerUpKind,
} from "./types";

export type PowerUpBehaviorRaw = PowerUpBehaviorStats & {
  initialDistanceTotal: number;
  closestDistanceTotal: number;
  claimTicksTotal: number;
};

export function emptyPowerUpBehavior(): Record<PowerUpKind, PowerUpBehaviorRaw> {
  return Object.fromEntries(POWER_UP_KINDS.map((kind) => [kind, {
    seen: 0, reachable: 0, pursued: 0, claimed: 0, ignored: 0, pursuitWithoutClaim: 0, pursuitDeaths: 0,
    avgInitialDistance: null, avgClosestDistance: null, avgClaimTicks: null,
    initialDistanceTotal: 0, closestDistanceTotal: 0, claimTicksTotal: 0,
  }])) as Record<PowerUpKind, PowerUpBehaviorRaw>;
}

export function emptyDeathCauses(): Record<DeathCause, number> {
  return { wall: 0, obstacle: 0, headOn: 0, snakeBody: 0, other: 0 };
}

export function emptyAdaptiveBehavior(): AdaptiveBehaviorStats {
  return {
    zoneWarnings: 0, zoneRepositions: 0, zoneDeaths: 0,
    hazardEncounters: 0, hazardEvasions: 0, hazardShieldBlocks: 0, hazardDeaths: 0,
    objectiveOpportunities: 0, objectivePursuits: 0, objectiveContests: 0,
    objectiveCaptures: 0, objectivePursuitDeaths: 0,
    leaderTicks: 0, bountyKills: 0, leaderDeaths: 0,
    crownClaims: 0, crownActiveTicks: 0, crownDeaths: 0,
    normalFoodClaims: 0, rareFoodClaims: 0, foodExpirationsObserved: 0,
    powerUpCounters: Object.fromEntries(POWER_UP_KINDS.map((kind) => [kind, { opportunities: 0, successes: 0 }])) as AdaptiveBehaviorStats["powerUpCounters"],
  };
}

export function normalizeDeathCause(reason?: string): DeathCause {
  if (reason?.includes("wall")) return "wall";
  if (reason?.includes("obstacle")) return "obstacle";
  if (reason?.includes("head-on")) return "headOn";
  if (reason?.includes("snake")) return "snakeBody";
  return "other";
}

type PursuitRecord = {
  initial: number;
  closest: number;
  reachable: boolean;
  pursued: boolean;
  claimed: boolean;
  pursuitDeathCounted: boolean;
};

type ActivePowerUp = {
  id: number;
  kind: PowerUpKind;
  spawnedAt: number;
  distanceField: Map<string, number> | null;
  records: Map<string, PursuitRecord>;
};

export class PowerUpBehaviorTracker {
  private active: ActivePowerUp | null = null;

  constructor(private readonly metrics: Record<string, Record<PowerUpKind, PowerUpBehaviorRaw>>) {}

  observe(before: GameState, after: GameState): void {
    if (before.powerUp && (!this.active || this.active.id !== before.powerUp.id)) this.start(before);
    if (before.powerUp && this.active?.id === before.powerUp.id) {
      for (const snake of before.snakes.filter((entry) => entry.alive)) {
        const record = this.active.records.get(snake.id);
        const next = after.snakes.find((entry) => entry.id === snake.id);
        if (!record || !next) continue;
        const distance = this.active.distanceField?.get(pointKey(next.segments[0]));
        if (distance !== undefined) record.closest = Math.min(record.closest, distance);
        const threshold = Math.min(3, Math.max(1, Math.ceil(record.initial * .2)));
        if (!record.pursued && record.initial - record.closest >= threshold) {
          record.pursued = true;
          this.metrics[snake.id][this.active.kind].pursued += 1;
        }
        if (next.powerUps > snake.powerUps) {
          record.claimed = true;
          record.closest = 0;
          const metric = this.metrics[snake.id][this.active.kind];
          metric.claimed += 1;
          metric.claimTicksTotal += Math.max(0, after.tick - this.active.spawnedAt);
        }
        if (!next.alive && record.pursued && !record.pursuitDeathCounted) {
          this.metrics[snake.id][this.active.kind].pursuitDeaths += 1;
          record.pursuitDeathCounted = true;
        }
      }
      if (!after.powerUp || after.powerUp.id !== before.powerUp.id) this.complete();
    }
    if (!before.powerUp && after.powerUp) this.start(after);
  }

  finish(): void { this.complete(); }

  private start(state: GameState): void {
    if (!state.powerUp) return;
    if (this.active) this.complete();
    const field = powerUpDistanceField(state);
    const remaining = Math.max(0, state.powerUp.expiresAt - state.tick);
    const records = new Map<string, PursuitRecord>();
    for (const snake of state.snakes.filter((entry) => entry.alive && this.metrics[entry.id])) {
      const distance = field?.get(pointKey(snake.segments[0]));
      const initial = distance ?? state.width + state.height;
      const metric = this.metrics[snake.id][state.powerUp.kind];
      metric.seen += 1;
      metric.initialDistanceTotal += initial;
      const reachable = distance !== undefined && distance <= remaining;
      if (reachable) metric.reachable += 1;
      records.set(snake.id, { initial, closest: initial, reachable, pursued: false, claimed: false, pursuitDeathCounted: false });
    }
    this.active = { id: state.powerUp.id, kind: state.powerUp.kind, spawnedAt: state.tick, distanceField: field, records };
  }

  private complete(): void {
    if (!this.active) return;
    for (const [snakeId, record] of this.active.records) {
      const metric = this.metrics[snakeId][this.active.kind];
      metric.closestDistanceTotal += record.closest;
      if (record.reachable && !record.pursued) metric.ignored += 1;
      if (record.pursued && !record.claimed) metric.pursuitWithoutClaim += 1;
    }
    this.active = null;
  }
}

type ObjectiveRecord = {
  id: number;
  initial: Map<string, number>;
  closest: Map<string, number>;
  pursued: Set<string>;
  contested: Set<string>;
};

type HazardRecord = { hazard: ArenaHazard; threatened: Set<string> };

const manhattan = (left: { x: number; y: number }, right: { x: number; y: number }) =>
  Math.abs(left.x - right.x) + Math.abs(left.y - right.y);

/**
 * Derives schema-v3 Adaptive Arena diagnostics from state transitions. Keeping
 * this outside the reward function prevents analytics from changing gameplay.
 */
export class AdaptiveBehaviorTracker {
  private objective: ObjectiveRecord | null = null;
  private hazards = new Map<number, HazardRecord>();
  private zoneRisks = new Map<number, Set<string>>();
  private seenZoneWarnings = new Set<number>();

  constructor(private readonly metrics: Record<string, AdaptiveBehaviorStats>) {}

  observe(before: GameState, after: GameState): void {
    this.trackFood(before, after);
    this.trackLeader(before, after);
    this.trackPowerUpCounters(before, after);
    this.trackZone(before, after);
    this.trackHazards(before, after);
    this.trackObjective(before, after);
  }

  finish(state: GameState): void {
    if (this.objective) this.completeObjective(state);
    for (const record of this.hazards.values()) this.completeHazard(record, state);
    this.hazards.clear();
  }

  private trackFood(before: GameState, after: GameState): void {
    const removed = before.food.filter((food) => !after.food.some((next) => next.id === food.id));
    let claims = 0;
    for (const snake of before.snakes) {
      const next = after.snakes.find((entry) => entry.id === snake.id);
      const metric = this.metrics[snake.id];
      if (!next || !metric) continue;
      const foodClaims = Math.max(0, next.foodEaten - snake.foodEaten);
      const rareClaims = Math.max(0, next.rareFoodEaten - snake.rareFoodEaten);
      claims += foodClaims;
      metric.normalFoodClaims += Math.max(0, foodClaims - rareClaims);
      metric.rareFoodClaims += rareClaims;
    }
    const expirations = Math.max(0, removed.length - claims);
    if (expirations) for (const snake of before.snakes.filter((entry) => entry.alive)) this.metrics[snake.id]!.foodExpirationsObserved += expirations;
  }

  private trackLeader(before: GameState, after: GameState): void {
    const leaderId = before.arena.leaderId;
    if (leaderId && this.metrics[leaderId]) {
      this.metrics[leaderId].leaderTicks += 1;
      const wasAlive = before.snakes.find((entry) => entry.id === leaderId)?.alive;
      const isAlive = after.snakes.find((entry) => entry.id === leaderId)?.alive;
      if (wasAlive && !isAlive) this.metrics[leaderId].leaderDeaths += 1;
    }
    for (const snake of before.snakes) {
      const next = after.snakes.find((entry) => entry.id === snake.id);
      const metric = this.metrics[snake.id];
      if (!next || !metric) continue;
      metric.bountyKills += Math.max(0, next.bountyKills - snake.bountyKills);
      if (snake.buffs.crownUntil > before.tick) {
        metric.crownActiveTicks += 1;
        if (snake.alive && !next.alive) metric.crownDeaths += 1;
      }
      if (before.powerUp?.kind === "crown" && !after.powerUp && next.powerUps > snake.powerUps) metric.crownClaims += 1;
    }
  }

  private counter(id: string, kind: PowerUpKind, success: boolean): void {
    const row = this.metrics[id]?.powerUpCounters[kind];
    if (!row) return;
    row.opportunities += 1;
    row.successes += Number(success);
  }

  private trackPowerUpCounters(before: GameState, after: GameState): void {
    for (const snake of before.snakes.filter((entry) => entry.alive)) {
      const next = after.snakes.find((entry) => entry.id === snake.id);
      if (!next) continue;
      if (next.buffs.shield < snake.buffs.shield) this.counter(snake.id, "shield", next.alive);
      if (next.buffs.secondChance < snake.buffs.secondChance) this.counter(snake.id, "secondChance", next.alive);

      if (snake.buffs.phaseUntil > before.tick) {
        const head = next.segments[0];
        const bodyOrObstacle = before.obstacles.some((point) => point.x === head.x && point.y === head.y)
          || before.snakes.some((other) => other.id !== snake.id && other.alive && other.segments.some((point) => point.x === head.x && point.y === head.y));
        const phasedHazard = before.arena.hazards.some((hazard) => hazard.kind !== "laser" && hazardCells(hazard, after.tick).some((point) => point.x === head.x && point.y === head.y));
        if (bodyOrObstacle || phasedHazard) this.counter(snake.id, "phase", next.alive);
      }
      if (snake.buffs.hasteUntil > before.tick) this.counter(snake.id, "haste", next.alive);

      const ate = next.foodEaten > snake.foodEaten;
      if (ate && snake.buffs.doubleUntil > before.tick) this.counter(snake.id, "double", next.alive);
      if (ate && snake.buffs.magnetUntil > before.tick) {
        const remotelyCollected = before.food.some((food) => manhattan(food.position, next.segments[0]) > 0 && manhattan(food.position, next.segments[0]) <= 3);
        this.counter(snake.id, "magnet", remotelyCollected);
      }
      if (snake.buffs.crownUntil > before.tick && after.tick % 10 === 0) this.counter(snake.id, "crown", next.alive && next.score > snake.score);
      if (snake.buffs.visionUntil > before.tick) {
        const threatened = ([0, 1, 2] as const).some((action) => isDangerAt(before, snake, relativeDirection(snake.direction, action), 2));
        if (threatened) this.counter(snake.id, "vision", next.alive);
      }

      if (before.powerUp && !after.powerUp && next.powerUps > snake.powerUps) {
        const kind = before.powerUp.kind;
        if (kind === "growth") this.counter(snake.id, kind, next.segments.length >= snake.segments.length + 4);
        if (kind === "trim") this.counter(snake.id, kind, next.segments.length < snake.segments.length);
        if (kind === "warp") this.counter(snake.id, kind, insideSafeBounds(next.segments[0], after.width, after.height, after.arena.safeZone.pendingInset));
        if (kind === "freeze") {
          const affected = before.snakes.some((rival) => {
            if (rival.id === snake.id) return false;
            const changed = after.snakes.find((entry) => entry.id === rival.id);
            return Boolean(changed && (changed.buffs.frozenUntil > rival.buffs.frozenUntil || changed.buffs.shield < rival.buffs.shield));
          });
          this.counter(snake.id, kind, affected);
        }
      }
    }
  }

  private trackZone(before: GameState, after: GameState): void {
    const zone = after.arena.safeZone;
    if (after.tick >= zone.telegraphAt && !this.seenZoneWarnings.has(zone.closesAt)) {
      this.seenZoneWarnings.add(zone.closesAt);
      const risks = new Set<string>();
      for (const snake of after.snakes.filter((entry) => entry.alive)) {
        if (!insideSafeBounds(snake.segments[0], after.width, after.height, zone.pendingInset)) {
          risks.add(snake.id);
          this.metrics[snake.id]!.zoneWarnings += 1;
        }
      }
      this.zoneRisks.set(zone.closesAt, risks);
    }
    if (after.arena.safeZone.inset > before.arena.safeZone.inset) {
      const risks = this.zoneRisks.get(before.arena.safeZone.closesAt) ?? new Set<string>();
      for (const id of risks) {
        const snake = after.snakes.find((entry) => entry.id === id);
        if (snake?.alive && insideSafeBounds(snake.segments[0], after.width, after.height, after.arena.safeZone.inset)) this.metrics[id]!.zoneRepositions += 1;
        else if (snake && !snake.alive && snake.deathReason?.includes("zone")) this.metrics[id]!.zoneDeaths += 1;
      }
      this.zoneRisks.delete(before.arena.safeZone.closesAt);
    }
  }

  private threatSet(state: GameState, hazard: ArenaHazard): Set<string> {
    const cells = hazardCells(hazard, hazard.activatesAt);
    return new Set(state.snakes.filter((snake) => snake.alive && cells.some((cell) => manhattan(cell, snake.segments[0]) <= 3)).map((snake) => snake.id));
  }

  private trackHazards(before: GameState, after: GameState): void {
    for (const hazard of after.arena.hazards) {
      if (!this.hazards.has(hazard.id)) {
        const threatened = this.threatSet(after, hazard);
        this.hazards.set(hazard.id, { hazard, threatened });
        for (const id of threatened) this.metrics[id]!.hazardEncounters += 1;
      }
    }
    for (const [id, record] of this.hazards) {
      for (const snake of before.snakes.filter((entry) => record.threatened.has(entry.id) && entry.alive)) {
        const next = after.snakes.find((entry) => entry.id === snake.id);
        if (!next) continue;
        if (next.buffs.shield < snake.buffs.shield && next.alive) this.metrics[snake.id]!.hazardShieldBlocks += 1;
        if (!next.alive && next.deathReason?.includes("hazard")) this.metrics[snake.id]!.hazardDeaths += 1;
      }
      if (!after.arena.hazards.some((hazard) => hazard.id === id)) {
        this.completeHazard(record, after);
        this.hazards.delete(id);
      }
    }
  }

  private completeHazard(record: HazardRecord, state: GameState): void {
    for (const id of record.threatened) if (state.snakes.find((snake) => snake.id === id)?.alive) this.metrics[id]!.hazardEvasions += 1;
  }

  private startObjective(state: GameState): void {
    if (!state.arena.objective) return;
    const initial = new Map<string, number>();
    for (const snake of state.snakes.filter((entry) => entry.alive)) {
      initial.set(snake.id, manhattan(snake.segments[0], state.arena.objective.position));
      this.metrics[snake.id]!.objectiveOpportunities += 1;
    }
    this.objective = { id: state.arena.objective.id, initial, closest: new Map(initial), pursued: new Set(), contested: new Set() };
  }

  private trackObjective(before: GameState, after: GameState): void {
    if (before.arena.objective && (!this.objective || this.objective.id !== before.arena.objective.id)) this.startObjective(before);
    if (this.objective && before.arena.objective) {
      for (const snake of after.snakes) {
        const initial = this.objective.initial.get(snake.id);
        if (initial === undefined) continue;
        const closest = Math.min(this.objective.closest.get(snake.id) ?? initial, manhattan(snake.segments[0], before.arena.objective.position));
        this.objective.closest.set(snake.id, closest);
        const threshold = Math.min(3, Math.max(1, Math.ceil(initial * .2)));
        if (!this.objective.pursued.has(snake.id) && initial - closest >= threshold) {
          this.objective.pursued.add(snake.id);
          this.metrics[snake.id]!.objectivePursuits += 1;
        }
        if (before.arena.objective.contested && manhattan(snake.segments[0], before.arena.objective.position) <= before.arena.objective.radius && !this.objective.contested.has(snake.id)) {
          this.objective.contested.add(snake.id);
          this.metrics[snake.id]!.objectiveContests += 1;
        }
        const previous = before.snakes.find((entry) => entry.id === snake.id);
        if (previous?.alive && !snake.alive && this.objective.pursued.has(snake.id)) this.metrics[snake.id]!.objectivePursuitDeaths += 1;
        if ((snake.objectiveCaptures - (previous?.objectiveCaptures ?? 0)) > 0) this.metrics[snake.id]!.objectiveCaptures += 1;
      }
      if (!after.arena.objective || after.arena.objective.id !== this.objective.id) this.completeObjective(after);
    }
    if (after.arena.objective && (!this.objective || this.objective.id !== after.arena.objective.id)) this.startObjective(after);
  }

  private completeObjective(_state: GameState): void { this.objective = null; }
}

export function finalizePowerUpBehavior(raw: Record<PowerUpKind, PowerUpBehaviorRaw>): Record<PowerUpKind, PowerUpBehaviorStats> {
  return Object.fromEntries(POWER_UP_KINDS.map((kind) => {
    const value = raw[kind];
    return [kind, {
      seen: value.seen, reachable: value.reachable, pursued: value.pursued, claimed: value.claimed,
      ignored: value.ignored, pursuitWithoutClaim: value.pursuitWithoutClaim, pursuitDeaths: value.pursuitDeaths,
      avgInitialDistance: value.seen ? value.initialDistanceTotal / value.seen : null,
      avgClosestDistance: value.seen ? value.closestDistanceTotal / value.seen : null,
      avgClaimTicks: value.claimed ? value.claimTicksTotal / value.claimed : null,
    }];
  })) as Record<PowerUpKind, PowerUpBehaviorStats>;
}
