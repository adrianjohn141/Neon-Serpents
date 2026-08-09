import { castRays, isDangerAt, relativeDirection } from "./engine";
import { boundsForInset, defaultOpponentProfile, hazardCells, insideSafeBounds } from "./adaptive-arena";
import { ADAPTIVE_PHASE_ENDS, MAP_ARCHETYPES, OBJECTIVE_CAPTURE_TICKS, POWER_UP_KINDS } from "./constants";
import type { DeathCause, Direction, GameState, OpponentTarget, Point, RelativeAction, Snake } from "./types";

export const ACTION_COUNT = 3;
export const RAY_COUNT = 16;
export const ENTITY_INDEX = { wall: 0, obstacle: 1, self: 2, enemy: 3, food: 4, powerUp: 5 } as const;
export const ENTITY_COUNT = Object.keys(ENTITY_INDEX).length;

// Layout of the 228 normalized AI inputs, kept as named slices for tests and
// future maintenance. Keep OBSERVATION_LAYOUT lengths in sync with the encoder.
export const OBSERVATION_LAYOUT = {
  rays: { start: 0, length: RAY_COUNT * (1 + ENTITY_COUNT) }, // 112
  food: { start: 112, length: 2 },
  powerUp: { start: 114, length: 17 }, // present 1 + kind 12 + expiry 1 + vector 2 + next-spawn 1
  enemies: { start: 131, length: 14 }, // vectors/dist/count 6 + two relative heading one-hots 8
  self: { start: 145, length: 2 }, // length 1 + rank 1
  buffs: { start: 147, length: 9 },
  danger: { start: 156, length: ACTION_COUNT },
  phase: { start: 159, length: 4 },
  safeZone: { start: 163, length: 7 },
  objective: { start: 170, length: 6 },
  hazard: { start: 176, length: 9 },
  map: { start: 185, length: 6 },
  bounty: { start: 191, length: 4 },
  rareFood: { start: 195, length: 4 },
  opponentProfiles: { start: 199, length: 26 },
  seriesRound: { start: 225, length: 3 },
} as const;

export const OBSERVATION_SIZE =
  OBSERVATION_LAYOUT.seriesRound.start + OBSERVATION_LAYOUT.seriesRound.length; // 228

function nearest(head: Point, points: Point[]): Point | null {
  let result: Point | null = null;
  let best = Infinity;
  for (const point of points) {
    const candidate = Math.abs(head.x - point.x) + Math.abs(head.y - point.y);
    if (candidate < best) { best = candidate; result = point; }
  }
  return result;
}

function relativeVector(head: Point, target: Point | null, facing: Direction, width: number, height: number): [number, number] {
  if (!target) return [0, 0];
  const dx = (target.x - head.x) / Math.max(1, width - 1);
  const dy = (target.y - head.y) / Math.max(1, height - 1);
  if (facing === "up") return [dx, -dy];
  if (facing === "right") return [dy, dx];
  if (facing === "down") return [-dx, dy];
  return [-dy, -dx];
}

function sortedEnemyHeads(state: GameState, snake: Snake): Point[] {
  const head = snake.segments[0];
  return state.snakes
    .filter((entry) => entry.id !== snake.id && entry.alive)
    .map((entry) => entry.segments[0])
    .sort((a, b) =>
      (Math.abs(head.x - a.x) + Math.abs(head.y - a.y)) -
      (Math.abs(head.x - b.x) + Math.abs(head.y - b.y)),
    );
}

function sortedEnemies(state: GameState, snake: Snake): Snake[] {
  const head = snake.segments[0];
  return state.snakes
    .filter((entry) => entry.id !== snake.id && entry.alive)
    .sort((a, b) => distance(head, a.segments[0]) - distance(head, b.segments[0]));
}

const distance = (a: Point, b: Point) => Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
const clamp = (value: number) => Math.max(0, Math.min(1, value));

function distanceToHazardSafety(state: GameState, head: Point, projected: Point[]): number {
  const danger = new Set(projected.map((point) => `${point.x},${point.y}`));
  if (!danger.has(`${head.x},${head.y}`)) return 0;
  const blocked = new Set(state.obstacles.map((point) => `${point.x},${point.y}`));
  const queue: Array<{ point: Point; distance: number }> = [{ point: head, distance: 0 }];
  const seen = new Set([`${head.x},${head.y}`]);
  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const current = queue[cursor];
    if (!danger.has(`${current.point.x},${current.point.y}`)) return current.distance;
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const point = { x: current.point.x + dx, y: current.point.y + dy };
      const key = `${point.x},${point.y}`;
      if (point.x < 0 || point.y < 0 || point.x >= state.width || point.y >= state.height || blocked.has(key) || seen.has(key)) continue;
      seen.add(key); queue.push({ point, distance: current.distance + 1 });
    }
  }
  return state.width + state.height;
}

function dangerFlags(state: GameState, snake: Snake): [0 | 1, 0 | 1, 0 | 1] {
  const lookahead = snake.buffs.visionUntil > state.tick || snake.buffs.hasteUntil > state.tick ? 2 : 1;
  return ([0, 1, 2] as RelativeAction[]).map((action) =>
    isDangerAt(state, snake, relativeDirection(snake.direction, action), lookahead) ? 1 as const : 0 as const,
  ) as [0 | 1, 0 | 1, 0 | 1];
}

export function encodeObservation(state: GameState, snake: Snake): Float32Array {
  const values: number[] = [];

  for (const ray of castRays(state, snake)) {
    values.push(ray.distance);
    for (let index = 0; index < ENTITY_COUNT; index += 1) values.push(index === ENTITY_INDEX[ray.entity] ? 1 : 0);
  }

  const head = snake.segments[0];
  values.push(...relativeVector(head, nearest(head, state.food.map((food) => food.position)), snake.direction, state.width, state.height));

  // Power-up awareness: presence, kind one-hot, expiry pressure, relative vector.
  const powerUp = state.powerUp;
  values.push(powerUp ? 1 : 0);
  if (powerUp) {
    const kindIndex = POWER_UP_KINDS.indexOf(powerUp.kind);
    for (let index = 0; index < POWER_UP_KINDS.length; index += 1) values.push(index === kindIndex ? 1 : 0);
    values.push(Math.max(0, Math.min(1, (powerUp.expiresAt - state.tick) / 180)));
  } else {
    for (let index = 0; index < POWER_UP_KINDS.length; index += 1) values.push(0);
    values.push(0);
  }
  values.push(...relativeVector(head, powerUp?.position ?? null, snake.direction, state.width, state.height));
  values.push(powerUp ? 0 : Math.max(0, Math.min(1, (state.nextPowerUpAt - state.tick) / 180)));

  // Enemy awareness: two nearest enemy heads (relative), nearest distance, count.
  const enemies = sortedEnemyHeads(state, snake);
  const first = enemies[0] ?? null;
  const second = enemies[1] ?? null;
  values.push(...relativeVector(head, first, snake.direction, state.width, state.height));
  values.push(...relativeVector(head, second, snake.direction, state.width, state.height));
  values.push(first ? (Math.abs(head.x - first.x) + Math.abs(head.y - first.y)) / Math.max(1, state.width + state.height) : 0);
  values.push(Math.max(0, state.snakes.filter((entry) => entry.id !== snake.id && entry.alive).length) / 7);
  const relativeHeading = (enemy: Point | null): number[] => {
    const target = enemy && state.snakes.find((entry) => entry.alive && entry.id !== snake.id && entry.segments[0].x === enemy.x && entry.segments[0].y === enemy.y);
    if (!target) return [0, 0, 0, 0];
    const directions: Direction[] = ["up", "right", "down", "left"];
    const facingIndex = directions.indexOf(snake.direction);
    const enemyIndex = directions.indexOf(target.direction);
    const delta = (enemyIndex - facingIndex + 4) % 4;
    return directions.map((_, index) => index === delta ? 1 : 0);
  };
  values.push(...relativeHeading(first), ...relativeHeading(second));

  values.push(Math.min(1, snake.segments.length / 50));

  const alive = state.snakes.filter((entry) => entry.alive).sort((a, b) => b.score - a.score);
  const rank = Math.max(0, alive.findIndex((entry) => entry.id === snake.id));
  values.push(alive.length <= 1 ? 1 : 1 - rank / (alive.length - 1));

  const remaining = (until: number, duration: number) => Math.max(0, Math.min(1, (until - state.tick) / duration));
  values.push(
    Math.min(1, snake.buffs.shield / 3),
    remaining(snake.buffs.phaseUntil, 90),
    remaining(snake.buffs.hasteUntil, 55),
    remaining(snake.buffs.doubleUntil, 120),
    remaining(snake.buffs.magnetUntil, 120),
    Math.min(1, snake.buffs.secondChance / 3),
    remaining(snake.buffs.frozenUntil, 18),
    remaining(snake.buffs.crownUntil, 150),
    remaining(snake.buffs.visionUntil, 160),
  );

  values.push(...dangerFlags(state, snake));

  const phases = ["opening", "midgame", "endgame"] as const;
  for (const phase of phases) values.push(state.arena.phase === phase ? 1 : 0);
  const phaseStart = state.arena.phase === "opening" ? 0 : state.arena.phase === "midgame" ? ADAPTIVE_PHASE_ENDS.opening : ADAPTIVE_PHASE_ENDS.midgame;
  values.push(clamp((state.arena.phaseEndsAt - state.tick) / Math.max(1, state.arena.phaseEndsAt - phaseStart)));

  const safe = boundsForInset(state.width, state.height, state.arena.safeZone.inset);
  values.push(
    clamp((head.x - safe.minX) / Math.max(1, state.width - 1)),
    clamp((safe.maxX - head.x) / Math.max(1, state.width - 1)),
    clamp((head.y - safe.minY) / Math.max(1, state.height - 1)),
    clamp((safe.maxY - head.y) / Math.max(1, state.height - 1)),
    clamp((state.arena.safeZone.closesAt - state.tick) / 90),
    clamp(state.arena.safeZone.pendingInset / Math.max(1, Math.min(state.width, state.height) / 2)),
    insideSafeBounds(head, state.width, state.height, state.arena.safeZone.inset) ? 0 : 1,
  );

  const objective = state.arena.objective;
  values.push(objective ? 1 : 0);
  values.push(...relativeVector(head, objective?.position ?? null, snake.direction, state.width, state.height));
  values.push(objective ? clamp((objective.expiresAt - state.tick) / 180) : 0);
  values.push(objective ? clamp(state.snakes.filter((entry) => entry.alive && distance(entry.segments[0], objective.position) <= objective.radius).length / 7) : 0);
  values.push(objective ? clamp((objective.progress[snake.id] ?? 0) / OBJECTIVE_CAPTURE_TICKS) : 0);

  const nearestHazard = state.arena.hazards
    .filter((hazard) => hazard.activeUntil >= state.tick)
    .sort((a, b) => distance(head, a.origin) - distance(head, b.origin))[0] ?? null;
  values.push(nearestHazard ? 1 : 0);
  for (const kind of ["laser", "sweeper", "blocks"] as const) values.push(nearestHazard?.kind === kind ? 1 : 0);
  values.push(...relativeVector(head, nearestHazard?.origin ?? null, snake.direction, state.width, state.height));
  values.push(nearestHazard ? clamp((nearestHazard.activatesAt - state.tick) / 180) : 0);
  values.push(nearestHazard && nearestHazard.activatesAt <= state.tick ? 1 : 0);
  const projected = nearestHazard ? hazardCells(nearestHazard, nearestHazard.activatesAt) : [];
  values.push(clamp(distanceToHazardSafety(state, head, projected) / Math.max(state.width, state.height)));

  for (const archetype of MAP_ARCHETYPES) values.push(state.arena.mapArchetype === archetype ? 1 : 0);

  const leader = state.arena.leaderId ? state.snakes.find((entry) => entry.id === state.arena.leaderId && entry.alive) ?? null : null;
  values.push(state.arena.leaderId === snake.id ? 1 : 0);
  values.push(...relativeVector(head, leader?.segments[0] ?? null, snake.direction, state.width, state.height));
  values.push(leader ? clamp((leader.score - snake.score + 20) / 40) : 0);

  const rareFood = nearest(head, state.food.filter((food) => food.kind === "rare").map((food) => food.position));
  const rareEntry = rareFood && state.food.find((food) => food.position.x === rareFood.x && food.position.y === rareFood.y);
  values.push(rareFood ? 1 : 0, ...relativeVector(head, rareFood, snake.direction, state.width, state.height), rareEntry ? clamp((rareEntry.expiresAt - state.tick) / 300) : 0);

  const targets: OpponentTarget[] = ["survival", "food", "powerup", "objective", "leader"];
  const deaths: DeathCause[] = ["wall", "obstacle", "headOn", "snakeBody", "other"];
  const enemyRows = sortedEnemies(state, snake);
  for (let index = 0; index < 2; index += 1) {
    const profile = enemyRows[index] ? state.arena.opponentProfiles[enemyRows[index].id] ?? defaultOpponentProfile() : defaultOpponentProfile();
    values.push(Math.max(-1, Math.min(1, profile.turnBias)), clamp(profile.aggressionRate));
    for (const target of targets) values.push(profile.samples && profile.commonTarget === target ? 1 : 0);
    values.push(clamp(profile.powerUpRate));
    for (const cause of deaths) values.push(profile.samples && profile.typicalDeathCause === cause ? 1 : 0);
  }

  for (const round of [1, 2, 3] as const) values.push(state.arena.seriesRound === round ? 1 : 0);

  if (values.length !== OBSERVATION_SIZE) throw new Error(`Expected ${OBSERVATION_SIZE} AI inputs, received ${values.length}.`);
  return Float32Array.from(values);
}
