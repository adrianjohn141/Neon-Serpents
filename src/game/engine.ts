import {
  ADAPTIVE_PHASE_ENDS, BOARD_HEIGHT, BOARD_WIDTH, DIRECTION_VECTOR, DIRECTIONS, FOOD_COUNT,
  MAP_ARCHETYPES, OBJECTIVE_CAPTURE_TICKS, OBJECTIVE_COOLDOWN_TICKS, OBJECTIVE_LIFETIME_TICKS,
  PHASE_FOOD_COUNT, PHASE_FOOD_LIFETIME, POWER_UP_KINDS, POWER_UP_META, SNAKES,
} from "./constants";
import {
  advanceSafeZone, arenaPhaseAt, boundsForInset, createArenaState, createHazard, hazardCells,
  insideSafeBounds, makeMapObstacles, midgameObstacleCells, projectedHazardCells,
} from "./adaptive-arena";
import {
  deathReward, foodDistanceDeltaReward, foodEatReward, killReward,
  powerUpApproachReward, powerUpClaimReward, stepPenalty, survivalReward, winReward,
} from "./rewards";
import { randomInt, seededFromNow } from "./rng";
import type {
  ArenaFood, Buffs, Direction, GameConfig, GameEvent, GameState, Point, RayEntity, RayHit,
  RelativeAction, RewardBreakdown, Snake, SnakeDefinition, StepResult,
} from "./types";

export const pointKey = (point: Point) => `${point.x},${point.y}`;
const samePoint = (a: Point, b: Point) => a.x === b.x && a.y === b.y;
const distance = (a: Point, b: Point) => Math.abs(a.x - b.x) + Math.abs(a.y - b.y);

function emptyBuffs(): Buffs {
  return {
    shield: 0, phaseUntil: 0, hasteUntil: 0, doubleUntil: 0,
    magnetUntil: 0, secondChance: 0, frozenUntil: 0, crownUntil: 0,
    visionUntil: 0,
  };
}

function turn(direction: Direction, action: RelativeAction): Direction {
  const current = DIRECTIONS.indexOf(direction);
  const offset = action === 1 ? -1 : action === 2 ? 1 : 0;
  return DIRECTIONS[(current + offset + 4) % 4];
}

export function movePoint(point: Point, direction: Direction): Point {
  const vector = DIRECTION_VECTOR[direction];
  return { x: point.x + vector.x, y: point.y + vector.y };
}

function makeObstacles(width: number, height: number): Point[] {
  const cells: Point[] = [];
  const addLine = (x: number, y: number, dx: number, dy: number, length: number) => {
    for (let i = 0; i < length; i += 1) cells.push({ x: x + dx * i, y: y + dy * i });
  };
  addLine(Math.floor(width * 0.27), Math.floor(height * 0.23), 1, 0, 9);
  addLine(Math.floor(width * 0.61), Math.floor(height * 0.23), 1, 0, 9);
  addLine(Math.floor(width * 0.27), Math.floor(height * 0.75), 1, 0, 9);
  addLine(Math.floor(width * 0.61), Math.floor(height * 0.75), 1, 0, 9);
  addLine(Math.floor(width * 0.22), Math.floor(height * 0.39), 0, 1, 9);
  addLine(Math.floor(width * 0.77), Math.floor(height * 0.39), 0, 1, 9);
  const cx = Math.floor(width / 2);
  const cy = Math.floor(height / 2);
  addLine(cx - 4, cy - 3, 1, 0, 3);
  addLine(cx + 2, cy - 3, 1, 0, 3);
  addLine(cx - 4, cy + 3, 1, 0, 3);
  addLine(cx + 2, cy + 3, 1, 0, 3);
  return cells.filter((point) => point.x > 2 && point.y > 2 && point.x < width - 3 && point.y < height - 3);
}

type SpawnSlot = { head: Point; direction: Direction };

function perimeterSpawnSlots(width: number, height: number): SpawnSlot[] {
  const top = 5;
  const bottom = height - 6;
  const left = 5;
  const right = width - 6;
  const quarterX = Math.floor(width * .25);
  const halfX = Math.floor(width * .5);
  const threeQuarterX = Math.floor(width * .75);
  const quarterY = Math.floor(height * .25);
  const halfY = Math.floor(height * .5);
  const threeQuarterY = Math.floor(height * .75);
  return [
    { head: { x: halfX, y: top }, direction: "down" },
    { head: { x: threeQuarterX, y: top }, direction: "down" },
    { head: { x: right, y: halfY }, direction: "left" },
    { head: { x: right, y: threeQuarterY }, direction: "left" },
    { head: { x: halfX, y: bottom }, direction: "up" },
    { head: { x: quarterX, y: bottom }, direction: "up" },
    { head: { x: left, y: halfY }, direction: "right" },
    { head: { x: left, y: quarterY }, direction: "right" },
  ];
}

function initialSnake(definition: SnakeDefinition, index: number, count: number, width: number, height: number): Snake {
  const slots = perimeterSpawnSlots(width, height);
  const slotIndex = Math.floor(index * slots.length / count);
  const start = slots[slotIndex];
  const reverse = DIRECTIONS[(DIRECTIONS.indexOf(start.direction) + 2) % 4];
  const segments = [start.head];
  for (let i = 1; i < 5; i += 1) segments.push(movePoint(segments[i - 1], reverse));
  return { ...definition, segments, direction: start.direction, alive: true, score: 0, foodEaten: 0, powerUps: 0, rareFoodEaten: 0, objectiveCaptures: 0, bountyKills: 0, kills: 0, buffs: emptyBuffs() };
}

function occupiedCells(state: Pick<GameState, "snakes" | "obstacles" | "food" | "powerUp">): Set<string> {
  const occupied = new Set(state.obstacles.map(pointKey));
  state.snakes.forEach((snake) => snake.segments.forEach((point) => occupied.add(pointKey(point))));
  state.food.forEach((food) => occupied.add(pointKey(food.position)));
  if (state.powerUp) occupied.add(pointKey(state.powerUp.position));
  return occupied;
}

function randomOpenCell(state: GameState, margin = 2, pendingSafeArea = false): [Point, number] {
  let seed = state.seed;
  const occupied = occupiedCells(state);
  const inset = state.arena?.enabled ? (pendingSafeArea ? state.arena.safeZone.pendingInset : state.arena.safeZone.inset) : margin;
  const effectiveMargin = Math.max(margin, inset + 1);
  for (let attempt = 0; attempt < 600; attempt += 1) {
    let x: number;
    let y: number;
    [x, seed] = randomInt(seed, Math.max(1, state.width - effectiveMargin * 2));
    [y, seed] = randomInt(seed, Math.max(1, state.height - effectiveMargin * 2));
    const point = { x: x + effectiveMargin, y: y + effectiveMargin };
    if (!occupied.has(pointKey(point)) && !state.arena?.hazards.some((hazard) => projectedHazardCells(hazard, state.tick, 2).some((cell) => samePoint(cell, point)))) return [point, seed];
  }
  return [{ x: Math.floor(state.width / 2), y: Math.floor(state.height / 2) }, seed];
}

function replenishFood(state: GameState): GameState {
  const targetCount = state.arena.enabled ? PHASE_FOOD_COUNT[state.arena.phase] : state.mode === "battle" ? FOOD_COUNT : 1;
  const lifetime = state.arena.enabled ? PHASE_FOOD_LIFETIME[state.arena.phase] : 1_000_000;
  const liveFood = state.food
    .filter((food) => food.expiresAt > state.tick && (!state.arena.enabled || insideSafeBounds(food.position, state.width, state.height, state.arena.safeZone.pendingInset)))
    .slice(0, targetCount)
    .map((food) => ({ ...food, expiresAt: Math.min(food.expiresAt, state.tick + lifetime) }));
  let next = { ...state, food: liveFood };
  while (next.food.length < targetCount) {
    const [cell, seed] = randomOpenCell(next, 2, true);
    let rareRoll: number;
    [rareRoll, next.seed] = randomInt(seed, 100);
    const rare = next.arena.enabled && rareRoll < 12;
    const food: ArenaFood = { id: state.tick * 100 + next.food.length, position: cell, kind: rare ? "rare" : "normal", value: rare ? 3 : 1, spawnedAt: state.tick, expiresAt: state.tick + lifetime };
    next = { ...next, food: [...next.food, food] };
  }
  return next;
}

export function createGame(config: GameConfig = {}): GameState {
  const width = config.width ?? BOARD_WIDTH;
  const height = config.height ?? BOARD_HEIGHT;
  const mode = config.mode ?? "battle";
  const initialSeed = config.seed ?? seededFromNow();
  const adaptiveArena = config.adaptiveArena ?? (mode === "battle" || mode === "safezone" || mode === "hazard" || mode === "objective");
  const mapArchetype = config.mapArchetype ?? MAP_ARCHETYPES[initialSeed % MAP_ARCHETYPES.length];
  const definitions = config.snakes ?? SNAKES;
  if (mode === "battle" && (definitions.length < 2 || definitions.length > 8)) throw new Error("Battle mode requires between 2 and 8 snakes.");
  if (mode !== "battle" && definitions.length !== 1) throw new Error("Training modes require exactly one snake.");
  if (new Set(definitions.map((snake) => snake.id)).size !== definitions.length) throw new Error("Snake identifiers must be unique.");
  const snakes = definitions.map((definition, index) => initialSnake(definition, index, definitions.length, width, height));
  const spawnCells = new Set(snakes.flatMap((snake) => snake.segments.map(pointKey)));
  const arena = createArenaState(width, height, mapArchetype, adaptiveArena, config.seriesRound ?? 1, config.opponentProfiles ?? {});
  if (mode === "safezone") {
    arena.phase = "endgame"; arena.phaseEndsAt = ADAPTIVE_PHASE_ENDS.endgame;
    arena.safeZone.telegraphAt = 5; arena.safeZone.closesAt = 50;
  } else if (mode === "hazard") {
    arena.phase = "midgame"; arena.phaseEndsAt = ADAPTIVE_PHASE_ENDS.midgame; arena.nextHazardAt = 5;
  } else if (mode === "objective") {
    arena.phase = "midgame"; arena.phaseEndsAt = ADAPTIVE_PHASE_ENDS.midgame; arena.nextObjectiveAt = 5;
  }
  let state: GameState = {
    width, height, seed: initialSeed, tick: 0,
    mode, status: "running", winnerId: null,
    snakes, food: [], obstacles: (adaptiveArena ? makeMapObstacles(mapArchetype, width, height) : makeObstacles(width, height))
      .filter((point) => !spawnCells.has(pointKey(point))), powerUp: null,
    powerUpSpawnCount: 0, nextPowerUpAt: mode === "powerup" ? 5 : 45, arena, events: [],
  };
  state = replenishFood(state);
  return state;
}

function addEvent(events: GameEvent[], tick: number, text: string, tone: GameEvent["tone"]): GameEvent[] {
  return [{ id: tick * 100 + events.length, tick, text, tone }, ...events].slice(0, 6);
}

function safestOpenCell(state: GameState, snakeId: string): Point {
  const occupied = occupiedCells({ ...state, food: [], powerUp: null });
  const enemies = state.snakes.filter((snake) => snake.id !== snakeId && snake.alive).map((snake) => snake.segments[0]);
  let best = { x: Math.floor(state.width / 2), y: Math.floor(state.height / 2) };
  let bestScore = -Infinity;
  for (let y = 2; y < state.height - 2; y += 2) {
    for (let x = 2; x < state.width - 2; x += 2) {
      const point = { x, y };
      if (occupied.has(pointKey(point)) || (state.arena.enabled && !insideSafeBounds(point, state.width, state.height, state.arena.safeZone.pendingInset))) continue;
      if (state.arena.hazards.some((hazard) => projectedHazardCells(hazard, state.tick, 2).some((cell) => samePoint(cell, point)))) continue;
      const wallDistance = Math.min(x, y, state.width - 1 - x, state.height - 1 - y);
      const enemyDistance = enemies.length ? Math.min(...enemies.map((head) => distance(point, head))) : 20;
      const score = wallDistance * 2 + enemyDistance;
      if (score > bestScore) { best = point; bestScore = score; }
    }
  }
  return best;
}

function applyPowerUp(state: GameState, snake: Snake): { state: GameState; snake: Snake } {
  const powerUp = state.powerUp;
  if (!powerUp) return { state, snake };
  let nextSnake = { ...snake, buffs: { ...snake.buffs }, powerUps: snake.powerUps + 1, score: snake.score + 3 };
  let snakes = state.snakes;
  switch (powerUp.kind) {
    case "shield": nextSnake.buffs.shield += 1; break;
    case "phase": nextSnake.buffs.phaseUntil = state.tick + 90; break;
    case "haste": nextSnake.buffs.hasteUntil = state.tick + 55; break;
    case "double": nextSnake.buffs.doubleUntil = state.tick + 120; break;
    case "magnet": nextSnake.buffs.magnetUntil = state.tick + 120; break;
    case "growth": {
      const tail = nextSnake.segments[nextSnake.segments.length - 1];
      nextSnake.segments = [...nextSnake.segments, tail, tail, tail, tail];
      nextSnake.score += 6;
      break;
    }
    case "trim": nextSnake.segments = nextSnake.segments.slice(0, Math.max(3, Math.ceil(nextSnake.segments.length * 0.55))); break;
    case "secondChance": nextSnake.buffs.secondChance += 1; break;
    case "warp": nextSnake.segments = [safestOpenCell(state, snake.id)]; break;
    case "freeze": snakes = state.snakes.map((rival) => {
      if (rival.id === snake.id) return rival;
      if (rival.buffs.shield > 0) return { ...rival, buffs: { ...rival.buffs, shield: rival.buffs.shield - 1 } };
      return { ...rival, buffs: { ...rival.buffs, frozenUntil: state.tick + 18 } };
    }); break;
    case "crown": nextSnake.buffs.crownUntil = state.tick + 150; break;
    case "vision": nextSnake.buffs.visionUntil = state.tick + 160; break;
  }
  const meta = POWER_UP_META[powerUp.kind];
  return {
    snake: nextSnake,
    state: {
      ...state, snakes, powerUp: null, nextPowerUpAt: state.tick + (state.mode === "powerup" ? 5 : 55),
      events: addEvent(state.events, state.tick, `${snake.name} claimed ${meta.name}`, "power"),
    },
  };
}

function resolveFatalCollision(state: GameState, snake: Snake, reason: string, shieldable = true): Snake {
  if (shieldable && snake.buffs.shield > 0) {
    return { ...snake, buffs: { ...snake.buffs, shield: snake.buffs.shield - 1 } };
  }
  if (snake.buffs.secondChance > 0) {
    return {
      ...snake, segments: [safestOpenCell(state, snake.id)],
      buffs: { ...snake.buffs, secondChance: snake.buffs.secondChance - 1 },
    };
  }
  return { ...snake, alive: false, deathReason: reason };
}

function isWall(state: GameState, point: Point): boolean {
  return point.x < 0 || point.y < 0 || point.x >= state.width || point.y >= state.height;
}

function distanceToInset(point: Point, width: number, height: number, inset: number): number {
  const bounds = boundsForInset(width, height, inset);
  return Math.max(0, bounds.minX - point.x) + Math.max(0, point.x - bounds.maxX) + Math.max(0, bounds.minY - point.y) + Math.max(0, point.y - bounds.maxY);
}

function potentialShaping(beforeDistance: number, afterDistance: number, maxDistance: number, strength: number): number {
  const before = 1 - Math.min(1, beforeDistance / Math.max(1, maxDistance));
  const after = 1 - Math.min(1, afterDistance / Math.max(1, maxDistance));
  return strength * (.95 * after - before);
}

const emptyRewardBreakdown = (): RewardBreakdown => ({
  step: 0, survival: 0, foodApproach: 0, foodClaim: 0, powerUpApproach: 0,
  powerUpClaim: 0, zonePositioning: 0, objectiveApproach: 0, objectiveCapture: 0,
  bountyKill: 0, rareFoodClaim: 0, kill: 0, death: 0, win: 0,
});

function addReward(
  rewards: Record<string, number>,
  breakdowns: Record<string, RewardBreakdown>,
  snakeId: string,
  component: keyof RewardBreakdown,
  value: number,
): void {
  rewards[snakeId] = (rewards[snakeId] ?? 0) + value;
  const breakdown = breakdowns[snakeId] ?? emptyRewardBreakdown();
  breakdown[component] += value;
  breakdowns[snakeId] = breakdown;
}

/** Static-obstacle distance field shared by every snake for this substep. */
export function powerUpDistanceField(state: GameState): Map<string, number> | null {
  if (!state.powerUp) return null;
  const blocked = new Set(state.obstacles.map(pointKey));
  const field = new Map<string, number>([[pointKey(state.powerUp.position), 0]]);
  const queue: Point[] = [state.powerUp.position];
  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const point = queue[cursor];
    const nextDistance = field.get(pointKey(point))! + 1;
    for (const direction of DIRECTIONS) {
      const next = movePoint(point, direction);
      const key = pointKey(next);
      if (isWall(state, next) || (state.arena.enabled && !insideSafeBounds(next, state.width, state.height, state.arena.safeZone.pendingInset)) || blocked.has(key) || field.has(key)) continue;
      field.set(key, nextDistance);
      queue.push(next);
    }
  }
  return field;
}

function executeSubstep(state: GameState, actions: Record<string, RelativeAction>, fastOnly: boolean): StepResult {
  const rewards: Record<string, number> = {};
  const rewardBreakdowns: Record<string, RewardBreakdown> = {};
  const powerDistances = powerUpDistanceField(state);
  const maxDistance = Math.max(1, state.width + state.height - 2);
  const activeHazards = state.arena.enabled
    ? state.arena.hazards.flatMap((hazard) => hazardCells(hazard, state.tick).map((point) => ({ point, kind: hazard.kind })))
    : [];
  const moving = state.snakes.filter((snake) => snake.alive && snake.buffs.frozenUntil <= state.tick && (!fastOnly || snake.buffs.hasteUntil > state.tick));
  const candidates = new Map<string, { head: Point; direction: Direction }>();
  for (const snake of moving) {
    const direction = turn(snake.direction, actions[snake.id] ?? 0);
    candidates.set(snake.id, { head: movePoint(snake.segments[0], direction), direction });
    addReward(rewards, rewardBreakdowns, snake.id, "step", stepPenalty());
    if (state.mode === "battle") addReward(rewards, rewardBreakdowns, snake.id, "survival", survivalReward());
  }

  const headCounts = new Map<string, number>();
  candidates.forEach(({ head }) => headCounts.set(pointKey(head), (headCounts.get(pointKey(head)) ?? 0) + 1));
  const bodies = new Set<string>();
  state.snakes.filter((snake) => snake.alive).forEach((snake) => snake.segments.forEach((segment) => bodies.add(pointKey(segment))));

  let nextState = state;
  let nextSnakes = state.snakes.map((snake) => ({ ...snake, segments: snake.segments.map((point) => ({ ...point })), buffs: { ...snake.buffs } }));
  for (let index = 0; index < nextSnakes.length; index += 1) {
    let snake = nextSnakes[index];
    const candidate = candidates.get(snake.id);
    if (!candidate || !snake.alive) continue;
    const phasing = snake.buffs.phaseUntil > state.tick;
    const collidedWall = isWall(state, candidate.head);
    const collidedHead = (headCounts.get(pointKey(candidate.head)) ?? 0) > 1;
    const collidedObstacle = !phasing && state.obstacles.some((point) => samePoint(point, candidate.head));
    const collidedBody = !phasing && bodies.has(pointKey(candidate.head));
    const collidedHazard = activeHazards.find((entry) => samePoint(entry.point, candidate.head) && (!phasing || entry.kind === "laser"));
    if (collidedWall || collidedHead || collidedObstacle || collidedBody || collidedHazard) {
      const reason = collidedWall ? "hit the arena wall" : collidedHead ? "lost a head-on clash" : collidedObstacle ? "hit an obstacle" : collidedBody ? "hit a snake" : `hit a ${collidedHazard!.kind} hazard`;
      // Walls and the closing zone are absolute arena boundaries. Shield and
      // Phase counter combat/obstacle hazards, never boundary violations.
      const resolved = resolveFatalCollision({ ...nextState, snakes: nextSnakes }, snake, reason, !collidedWall);
      if (!resolved.alive) {
        addReward(rewards, rewardBreakdowns, snake.id, "death", deathReward());
        if (collidedBody || collidedHead) {
          const killer = state.snakes.find((other) => other.id !== snake.id && other.alive && (
            other.segments.some((seg) => samePoint(seg, candidate.head)) || candidates.get(other.id)?.head && samePoint(candidates.get(other.id)!.head, candidate.head)
          ));
          if (killer) {
            addReward(rewards, rewardBreakdowns, killer.id, "kill", killReward());
            const leaderBounty = state.arena.enabled && state.arena.leaderId === snake.id ? 10 : 0;
            const crownBounty = snake.buffs.crownUntil > state.tick ? 5 : 0;
            if (leaderBounty || crownBounty) addReward(rewards, rewardBreakdowns, killer.id, "bountyKill", leaderBounty + crownBounty);
            nextSnakes = nextSnakes.map((entry) => entry.id === killer.id ? {
              ...entry, score: entry.score + leaderBounty + crownBounty, kills: entry.kills + 1,
              bountyKills: entry.bountyKills + (leaderBounty || crownBounty ? 1 : 0),
            } : entry);
          }
        }
      }
      nextSnakes[index] = resolved;
      continue;
    }

    const previousHead = snake.segments[0];
    const previousDistance = state.food.length ? Math.min(...state.food.map((food) => distance(previousHead, food.position))) : 0;
    const powerUpPos = state.powerUp?.position ?? null;
    // A phased snake may occupy a normally blocked obstacle cell that is not
    // in the static BFS field. Manhattan distance avoids an artificial spike.
    const powerUpBefore = powerDistances?.get(pointKey(previousHead)) ?? (powerUpPos ? distance(previousHead, powerUpPos) : maxDistance);
    const magnetRadius = snake.buffs.magnetUntil > state.tick ? 3 : 0;
    const eatenIndex = nextState.food.findIndex((food) => distance(food.position, candidate.head) <= magnetRadius);
    const ate = eatenIndex >= 0;
    const multiplier = snake.buffs.doubleUntil > state.tick ? 2 : 1;
    const eatenFood = ate ? nextState.food[eatenIndex] : null;
    const newSegments = [candidate.head, ...snake.segments];
    if (!ate) newSegments.pop();
    snake = {
      ...snake, direction: candidate.direction, segments: newSegments,
      score: snake.score + (eatenFood ? eatenFood.value * multiplier : 0) + (snake.buffs.crownUntil > state.tick && state.tick % 10 === 0 ? multiplier : 0),
      foodEaten: snake.foodEaten + (ate ? 1 : 0),
      rareFoodEaten: snake.rareFoodEaten + (eatenFood?.kind === "rare" ? 1 : 0),
    };
    if (state.arena.enabled && state.arena.phase === "endgame" && state.tick >= state.arena.safeZone.telegraphAt) {
      const beforeZone = distanceToInset(previousHead, state.width, state.height, state.arena.safeZone.pendingInset);
      const afterZone = distanceToInset(candidate.head, state.width, state.height, state.arena.safeZone.pendingInset);
      addReward(rewards, rewardBreakdowns, snake.id, "zonePositioning", potentialShaping(beforeZone, afterZone, maxDistance, .35));
    }
    if (state.arena.objective) {
      const beforeObjective = distance(previousHead, state.arena.objective.position);
      const afterObjective = distance(candidate.head, state.arena.objective.position);
      addReward(rewards, rewardBreakdowns, snake.id, "objectiveApproach", potentialShaping(beforeObjective, afterObjective, maxDistance, .25));
    }
    if (ate) {
      const food = [...nextState.food];
      food.splice(eatenIndex, 1);
      nextState = { ...nextState, food };
      addReward(rewards, rewardBreakdowns, snake.id, "foodClaim", foodEatReward(multiplier));
      if (eatenFood?.kind === "rare") addReward(rewards, rewardBreakdowns, snake.id, "rareFoodClaim", foodEatReward((eatenFood.value - 1) * multiplier));
    } else if (state.food.length) {
      const newDistance = Math.min(...state.food.map((food) => distance(candidate.head, food.position)));
      addReward(rewards, rewardBreakdowns, snake.id, "foodApproach", foodDistanceDeltaReward(previousDistance, newDistance));
    }
    if (powerUpPos) {
      const claimed = samePoint(candidate.head, powerUpPos);
      const powerUpAfter = powerDistances?.get(pointKey(candidate.head)) ?? distance(candidate.head, powerUpPos);
      addReward(
        rewards, rewardBreakdowns, snake.id, "powerUpApproach",
        powerUpApproachReward(powerUpBefore / maxDistance, powerUpAfter / maxDistance, true, .95, claimed),
      );
    }
    if (nextState.powerUp && samePoint(nextState.powerUp.position, candidate.head)) {
      const claimedKind = nextState.powerUp.kind;
      const applied = applyPowerUp({ ...nextState, snakes: nextSnakes }, snake);
      nextState = applied.state;
      snake = applied.snake;
      if (claimedKind === "freeze") {
        const affected = new Map(applied.state.snakes.map((rival) => [rival.id, rival]));
        nextSnakes = nextSnakes.map((rival) => rival.id === snake.id ? rival : { ...rival, buffs: { ...affected.get(rival.id)!.buffs } });
      }
      addReward(rewards, rewardBreakdowns, snake.id, "powerUpClaim", powerUpClaimReward());
    }
    nextSnakes[index] = snake;
  }

  nextState = { ...nextState, snakes: nextSnakes };
  return { state: nextState, rewards, rewardBreakdowns };
}

function maybeSpawnPowerUp(state: GameState): GameState {
  if ((state.mode !== "battle" && state.mode !== "powerup") || state.powerUp || state.tick < state.nextPowerUpAt) return state;
  let kindIndex: number;
  let seed = state.seed;
  if (state.mode === "powerup") kindIndex = state.powerUpSpawnCount % POWER_UP_KINDS.length;
  else [kindIndex, seed] = randomInt(seed, POWER_UP_KINDS.length);
  const withSeed = { ...state, seed };
  const [position, finalSeed] = randomOpenCell(withSeed, 2, true);
  return {
    ...withSeed, seed: finalSeed,
    powerUpSpawnCount: state.powerUpSpawnCount + 1,
    powerUp: { id: state.tick, kind: POWER_UP_KINDS[kindIndex], position, expiresAt: state.tick + (state.mode === "powerup" ? 25 : 180) },
  };
}

function updateArenaPhase(state: GameState): GameState {
  if (!state.arena.enabled) return state;
  const phase = state.mode === "safezone" ? "endgame" : state.mode === "hazard" || state.mode === "objective" ? "midgame" : arenaPhaseAt(state.tick);
  if (phase === state.arena.phase) return state;
  const phaseEndsAt = phase === "opening" ? ADAPTIVE_PHASE_ENDS.opening : phase === "midgame" ? ADAPTIVE_PHASE_ENDS.midgame : ADAPTIVE_PHASE_ENDS.endgame;
  return {
    ...state,
    arena: { ...state.arena, phase, phaseEndsAt },
    events: addEvent(state.events, state.tick, `${phase.toUpperCase()} phase engaged`, phase === "endgame" ? "danger" : "info"),
  };
}

function activateDormantObstacles(state: GameState): GameState {
  if (!state.arena.enabled || !state.arena.dormantObstacles.length || state.tick < state.arena.obstaclesActivateAt) return state;
  const occupied = new Set(state.snakes.filter((snake) => snake.alive).flatMap((snake) => snake.segments.map(pointKey)));
  const activated = state.arena.dormantObstacles.filter((point) => !occupied.has(pointKey(point)) && insideSafeBounds(point, state.width, state.height, state.arena.safeZone.pendingInset));
  return {
    ...state,
    obstacles: [...state.obstacles, ...activated],
    arena: { ...state.arena, dormantObstacles: [] },
    events: addEvent(state.events, state.tick, "Dormant arena barriers activated", "danger"),
  };
}

function maybeScheduleHazard(state: GameState): GameState {
  if (!state.arena.enabled) return state;
  const hazards = state.arena.hazards.filter((hazard) => hazard.activeUntil >= state.tick);
  if ((state.arena.phase === "opening" && state.mode !== "hazard" && state.arena.mapArchetype !== "hazard") || state.tick < state.arena.nextHazardAt) {
    return hazards.length === state.arena.hazards.length ? state : { ...state, arena: { ...state.arena, hazards } };
  }
  const [hazard, seed] = createHazard(state.seed, state.tick, state.width, state.height, state.arena.phase);
  const interval = state.arena.phase === "endgame" || state.arena.mapArchetype === "hazard" ? 120 : 180;
  return {
    ...state,
    seed,
    arena: { ...state.arena, hazards: [...hazards, hazard], nextHazardAt: state.tick + interval },
    events: addEvent(state.events, state.tick, `${hazard.kind.toUpperCase()} hazard telegraph detected`, "danger"),
  };
}

function maybeSpawnObjective(state: GameState): GameState {
  if (!state.arena.enabled || state.arena.phase === "opening" || state.tick < state.arena.nextObjectiveAt || state.arena.objective) return state;
  const [position, seed] = randomOpenCell(state, 3, true);
  return {
    ...state,
    seed,
    arena: {
      ...state.arena,
      objective: {
        id: state.tick,
        position,
        radius: 2,
        expiresAt: state.tick + (state.arena.phase === "endgame" ? Math.floor(OBJECTIVE_LIFETIME_TICKS * 2 / 3) : OBJECTIVE_LIFETIME_TICKS),
        captureRequired: OBJECTIVE_CAPTURE_TICKS,
        progress: {},
        contested: false,
      },
    },
    events: addEvent(state.events, state.tick, "Energy core available", "power"),
  };
}

function processObjective(state: GameState, rewards: Record<string, number>, rewardBreakdowns: Record<string, RewardBreakdown>): GameState {
  const objective = state.arena.objective;
  if (!state.arena.enabled || !objective) return state;
  if (state.tick >= objective.expiresAt) return {
    ...state,
    arena: {
      ...state.arena,
      objective: null,
      nextObjectiveAt: state.tick + (state.arena.phase === "endgame" ? Math.floor(OBJECTIVE_COOLDOWN_TICKS * 2 / 3) : OBJECTIVE_COOLDOWN_TICKS),
    },
  };
  const occupants = state.snakes.filter((snake) => snake.alive && distance(snake.segments[0], objective.position) <= objective.radius);
  const contested = occupants.length > 1;
  if (occupants.length !== 1) return { ...state, arena: { ...state.arena, objective: { ...objective, contested } } };
  const controller = occupants[0];
  const progress = { ...objective.progress, [controller.id]: (objective.progress[controller.id] ?? 0) + 1 };
  if (progress[controller.id] < objective.captureRequired) return { ...state, arena: { ...state.arena, objective: { ...objective, progress, contested: false } } };
  addReward(rewards, rewardBreakdowns, controller.id, "objectiveCapture", 8);
  return {
    ...state,
    snakes: state.snakes.map((snake) => snake.id === controller.id ? { ...snake, score: snake.score + 5, objectiveCaptures: snake.objectiveCaptures + 1 } : snake),
    arena: {
      ...state.arena,
      objective: null,
      nextObjectiveAt: state.tick + (state.arena.phase === "endgame" ? Math.floor(OBJECTIVE_COOLDOWN_TICKS * 2 / 3) : OBJECTIVE_COOLDOWN_TICKS),
    },
    events: addEvent(state.events, state.tick, `${controller.name} captured the energy core`, "power"),
  };
}

function updateLeader(state: GameState): GameState {
  if (!state.arena.enabled || state.tick < 400) return state.arena.leaderId ? { ...state, arena: { ...state.arena, leaderId: null } } : state;
  const ranked = state.snakes.filter((snake) => snake.alive).sort((a, b) => b.score - a.score);
  const leaderId = ranked.length > 1 && ranked[0].score - ranked[1].score >= 3 ? ranked[0].id : null;
  return leaderId === state.arena.leaderId ? state : { ...state, arena: { ...state.arena, leaderId } };
}

function relocatePendingItems(state: GameState): GameState {
  if (!state.arena.enabled) return state;
  let next = { ...state, food: state.food.filter((food) => insideSafeBounds(food.position, state.width, state.height, state.arena.safeZone.pendingInset)) };
  if (next.powerUp && !insideSafeBounds(next.powerUp.position, next.width, next.height, next.arena.safeZone.pendingInset)) {
    const [position, seed] = randomOpenCell(next, 2, true);
    next = { ...next, seed, powerUp: { ...next.powerUp, position } };
  }
  if (next.arena.objective && !insideSafeBounds(next.arena.objective.position, next.width, next.height, next.arena.safeZone.pendingInset)) {
    const [position, seed] = randomOpenCell(next, 3, true);
    next = { ...next, seed, arena: { ...next.arena, objective: { ...next.arena.objective, position } } };
  }
  return next;
}

function closeSafeZone(state: GameState, rewards: Record<string, number>, rewardBreakdowns: Record<string, RewardBreakdown>): GameState {
  if (!state.arena.enabled || state.arena.phase !== "endgame" || state.tick < state.arena.safeZone.closesAt) return state;
  const closingInset = state.arena.safeZone.pendingInset;
  let next = { ...state, arena: { ...state.arena, safeZone: advanceSafeZone(state.arena.safeZone, state.width, state.height) } };
  next = relocatePendingItems(next);
  next = {
    ...next,
    snakes: next.snakes.map((snake) => {
      if (!snake.alive || insideSafeBounds(snake.segments[0], next.width, next.height, closingInset)) return snake;
      if (snake.buffs.secondChance > 0) return { ...snake, segments: [safestOpenCell(next, snake.id)], buffs: { ...snake.buffs, secondChance: snake.buffs.secondChance - 1 } };
      addReward(rewards, rewardBreakdowns, snake.id, "death", deathReward());
      return { ...snake, alive: false, deathReason: "was consumed by the closing zone" };
    }),
    events: addEvent(next.events, next.tick, "Safe zone contracted", "danger"),
  };
  return next;
}

function applyArenaHazards(state: GameState, rewards: Record<string, number>, rewardBreakdowns: Record<string, RewardBreakdown>): GameState {
  if (!state.arena.enabled) return state;
  const active = state.arena.hazards.flatMap((hazard) => hazardCells(hazard, state.tick).map((point) => ({ point, kind: hazard.kind })));
  if (!active.length) return state;
  return {
    ...state,
    snakes: state.snakes.map((snake) => {
      if (!snake.alive) return snake;
      const collision = active.find((entry) => samePoint(entry.point, snake.segments[0]) && (snake.buffs.phaseUntil <= state.tick || entry.kind === "laser"));
      if (!collision) return snake;
      const resolved = resolveFatalCollision(state, snake, `hit a ${collision.kind} hazard`);
      if (!resolved.alive) addReward(rewards, rewardBreakdowns, snake.id, "death", deathReward());
      return resolved;
    }),
  };
}

export function stepGame(state: GameState, actions: Record<string, RelativeAction>): StepResult {
  if (state.status === "finished") return { state, rewards: {}, rewardBreakdowns: {} };
  let next: GameState = { ...state, tick: state.tick + 1 };
  next = updateArenaPhase(next);
  next = activateDormantObstacles(next);
  next = maybeScheduleHazard(next);
  next = maybeSpawnObjective(next);
  const first = executeSubstep(next, actions, false);
  next = first.state;
  const rewards = { ...first.rewards };
  const rewardBreakdowns = { ...first.rewardBreakdowns };
  if (next.snakes.some((snake) => snake.alive && snake.buffs.hasteUntil > next.tick)) {
    // Haste is a two-cell dash using the heading selected for this tick. The
    // observation and safe-action mask therefore look through both cells.
    const second = executeSubstep(next, {}, true);
    next = second.state;
    Object.entries(second.rewards).forEach(([id, reward]) => { rewards[id] = (rewards[id] ?? 0) + reward; });
    Object.entries(second.rewardBreakdowns).forEach(([id, incoming]) => {
      const current = rewardBreakdowns[id] ?? emptyRewardBreakdown();
      for (const component of Object.keys(incoming) as Array<keyof RewardBreakdown>) current[component] += incoming[component];
      rewardBreakdowns[id] = current;
    });
  }
  next = applyArenaHazards(next, rewards, rewardBreakdowns);
  next = processObjective(next, rewards, rewardBreakdowns);
  if (next.arena.enabled && next.arena.phase === "endgame" && next.tick >= next.arena.safeZone.telegraphAt) next = relocatePendingItems(next);
  next = closeSafeZone(next, rewards, rewardBreakdowns);
  next = updateLeader(next);
  next = replenishFood(next);
  next = maybeSpawnPowerUp(next);
  if (next.powerUp && next.tick >= next.powerUp.expiresAt) next = { ...next, powerUp: null, nextPowerUpAt: next.tick + (next.mode === "powerup" ? 2 : 35) };

  const justDied = next.snakes.filter((snake) => !snake.alive && state.snakes.find((old) => old.id === snake.id)?.alive);
  let events = next.events;
  justDied.forEach((snake) => { events = addEvent(events, next.tick, `${snake.name} ${snake.deathReason}`, "danger"); });
  next = { ...next, events };

  const alive = next.snakes.filter((snake) => snake.alive);
  const battleFinished = next.mode === "battle" && alive.length <= 1;
  const trainingFinished = next.mode !== "battle" && (alive.length === 0 || next.tick >= (next.mode === "powerup" ? 360 : 900));
  if (battleFinished || trainingFinished) {
    const winner = battleFinished ? alive[0] : null;
    if (winner) {
      next = {
        ...next, status: "finished", winnerId: winner.id,
        snakes: next.snakes.map((snake) => snake.id === winner.id ? { ...snake, score: snake.score + 15 } : snake),
        events: addEvent(next.events, next.tick, `${winner.name} is the last snake standing!`, "victory"),
      };
      addReward(rewards, rewardBreakdowns, winner.id, "win", winReward());
    } else next = { ...next, status: "finished" };
  }
  return { state: next, rewards, rewardBreakdowns };
}

export function relativeDirection(direction: Direction, action: RelativeAction): Direction {
  return turn(direction, action);
}

export function isDangerAt(state: GameState, snake: Snake, direction: Direction, steps = 1): boolean {
  let point = snake.segments[0];
  for (let step = 0; step < steps; step += 1) {
    point = movePoint(point, direction);
    if (isWall(state, point)) return true;
    if (state.arena.enabled && state.arena.phase === "endgame" && state.tick + step + 1 >= state.arena.safeZone.closesAt && !insideSafeBounds(point, state.width, state.height, state.arena.safeZone.pendingInset)) return true;
    if (state.arena.enabled && state.arena.hazards.some((hazard) =>
      projectedHazardCells(hazard, state.tick, step + 1).some((cell) => samePoint(cell, point)) && (snake.buffs.phaseUntil <= state.tick || hazard.kind === "laser"))) return true;
    if (state.arena.enabled && state.arena.obstaclesActivateAt <= state.tick + step + 1 && state.arena.dormantObstacles.some((cell) => samePoint(cell, point))) return true;
    if (snake.buffs.phaseUntil <= state.tick) {
      if (state.obstacles.some((obstacle) => samePoint(obstacle, point))) return true;
      if (state.snakes.some((other) => other.alive && other.segments.some((segment) => samePoint(segment, point)))) return true;
    }
  }
  return false;
}

const HEADING_ANGLE: Record<Direction, number> = {
  up: -Math.PI / 2,
  right: 0,
  down: Math.PI / 2,
  left: Math.PI,
};

function rayEntityAt(state: GameState, snake: Snake, point: Point): RayEntity | null {
  if (isWall(state, point)) return "wall";
  if (state.obstacles.some((entry) => samePoint(entry, point))) return "obstacle";
  if (snake.segments.some((entry) => samePoint(entry, point))) return "self";
  if (state.snakes.some((entry) => entry.id !== snake.id && entry.alive && entry.segments.some((segment) => samePoint(segment, point)))) return "enemy";
  if (state.food.some((entry) => samePoint(entry.position, point))) return "food";
  if (state.powerUp && samePoint(state.powerUp.position, point)) return "powerUp";
  return null;
}

export function castRays(state: GameState, snake: Snake, rayCount = 16): RayHit[] {
  const head = snake.segments[0];
  const diagonal = Math.hypot(state.width, state.height);
  const hits: RayHit[] = [];
  for (let index = 0; index < rayCount; index += 1) {
    const angle = HEADING_ANGLE[snake.direction] + index * Math.PI * 2 / rayCount;
    const dx = Math.cos(angle);
    const dy = Math.sin(angle);
    let lastKey = pointKey(head);
    let hit: RayHit | null = null;
    for (let distanceAlongRay = .25; distanceAlongRay <= diagonal + 1; distanceAlongRay += .25) {
      const point = {
        x: Math.floor(head.x + .5 + dx * distanceAlongRay),
        y: Math.floor(head.y + .5 + dy * distanceAlongRay),
      };
      const key = pointKey(point);
      if (key === lastKey) continue;
      lastKey = key;
      const entity = rayEntityAt(state, snake, point);
      if (entity) {
        hit = { distance: Math.min(1, Math.hypot(point.x - head.x, point.y - head.y) / diagonal), entity, point };
        break;
      }
    }
    hits.push(hit ?? { distance: 1, entity: "wall", point: head });
  }
  return hits;
}
