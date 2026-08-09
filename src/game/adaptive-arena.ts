import {
  ADAPTIVE_PHASE_ENDS, MAP_ARCHETYPES, SAFE_ZONE_INTERVAL_TICKS, SAFE_ZONE_TELEGRAPH_TICKS,
} from "./constants";
import { randomInt } from "./rng";
import type {
  ArenaBounds, ArenaHazard, ArenaPhase, ArenaState, Direction, MapArchetype,
  OpponentProfile, Point,
} from "./types";

const key = (point: Point) => `${point.x},${point.y}`;

export function arenaPhaseAt(tick: number): ArenaPhase {
  if (tick < ADAPTIVE_PHASE_ENDS.opening) return "opening";
  if (tick < ADAPTIVE_PHASE_ENDS.midgame) return "midgame";
  return "endgame";
}

export function isHeldOutMapSeed(seed: number, archetype: MapArchetype): boolean {
  return ((seed >>> 0) + MAP_ARCHETYPES.indexOf(archetype)) % 5 === 0;
}

export function mapForSeed(seed: number, heldOut: boolean): MapArchetype {
  for (let offset = 0; offset < MAP_ARCHETYPES.length * 2; offset += 1) {
    const archetype = MAP_ARCHETYPES[(seed + offset) % MAP_ARCHETYPES.length];
    if (isHeldOutMapSeed(seed, archetype) === heldOut) return archetype;
  }
  return MAP_ARCHETYPES[seed % MAP_ARCHETYPES.length];
}

export function boundsForInset(width: number, height: number, inset: number) {
  return { minX: inset, minY: inset, maxX: width - 1 - inset, maxY: height - 1 - inset };
}

export function insideSafeBounds(point: Point, width: number, height: number, inset: number): boolean {
  const bounds = boundsForInset(width, height, inset);
  return point.x >= bounds.minX && point.x <= bounds.maxX && point.y >= bounds.minY && point.y <= bounds.maxY;
}

export function defaultOpponentProfile(): OpponentProfile {
  return { turnBias: 0, aggressionRate: 0, commonTarget: "survival", powerUpRate: 0, typicalDeathCause: "other", samples: 0 };
}

export function createArenaState(
  width: number,
  height: number,
  mapArchetype: MapArchetype,
  enabled: boolean,
  seriesRound: 1 | 2 | 3,
  opponentProfiles: Record<string, OpponentProfile>,
): ArenaState {
  const safeZone: ArenaBounds = {
    inset: 0,
    pendingInset: 1,
    telegraphAt: ADAPTIVE_PHASE_ENDS.midgame,
    closesAt: ADAPTIVE_PHASE_ENDS.midgame + SAFE_ZONE_TELEGRAPH_TICKS,
    minimumWidth: Math.min(56, width),
    minimumHeight: Math.min(28, height),
  };
  return {
    enabled,
    phase: "opening",
    phaseEndsAt: ADAPTIVE_PHASE_ENDS.opening,
    mapArchetype,
    safeZone,
    objective: null,
    nextObjectiveAt: ADAPTIVE_PHASE_ENDS.opening + 50,
    hazards: [],
    // The rapid-hazard archetype is the one deliberate opening-phase
    // exception; every other map remains hazard-free until midgame.
    nextHazardAt: mapArchetype === "hazard" ? 180 : ADAPTIVE_PHASE_ENDS.opening + 90,
    dormantObstacles: midgameObstacleCells(mapArchetype, width, height),
    obstaclesActivateAt: ADAPTIVE_PHASE_ENDS.opening + 50,
    seriesRound,
    opponentProfiles,
    leaderId: null,
  };
}

function addLine(cells: Point[], x: number, y: number, dx: number, dy: number, length: number, holes: number[] = []) {
  for (let index = 0; index < length; index += 1) if (!holes.includes(index)) cells.push({ x: x + dx * index, y: y + dy * index });
}

export function makeMapObstacles(archetype: MapArchetype, width: number, height: number): Point[] {
  const cells: Point[] = [];
  const cx = Math.floor(width / 2);
  const cy = Math.floor(height / 2);
  if (archetype === "corridors") {
    for (const [index, x] of [Math.floor(width * .25), cx, Math.floor(width * .75)].entries()) {
      const gap = index % 2 ? Math.floor(height * .7) : Math.floor(height * .3);
      for (let y = 6; y < height - 6; y += 1) if (Math.abs(y - gap) > 2) cells.push({ x, y });
    }
  } else if (archetype === "islands" || archetype === "hazard") {
    const centers = [{ x: cx - 16, y: cy - 8 }, { x: cx + 16, y: cy - 8 }, { x: cx - 16, y: cy + 8 }, { x: cx + 16, y: cy + 8 }];
    for (const center of centers) for (let y = -2; y <= 2; y += 1) for (let x = -3; x <= 3; x += 1) cells.push({ x: center.x + x, y: center.y + y });
    if (archetype === "hazard") {
      addLine(cells, cx - 5, cy, 1, 0, 11, [4, 5, 6]);
      addLine(cells, cx, cy - 5, 0, 1, 11, [4, 5, 6]);
    }
  } else if (archetype === "fortress") {
    addLine(cells, cx - 10, cy - 7, 1, 0, 21, [9, 10, 11]);
    addLine(cells, cx - 10, cy + 7, 1, 0, 21, [9, 10, 11]);
    addLine(cells, cx - 10, cy - 6, 0, 1, 13, [5, 6, 7]);
    addLine(cells, cx + 10, cy - 6, 0, 1, 13, [5, 6, 7]);
  } else if (archetype === "quadrants") {
    const verticalCenter = cy - 5;
    const horizontalCenter = cx - 5;
    addLine(cells, cx, 5, 0, 1, height - 10, [verticalCenter - 2, verticalCenter - 1, verticalCenter, verticalCenter + 1, verticalCenter + 2]);
    addLine(cells, 5, cy, 1, 0, width - 10, [horizontalCenter - 2, horizontalCenter - 1, horizontalCenter, horizontalCenter + 1, horizontalCenter + 2]);
  }
  const unique = new Map(cells
    .filter((point) => point.x > 4 && point.y > 4 && point.x < width - 5 && point.y < height - 5)
    .map((point) => [key(point), point]));
  return [...unique.values()];
}

export function midgameObstacleCells(archetype: MapArchetype, width: number, height: number): Point[] {
  if (archetype === "open") return [];
  const cx = Math.floor(width / 2);
  const cy = Math.floor(height / 2);
  const cells: Point[] = [];
  if (archetype === "corridors") {
    addLine(cells, cx - 7, cy, 1, 0, 15, [6, 7, 8]);
  } else if (archetype === "quadrants") {
    for (const point of [{ x: cx - 5, y: cy - 5 }, { x: cx + 5, y: cy - 5 }, { x: cx - 5, y: cy + 5 }, { x: cx + 5, y: cy + 5 }]) cells.push(point);
  } else {
    addLine(cells, cx - 4, cy - 3, 1, 0, 3);
    addLine(cells, cx + 2, cy + 3, 1, 0, 3);
  }
  return cells;
}

export function canShrink(safeZone: ArenaBounds, width: number, height: number): boolean {
  return width - (safeZone.inset + 1) * 2 >= safeZone.minimumWidth && height - (safeZone.inset + 1) * 2 >= safeZone.minimumHeight;
}

export function advanceSafeZone(safeZone: ArenaBounds, width: number, height: number): ArenaBounds {
  if (!canShrink(safeZone, width, height)) return { ...safeZone, telegraphAt: Number.MAX_SAFE_INTEGER, closesAt: Number.MAX_SAFE_INTEGER };
  const inset = safeZone.pendingInset;
  return {
    ...safeZone,
    inset,
    pendingInset: inset + 1,
    telegraphAt: safeZone.closesAt + SAFE_ZONE_INTERVAL_TICKS - SAFE_ZONE_TELEGRAPH_TICKS,
    closesAt: safeZone.closesAt + SAFE_ZONE_INTERVAL_TICKS,
  };
}

export function createHazard(seed: number, tick: number, width: number, height: number, phase: ArenaPhase): [ArenaHazard, number] {
  let nextSeed = seed;
  let kindIndex: number;
  [kindIndex, nextSeed] = randomInt(nextSeed, 3);
  let coordinate: number;
  [coordinate, nextSeed] = randomInt(nextSeed, kindIndex === 0 ? Math.max(1, height - 10) : Math.max(1, width - 10));
  const kind = (["laser", "sweeper", "blocks"] as const)[kindIndex];
  const direction: Direction = kindIndex === 0 ? "right" : kindIndex === 1 ? "down" : "right";
  const origin = kindIndex === 0 ? { x: 1, y: coordinate + 5 } : { x: coordinate + 5, y: 1 };
  const warning = phase === "endgame" ? 35 : 45;
  const activatesAt = tick + warning;
  return [{ id: tick, kind, origin, direction, length: kind === "laser" ? width - 2 : Math.min(9, kindIndex === 1 ? height - 2 : width - 2), telegraphAt: tick, activatesAt, activeUntil: activatesAt + (kind === "laser" ? 12 : 36) }, nextSeed];
}

export function hazardCells(hazard: ArenaHazard, tick: number): Point[] {
  if (tick < hazard.activatesAt || tick > hazard.activeUntil) return [];
  const activeTick = Math.max(0, tick - hazard.activatesAt);
  if (hazard.kind === "laser") {
    const horizontal = hazard.direction === "left" || hazard.direction === "right";
    return Array.from({ length: hazard.length }, (_, index) => horizontal ? { x: hazard.origin.x + index, y: hazard.origin.y } : { x: hazard.origin.x, y: hazard.origin.y + index });
  }
  if (hazard.kind === "sweeper") {
    const offset = Math.max(0, Math.floor(activeTick / 3));
    const center = { x: hazard.origin.x, y: hazard.origin.y + offset };
    return Array.from({ length: hazard.length }, (_, index) => ({ x: center.x + index - Math.floor(hazard.length / 2), y: center.y }));
  }
  return Array.from({ length: hazard.length }, (_, index) => ({ x: hazard.origin.x + index, y: hazard.origin.y }));
}

export function projectedHazardCells(hazard: ArenaHazard, tick: number, lookahead: number): Point[] {
  if (hazard.activatesAt > tick + lookahead || hazard.activeUntil < tick) return [];
  const at = Math.max(tick + lookahead, hazard.activatesAt);
  return hazardCells(hazard, Math.min(at, hazard.activeUntil));
}
