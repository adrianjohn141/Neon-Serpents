export type ParticipantIdentity = { id: string };

/** Return a deterministic, duplicate-free roster anchored on the primary agent. */
export function selectParticipantRoster<T extends ParticipantIdentity>(list: T[], primary: T, battleSize: number): T[] {
  const primaryIndex = list.indexOf(primary);
  if (primaryIndex < 0) throw new Error("Primary training agent is not present in the roster.");
  const count = Math.min(Math.max(1, battleSize), list.length);
  return Array.from({ length: count }, (_, offset) => list[(primaryIndex + offset) % list.length]);
}
