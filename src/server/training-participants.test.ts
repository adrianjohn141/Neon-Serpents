import { describe, expect, it } from "vitest";
import { selectParticipantRoster } from "./training-participants";

describe("training participant rosters", () => {
  it("keeps the primary and opponents unique for every roster position", () => {
    const agents = [{ id: "nova" }, { id: "ember" }, { id: "volt" }, { id: "echo" }];
    for (const primary of agents) {
      const roster = selectParticipantRoster(agents, primary, 4);
      expect(roster[0]).toBe(primary);
      expect(new Set(roster.map((agent) => agent.id)).size).toBe(4);
    }
  });
});
