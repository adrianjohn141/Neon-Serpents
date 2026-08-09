import "fake-indexeddb/auto";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { defaultRoster, loadAppData, saveRoster } from "./storage";

const localStorage = (() => {
  const values = new Map<string, string>();
  return {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value),
  };
})();

describe("IndexedDB persistence", () => {
  beforeAll(async () => {
    vi.stubGlobal("window", { localStorage });
    await new Promise<void>((resolve) => {
      const request = indexedDB.deleteDatabase("neon-serpents");
      request.onsuccess = () => resolve();
      request.onerror = () => resolve();
    });
  });

  it("migrates recovery data and round-trips the resulting v3 roster", async () => {
    const legacy = defaultRoster().profiles.map((profile) => ({ ...profile, brain: { ...profile.brain, generation: 12, q: { learned: [1, 2, 3] } } }));
    localStorage.setItem("neon-serpents:roster:v2", JSON.stringify({ version: 2, profiles: legacy }));
    const migrated = await loadAppData();
    expect(migrated.migrated).toBe(true);
    expect(migrated.roster.profiles[0].brain.generation).toBe(0);
    expect(migrated.roster.profiles[0].legacyTraining?.learnedStates).toBe(1);
    expect(localStorage.getItem("neon-serpents:roster:v2")).not.toBeNull();

    migrated.roster.profiles[0].brain.generation = 73;
    await saveRoster(migrated.roster);
    const loaded = await loadAppData();
    expect(loaded.persistent).toBe(true);
    expect(loaded.migrated).toBe(false);
    expect(loaded.roster.profiles[0].brain.generation).toBe(73);
  });
});
