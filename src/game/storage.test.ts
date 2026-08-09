import { describe, expect, it } from "vitest";
import { createBrain } from "./ai";
import { activeProfiles, addSnake, archiveSnake, defaultProfiles, deserializeRoster, migrateLegacyProfiles, normalizeRoster, restoreSnake } from "./storage";

describe("IndexedDB roster migration", () => {
  it("preserves legacy career data but resets incompatible learning", () => {
    const brain = { ...createBrain("nova"), generation: 91, episodes: 90, bestScore: 17, q: { state: [1, 2, 3] } };
    const migrated = migrateLegacyProfiles([{ snakeId: "nova", name: "Nova Viper", color: "#68f7c1", accent: "#d7fff1", highScore: 23, wins: 7, matches: 12, brain }]);
    const nova = migrated.find((profile) => profile.snakeId === "nova")!;
    expect(nova.highScore).toBe(23);
    expect(nova.brain.generation).toBe(0);
    expect(nova.legacyTraining).toMatchObject({ generation: 91, episodes: 90, learnedStates: 1 });
  });

  it("imports archived v2 custom snakes and retains their identity", () => {
    const custom = addSnake(defaultProfiles(), { name: "Arctic Byte", color: "#43d3ff", id: "arctic", createdAt: 99 }).profiles;
    const archived = archiveSnake(custom, "arctic").profiles;
    const roster = deserializeRoster(JSON.stringify({ version: 2, profiles: archived }), null);
    const arctic = roster.profiles.find((profile) => profile.snakeId === "arctic")!;
    expect(arctic.active).toBe(false);
    expect(arctic.name).toBe("Arctic Byte");
    expect(arctic.brain.generation).toBe(0);
  });

  it("normalizes a valid v3 roster without resetting DQN metadata", () => {
    const profiles = defaultProfiles().map((profile) => profile.snakeId === "nova" ? { ...profile, brain: { ...profile.brain, generation: 42 } } : profile);
    const normalized = normalizeRoster({ version: 3, profiles, hyperparameters: { learningRate: .001 } });
    expect(normalized?.profiles.find((profile) => profile.snakeId === "nova")?.brain.generation).toBe(42);
  });

  it("preserves career records but resets a pre-v2 model and its learning counters", () => {
    const profiles = defaultProfiles().map((profile) => profile.snakeId === "nova" ? {
      ...profile,
      highScore: 77,
      wins: 12,
      matches: 30,
      brain: { ...profile.brain, modelVersion: 1, trainingSpecVersion: 1, observationSize: 150, generation: 42, episodes: 500, epsilon: .05 },
    } : profile);
    const normalized = normalizeRoster({ version: 4, profiles, hyperparameters: {} });
    const nova = normalized?.profiles.find((profile) => profile.snakeId === "nova");
    expect(nova).toMatchObject({ highScore: 77, wins: 12, matches: 30 });
    expect(nova?.brain).toMatchObject({ modelVersion: 3, trainingSpecVersion: 3, observationSize: 228, generation: 0, episodes: 0, environmentSteps: 0 });
    expect(nova?.legacyTraining).toMatchObject({ generation: 42, episodes: 500 });
  });

  it("enforces unique names, two active snakes, and the eight-snake maximum", () => {
    let profiles = defaultProfiles();
    expect(addSnake(profiles, { name: "nova viper", color: "#123456" }).error).toMatch(/already exists/);
    for (let index = 0; index < 4; index += 1) profiles = addSnake(profiles, { name: `Custom ${index}`, color: "#123456", id: `custom-${index}` }).profiles;
    expect(activeProfiles(profiles)).toHaveLength(8);
    expect(addSnake(profiles, { name: "Ninth Snake", color: "#abcdef" }).error).toMatch(/limited to 8/);
    profiles = archiveSnake(profiles, "nova").profiles;
    profiles = archiveSnake(profiles, "ember").profiles;
    profiles = archiveSnake(profiles, "volt").profiles;
    profiles = archiveSnake(profiles, "echo").profiles;
    profiles = archiveSnake(profiles, "custom-0").profiles;
    profiles = archiveSnake(profiles, "custom-1").profiles;
    expect(activeProfiles(profiles)).toHaveLength(2);
    expect(archiveSnake(profiles, "custom-2").error).toMatch(/at least 2/);
    expect(restoreSnake(profiles, "nova").profiles.find((profile) => profile.snakeId === "nova")?.active).toBe(true);
  });
});
