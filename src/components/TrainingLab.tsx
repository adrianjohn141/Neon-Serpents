"use client";

import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { actionLabel, DEFAULT_HYPERPARAMETERS, effectiveHyperparameters, encodeObservation } from "@/game/ai";
import { createGame, stepGame, isDangerAt, relativeDirection } from "@/game/engine";
import {
  activeProfiles, addSnake, archiveSnake, archivedProfiles, defaultRoster, loadAppData,
  MAX_ACTIVE_SNAKES, mergeBrains, MIN_ACTIVE_SNAKES, normalizeHyperparameters, profileToDefinition, restoreSnake, saveProfiles,
} from "@/game/storage";
import type { CurriculumConfig, GameState, Hyperparameters, PersistedProfile, TrainingScenario } from "@/game/types";
import type { BrainBundle } from "@/game/model-bundle";
import { useAiWorker } from "@/game/useAiWorker";
import { ArenaCanvas } from "./ArenaCanvas";

type SoloScenario = Extract<TrainingScenario, "survival" | "powerup" | "safezone" | "hazard" | "objective">;
type VisualRun = { snakeId: string; state: GameState; scenario: SoloScenario };
const PARAMETER_FIELDS: Array<{ key: keyof Hyperparameters; label: string; min: number; max: number; step: number }> = [
  { key: "learningRate", label: "Learning rate", min: .00001, max: .01, step: .00001 },
  { key: "discountFactor", label: "Discount factor", min: .8, max: .999, step: .001 },
  { key: "epsilonStart", label: "Starting epsilon", min: .1, max: 1, step: .01 },
  { key: "epsilonMin", label: "Minimum epsilon", min: .001, max: .5, step: .001 },
  { key: "epsilonDecaySteps", label: "Exploration decay steps", min: 1000, max: 10000000, step: 1000 },
  { key: "batchSize", label: "Batch size", min: 16, max: 128, step: 16 },
  { key: "replayBufferSize", label: "Replay capacity", min: 1000, max: 50000, step: 1000 },
  { key: "warmupTransitions", label: "Warm-up", min: 0, max: 10000, step: 100 },
  { key: "trainEverySteps", label: "Train interval", min: 1, max: 32, step: 1 },
  { key: "targetSyncSteps", label: "Target sync", min: 100, max: 20000, step: 100 },
  { key: "nStep", label: "N-step return", min: 1, max: 10, step: 1 },
  { key: "priorityAlpha", label: "Replay priority", min: 0, max: 1, step: .05 },
  { key: "priorityBetaStart", label: "Bias correction", min: 0, max: 1, step: .05 },
];

function visualScenario(profile: PersistedProfile, curriculum: CurriculumConfig): SoloScenario {
  if (profile.brain.scenarioSteps.survival < curriculum.navigationWarmupSteps) return "survival";
  if (profile.brain.scenarioSteps.powerup < curriculum.powerUpWarmupSteps) return "powerup";
  if (profile.brain.scenarioSteps.safezone < curriculum.safeZoneWarmupSteps) return "safezone";
  if (profile.brain.scenarioSteps.hazard < curriculum.hazardWarmupSteps) return "hazard";
  if (profile.brain.scenarioSteps.objective < curriculum.objectiveWarmupSteps) return "objective";
  const weighted: Array<[SoloScenario, number]> = [["survival", curriculum.survivalRatio], ["powerup", curriculum.powerUpRatio], ["safezone", curriculum.safeZoneRatio], ["hazard", curriculum.hazardRatio], ["objective", curriculum.objectiveRatio]];
  let roll = Math.random() * weighted.reduce((sum, entry) => sum + entry[1], 0);
  for (const [scenario, weight] of weighted) { roll -= weight; if (roll <= 0) return scenario; }
  return "objective";
}

const gameModeFor = (scenario: SoloScenario) => scenario === "survival" ? "training" : scenario;

function makeRuns(profiles: PersistedProfile[], curriculum: CurriculumConfig): VisualRun[] {
  return activeProfiles(profiles).map((profile, index) => {
    const scenario = visualScenario(profile, curriculum);
    return {
      snakeId: profile.snakeId, scenario,
      state: createGame({ mode: gameModeFor(scenario), snakes: [profileToDefinition(profile)], seed: 8128 + index * 1009 }),
    };
  });
}

export function TrainingLab() {
  const initial = defaultRoster();
  const [profiles, setProfiles] = useState(initial.profiles);
  const [hyperparameters, setHyperparameters] = useState(initial.hyperparameters);
  const [curriculum, setCurriculum] = useState(initial.curriculum);
  const [runs, setRuns] = useState<VisualRun[]>(() => makeRuns(initial.profiles, initial.curriculum));
  const [loading, setLoading] = useState(true);
  const [storageMessage, setStorageMessage] = useState("");
  const [visualRunning, setVisualRunning] = useState(false);
  const [fastRunning, setFastRunning] = useState(false);
  const [lastActions, setLastActions] = useState<Record<string, string>>({});
  const [visualSpeed, setVisualSpeed] = useState(72);
  const [newName, setNewName] = useState("");
  const [newColor, setNewColor] = useState("#43d3ff");
  const [rosterError, setRosterError] = useState("");
  const [tuningOpen, setTuningOpen] = useState(false);
  const [hiveConfirmOpen, setHiveConfirmOpen] = useState(false);
  const [tuningTarget, setTuningTarget] = useState("global");
  const [draftParameters, setDraftParameters] = useState<Hyperparameters>(initial.hyperparameters);
  const profilesRef = useRef(profiles);
  const runsRef = useRef(runs);
  const hyperRef = useRef(hyperparameters);
  const curriculumRef = useRef(curriculum);
  const initializedRef = useRef(false);
  const ai = useAiWorker();

  const persist = useCallback(async (nextProfiles: PersistedProfile[], nextParameters = hyperRef.current) => {
    try { await saveProfiles(nextProfiles, nextParameters, curriculumRef.current); setStorageMessage(""); }
    catch { setStorageMessage("Training continues in memory, but IndexedDB could not save these changes."); }
  }, []);

  useEffect(() => {
    let alive = true;
    void loadAppData().then((result) => {
      if (!alive) return;
      profilesRef.current = result.roster.profiles;
      hyperRef.current = result.roster.hyperparameters;
      curriculumRef.current = result.roster.curriculum;
      const nextRuns = makeRuns(result.roster.profiles, result.roster.curriculum);
      runsRef.current = nextRuns;
      setProfiles(result.roster.profiles);
      setHyperparameters(result.roster.hyperparameters);
      setCurriculum(result.roster.curriculum);
      setDraftParameters(result.roster.hyperparameters);
      setRuns(nextRuns);
      setStorageMessage(result.error ?? (result.migrated ? "Legacy profiles found. Fresh DQN models are being created." : ""));
      setLoading(false);
    });
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    if (loading || !ai.workerAvailable || initializedRef.current) return;
    initializedRef.current = true;
    ai.initialize(activeProfiles(profilesRef.current), hyperRef.current, curriculumRef.current);
  }, [loading, ai.workerAvailable, ai.initialize]);

  useEffect(() => {
    if (!ai.ready) return;
    const merged = mergeBrains(profilesRef.current, ai.brains);
    profilesRef.current = merged;
    setProfiles(merged);
    void persist(merged);
  }, [ai.ready, ai.brains, ai.saveCounter, persist]);

  useEffect(() => {
    if (!visualRunning || fastRunning || !ai.ready) return;
    let cancelled = false;
    let timer = 0;
    const tick = async () => {
      const currentRuns = runsRef.current;
      const observations: Record<string, { data: number[], safeActions: import("@/game/types").RelativeAction[] }> = {};
      for (const run of currentRuns) {
        const snake = run.state.snakes[0];
        if (snake) {
          const data = Array.from(encodeObservation(run.state, snake));
          const lookahead = snake.buffs.hasteUntil > run.state.tick || snake.buffs.visionUntil > run.state.tick ? 2 : 1;
          const safeActions = ([0, 1, 2] as import("@/game/types").RelativeAction[]).filter(a => !isDangerAt(run.state, snake, relativeDirection(snake.direction, a), lookahead));
          observations[run.snakeId] = { data, safeActions };
        }
      }
      const actions = await ai.act(observations, true);
      if (cancelled) return;
      const experiences: Parameters<typeof ai.observe>[0] = {};
      const contexts: NonNullable<Parameters<typeof ai.observe>[1]> = {};
      const outcomes: Parameters<typeof ai.finishEpisodes>[0] = {};
      const labels: Record<string, string> = {};
      const nextRuns = currentRuns.map((run, index) => {
        const snake = run.state.snakes[0];
        const action = actions[run.snakeId] ?? 0;
        const result = stepGame(run.state, { [run.snakeId]: action });
        const nextSnake = result.state.snakes[0];
        experiences[run.snakeId] = {
          state: observations[run.snakeId].data, action, reward: result.rewards[run.snakeId] ?? 0,
          nextState: Array.from(encodeObservation(result.state, nextSnake)), terminal: result.state.status === "finished",
          rewardBreakdown: result.rewardBreakdowns[run.snakeId],
          deathCause: !nextSnake.alive ? (nextSnake.deathReason?.includes("zone") ? "zone" : nextSnake.deathReason?.includes("hazard") ? "hazard" : nextSnake.deathReason?.includes("wall") ? "wall" : nextSnake.deathReason?.includes("obstacle") ? "obstacle" : "other") : undefined,
        };
        const opportunity = !run.state.powerUp && Boolean(result.state.powerUp);
        const claimed = nextSnake.powerUps > snake.powerUps;
        contexts[run.snakeId] = {
          scenario: run.scenario, opportunity, claimed,
          approachMiss: Boolean(run.state.powerUp) && !result.state.powerUp && !claimed,
        };
        labels[run.snakeId] = actionLabel(action);
        if (result.state.status === "finished") {
          outcomes[run.snakeId] = { score: nextSnake.score, food: nextSnake.foodEaten, won: false };
          const profile = profilesRef.current.find((entry) => entry.snakeId === run.snakeId)!;
          const scenario = visualScenario(profile, curriculumRef.current);
          return { snakeId: run.snakeId, scenario, state: createGame({ mode: gameModeFor(scenario), snakes: [profileToDefinition(profile)], seed: Date.now() + index * 1009 }) };
        }
        return { snakeId: run.snakeId, scenario: run.scenario, state: result.state };
      });
      ai.observe(experiences, contexts);
      if (Object.keys(outcomes).length) ai.finishEpisodes(outcomes);
      runsRef.current = nextRuns;
      setRuns(nextRuns);
      setLastActions((current) => ({ ...current, ...labels }));
      timer = window.setTimeout(tick, visualSpeed);
    };
    void tick();
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [visualRunning, fastRunning, visualSpeed, ai.ready, ai.act]);

  const rebuild = (nextProfiles: PersistedProfile[], nextParameters = hyperRef.current) => {
    ai.stop();
    setVisualRunning(false); setFastRunning(false);
    const nextRuns = makeRuns(nextProfiles, curriculumRef.current);
    profilesRef.current = nextProfiles; runsRef.current = nextRuns; hyperRef.current = nextParameters;
    setProfiles(nextProfiles); setRuns(nextRuns); setHyperparameters(nextParameters); setLastActions({});
    void persist(nextProfiles, nextParameters);
    ai.initialize(activeProfiles(nextProfiles), nextParameters, curriculumRef.current);
  };

  const handleAdd = (event: FormEvent) => {
    event.preventDefault();
    const result = addSnake(profilesRef.current, { name: newName, color: newColor }, hyperRef.current);
    if (result.error) { setRosterError(result.error); return; }
    rebuild(result.profiles); setNewName(""); setRosterError("");
  };
  const mutateRoster = (result: ReturnType<typeof archiveSnake>) => {
    if (result.error) { setRosterError(result.error); return; }
    rebuild(result.profiles); setRosterError("");
  };
  const resetBrains = (snakeId?: string) => {
    const ids = activeProfiles(profilesRef.current).filter((entry) => !snakeId || entry.snakeId === snakeId).map((entry) => entry.snakeId);
    ai.stop(); setFastRunning(false); setVisualRunning(false); ai.reset(ids);
    const nextRuns = makeRuns(profilesRef.current, curriculumRef.current); runsRef.current = nextRuns; setRuns(nextRuns);
  };

  const startFast = () => {
    if (fastRunning) { ai.stop(); setFastRunning(false); return; }
    setVisualRunning(false); setFastRunning(true);
    ai.startFast(activeProfiles(profilesRef.current).map(profileToDefinition));
  };

  const loadTuningDraft = (target: string) => {
    setTuningTarget(target);
    const profile = profilesRef.current.find((entry) => entry.snakeId === target);
    setDraftParameters(profile ? effectiveHyperparameters(hyperRef.current, profile.hyperparameterOverrides) : hyperRef.current);
  };
  const applyTuning = () => {
    const validated = normalizeHyperparameters(draftParameters);
    let nextProfiles = profilesRef.current;
    let nextGlobal = hyperRef.current;
    if (tuningTarget === "global") nextGlobal = validated;
    else nextProfiles = nextProfiles.map((profile) => profile.snakeId === tuningTarget ? { ...profile, hyperparameterOverrides: { ...validated } } : profile);
    setDraftParameters(validated);
    hyperRef.current = nextGlobal; profilesRef.current = nextProfiles;
    setHyperparameters(nextGlobal); setProfiles(nextProfiles); setVisualRunning(false); setFastRunning(false);
    ai.stop(); ai.updateParameters(activeProfiles(nextProfiles), nextGlobal); void persist(nextProfiles, nextGlobal);
  };
  const clearTuningOverride = () => {
    if (tuningTarget === "global") return;
    const nextProfiles = profilesRef.current.map((profile) => profile.snakeId === tuningTarget
      ? { ...profile, hyperparameterOverrides: undefined } : profile);
    profilesRef.current = nextProfiles;
    setProfiles(nextProfiles);
    setDraftParameters(hyperRef.current);
    ai.updateParameters(activeProfiles(nextProfiles), hyperRef.current);
    void persist(nextProfiles, hyperRef.current);
  };

  const exportBrain = async (snakeId: string) => {
    try {
      const bundle = await ai.exportBundle(snakeId);
      const blob = new Blob([JSON.stringify(bundle)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${snakeId}-v3.nsbrain.json`;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      setStorageMessage(error instanceof Error ? error.message : "Could not export this brain.");
    }
  };

  const importBrain = (snakeId: string) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json,.nsbrain.json,application/json";
    input.onchange = () => {
      const file = input.files?.[0];
      if (!file) return;
      void file.text()
        .then((raw) => ai.importBundle(snakeId, JSON.parse(raw) as BrainBundle))
        .then(() => setStorageMessage("Validated brain bundle imported and saved."))
        .catch((error) => setStorageMessage(error instanceof Error ? error.message : "Could not import this brain."));
    };
    input.click();
  };

  const active = activeProfiles(profiles);
  const archived = archivedProfiles(profiles);
  const totalGenerations = active.reduce((sum, profile) => sum + profile.brain.generation, 0);
  const benchmarked = active.filter((profile) => profile.brain.lastBenchmark);
  const averageWinRate = benchmarked.length
    ? benchmarked.reduce((sum, profile) => sum + (profile.brain.lastBenchmark?.winRate ?? 0), 0) / benchmarked.length
    : null;
  const episodesPerSecond = ai.progress.elapsedMs ? Math.round(ai.progress.totalEpisodes / (ai.progress.elapsedMs / 1000)) : 0;

  if (loading) return <main className="page-shell status-screen"><h1>Loading your snake roster…</h1><p>Opening IndexedDB and preparing local AI profiles.</p></main>;

  return (
    <main className="page-shell training-page">
      <section className="hero-row training-hero"><div><span className="eyebrow">Deep reinforcement learning laboratory</span><h1>BUILD BETTER<br /><em>BRAINS.</em></h1></div><p>Train every independent DQN together, tune its learning strategy, or average the active squad with Hive Mind Sync.</p></section>
      {(storageMessage || ai.error) && <div className="system-banner" role="alert">{ai.error || storageMessage}</div>}

      <section className="roster-manager glass-card">
        <div className="roster-manager-heading"><div><span className="eyebrow">Roster manager</span><h2>Add or restore snakes</h2></div><span className="roster-capacity"><b>{active.length}</b> / {MAX_ACTIVE_SNAKES} ACTIVE</span></div>
        <form className="add-snake-form" onSubmit={handleAdd}>
          <label><span>Snake name</span><input value={newName} onChange={(event) => setNewName(event.target.value)} minLength={2} maxLength={24} placeholder="e.g. Arctic Byte" disabled={active.length >= MAX_ACTIVE_SNAKES} /></label>
          <label className="color-picker-field"><span>Snake color</span><div><input type="color" value={newColor} onChange={(event) => setNewColor(event.target.value)} /><code>{newColor.toUpperCase()}</code></div></label>
          <button className="primary-button add-snake-button" disabled={active.length >= MAX_ACTIVE_SNAKES}>Add snake <b>+</b></button>
        </form>
        {rosterError && <p className="roster-error" role="alert">{rosterError}</p>}
        {archived.length > 0 && <div className="archive-shelf"><span>ARCHIVED NEURAL NETWORKS</span><div>{archived.map((profile) => <button key={profile.snakeId} onClick={() => mutateRoster(restoreSnake(profilesRef.current, profile.snakeId))} disabled={active.length >= MAX_ACTIVE_SNAKES} style={{ "--snake-color": profile.color } as React.CSSProperties}><i /><strong>{profile.name}</strong><small>GEN {profile.brain.generation.toLocaleString()}</small><b>RESTORE</b></button>)}</div></div>}
      </section>

      <section className="all-training-controls glass-card">
        <div className="control-copy"><span className="eyebrow">Squad controls</span><h2>{ai.ready ? `DQN ready · ${ai.backend.toUpperCase()}` : "Loading neural networks…"}</h2></div>
        <div className="training-status-strip"><span><small>AGENTS</small><strong>{String(active.length).padStart(2, "0")}</strong></span><span><small>TOTAL GENERATIONS</small><strong>{totalGenerations.toLocaleString()}</strong></span><span><small>BENCHMARK WIN RATE</small><strong>{averageWinRate === null ? "—" : `${Math.round(averageWinRate * 100)}%`}</strong></span><span><small>SESSION EPISODES</small><strong>{ai.progress.totalEpisodes.toLocaleString()}</strong></span></div>
        <div className="control-actions lineup-actions">
          <button className={`training-button visual ${visualRunning ? "active" : ""}`} disabled={!ai.ready || fastRunning} onClick={() => setVisualRunning((value) => !value)}><span>▶</span><div><strong>{visualRunning ? "Pause all" : "Train all visually"}</strong><small>{active.length} live boards</small></div></button>
          <button className={`training-button fast ${fastRunning ? "active" : ""}`} disabled={!ai.ready} onClick={startFast}><span>⚡</span><div><strong>{fastRunning ? "Stop fast train" : "Fast train all"}</strong><small>Persistent scenario curriculum</small></div></button>
          <button className="training-button reset" disabled={!ai.ready} onClick={() => resetBrains()}><span>↻</span><div><strong>Reset active brains</strong><small>Career records stay intact</small></div></button>
        </div>
        <div className="speed-row lineup-speed"><label>Visual tick speed <input type="range" min="24" max="160" value={184 - visualSpeed} onChange={(event) => setVisualSpeed(184 - Number(event.target.value))} disabled={fastRunning} /></label><span>{visualSpeed}ms</span></div>
        <div className="ai-toolbar"><button onClick={() => { setTuningOpen((value) => !value); loadTuningDraft(tuningTarget); }}>Tune hyperparameters</button><button onClick={() => setHiveConfirmOpen(true)} disabled={!ai.ready}>Sync Hive Mind</button><button onClick={ai.undoHive} disabled={!ai.canUndoHive}>Undo Hive Sync</button><button className="evaluate-button" onClick={() => ai.evaluate(40)} disabled={!ai.ready || fastRunning}>Run evaluation</button></div>
      </section>

      {hiveConfirmOpen && <div className="confirm-overlay" role="dialog" aria-modal="true" aria-labelledby="hive-confirm-title"><div className="confirm-card glass-card"><span className="eyebrow">Federated learning checkpoint</span><h2 id="hive-confirm-title">Average all active brains?</h2><p>Each current model will be saved as a rollback checkpoint. Weights will be averaged, while replay memories and career statistics remain individual.</p><div><button onClick={() => setHiveConfirmOpen(false)}>Cancel</button><button className="primary-button" onClick={() => { setHiveConfirmOpen(false); setVisualRunning(false); setFastRunning(false); ai.stop(); ai.hiveSync(); }}>Create checkpoint &amp; sync</button></div></div></div>}

      {tuningOpen && <aside className="hyperparameter-panel glass-card">
        <header><div><span className="eyebrow">DQN tuning dashboard</span><h2>Learning controls</h2></div><button onClick={() => setTuningOpen(false)}>×</button></header>
        <label className="tuning-target">Apply to<select value={tuningTarget} onChange={(event) => loadTuningDraft(event.target.value)}><option value="global">Global defaults</option>{active.map((profile) => <option key={profile.snakeId} value={profile.snakeId}>{profile.name} override</option>)}</select></label>
        <div className="parameter-grid">{PARAMETER_FIELDS.map((field) => <label key={field.key}><span>{field.label}</span><input type="number" min={field.min} max={field.max} step={field.step} value={draftParameters[field.key]} onChange={(event) => setDraftParameters((current) => ({ ...current, [field.key]: Number(event.target.value) }))} /></label>)}</div>
        <footer>{tuningTarget !== "global" && <button onClick={clearTuningOverride}>Use global defaults</button>}<button onClick={() => setDraftParameters(DEFAULT_HYPERPARAMETERS)}>Restore defaults</button><button className="primary-button" onClick={applyTuning}>Apply settings</button></footer>
      </aside>}

      <section className="snake-training-lineup" aria-label="All active snake training arenas">{active.map((profile) => {
        const run = runs.find((entry) => entry.snakeId === profile.snakeId); if (!run) return null;
        const snake = run.state.snakes[0];
        return <article className="snake-training-box glass-card" key={profile.snakeId} style={{ "--snake-color": profile.color } as React.CSSProperties}>
          <header><span className="brain-avatar">⌁</span><div><span className="eyebrow">Independent PER DQN</span><h2>{profile.name}</h2></div><span className={`mode-badge ${visualRunning || fastRunning ? "active" : ""}`}><i /> {fastRunning ? "TURBO" : visualRunning ? run.scenario.toUpperCase() : "STANDBY"}</span><div className="snake-card-actions"><button title="Export brain" onClick={() => void exportBrain(profile.snakeId)}>⇩</button><button title="Import brain" onClick={() => importBrain(profile.snakeId)}>⇧</button><button className="mini-reset" onClick={() => resetBrains(profile.snakeId)}>↻</button><button className="mini-archive" onClick={() => mutateRoster(archiveSnake(profilesRef.current, profile.snakeId))} disabled={active.length <= MIN_ACTIVE_SNAKES}>−</button></div></header>
          <div className="snake-training-stage"><ArenaCanvas state={run.state} compact /></div>
          <div className="snake-training-readout"><span><small>SCORE</small><strong>{snake?.score ?? 0}</strong></span><span><small>LOSS</small><strong>{profile.brain.lastLoss?.toFixed(3) ?? "—"}</strong></span><span><small>ACTION</small><strong>{lastActions[profile.snakeId] ?? "—"}</strong></span><span><small>GEN</small><strong>{profile.brain.generation.toLocaleString()}</strong></span></div>
          <footer><span>EPSILON <b>{profile.brain.epsilon.toFixed(3)}</b></span><div><i style={{ width: `${Math.min(100, profile.brain.environmentSteps / 2500)}%` }} /></div><span><b>{profile.brain.powerUpOpportunities ? Math.round(profile.brain.powerUpsClaimed / profile.brain.powerUpOpportunities * 100) : 0}%</b> POWER CLAIM</span></footer>
        </article>;
      })}</section>

      <section className="training-bottom-row"><div className={`turbo-card glass-card ${fastRunning ? "running" : ""}`}><span>FAST TRAIN THROUGHPUT</span><strong>{episodesPerSecond.toLocaleString()}</strong><small>EPISODES / SECOND · {ai.progress.scenario.toUpperCase()}</small><div><i /></div><p>{ai.progress.totalSteps.toLocaleString()} environment steps this session</p></div><div className="algorithm-card glass-card"><span className="eyebrow">What each brain observes</span><h3>228 inputs → 128 → 64 → 3 actions</h3><p>Sixteen rays plus arena phases, safe-zone timing, objectives, projected hazards, rare food, maps, bounty leaders, opponent profiles, and rematch context.</p></div></section>
      {ai.evalReport && <section className="eval-panel glass-card"><div className="panel-heading"><div><span className="eyebrow">Greedy head-to-head</span><h2>Evaluation report</h2></div><span>{ai.evalReport.runs} RUNS</span></div><div className="eval-list">{ai.evalReport.snakes.slice().sort((a, b) => b.wins - a.wins).map((snake, index) => { const profile = profiles.find((entry) => entry.snakeId === snake.id); return <div className="eval-row" key={snake.id}><span className="rank">{String(index + 1).padStart(2, "0")}</span><span className="snake-swatch" style={{ "--snake-color": profile?.color ?? "#68f7c1" } as React.CSSProperties} /><strong>{profile?.name ?? snake.id}</strong><span><b>{snake.wins}</b> W</span><span>{snake.avgScore.toFixed(1)} pts</span><span>{snake.powerUpsClaimed} pwr</span><span>{snake.foodEaten} food</span></div>; })}</div></section>}
    </main>
  );
}
