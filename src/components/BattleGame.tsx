"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { encodeObservation } from "@/game/ai";
import { POWER_UP_KINDS, POWER_UP_META } from "@/game/constants";
import { createGame, stepGame, isDangerAt, relativeDirection } from "@/game/engine";
import { activeProfiles, defaultRoster, loadAppData, mergeBrains, profileToDefinition, recordMatch, saveProfiles } from "@/game/storage";
import type { CurriculumConfig, GameState, Hyperparameters, PersistedProfile } from "@/game/types";
import { useAiWorker } from "@/game/useAiWorker";
import { loadProductionModels } from "@/game/server-models";
import { ArenaCanvas } from "./ArenaCanvas";
import { LiveLeaderboard } from "./LiveLeaderboard";

export function BattleGame() {
  const initial = defaultRoster();
  const [profiles, setProfiles] = useState<PersistedProfile[]>(initial.profiles);
  const [parameters, setParameters] = useState<Hyperparameters>(initial.hyperparameters);
  const [curriculum, setCurriculum] = useState<CurriculumConfig>(initial.curriculum);
  const [game, setGame] = useState<GameState>(() => createGame({ seed: 4172026, snakes: initial.profiles.map(profileToDefinition) }));
  const [loading, setLoading] = useState(true);
  const [paused, setPaused] = useState(false);
  const [speed, setSpeed] = useState(92);
  const [postMatchTraining, setPostMatchTraining] = useState(false);
  const [storageError, setStorageError] = useState("");
  const [productionRelease, setProductionRelease] = useState<string | null>(null);
  const gameRef = useRef(game);
  const profilesRef = useRef(profiles);
  const parametersRef = useRef(parameters);
  const curriculumRef = useRef(curriculum);
  const initializedRef = useRef(false);
  const recordedRef = useRef(false);
  const productionAttemptedRef = useRef(false);
  const ai = useAiWorker();

  useEffect(() => {
    let active = true;
    void loadAppData().then((result) => {
      if (!active) return;
      const roster = result.roster;
      const fresh = createGame({ snakes: activeProfiles(roster.profiles).map(profileToDefinition) });
      profilesRef.current = roster.profiles;
      parametersRef.current = roster.hyperparameters;
      curriculumRef.current = roster.curriculum;
      gameRef.current = fresh;
      setProfiles(roster.profiles); setParameters(roster.hyperparameters); setCurriculum(roster.curriculum); setGame(fresh);
      setStorageError(result.error ?? ""); setLoading(false);
    });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (loading || !ai.workerAvailable || initializedRef.current) return;
    initializedRef.current = true;
    ai.initialize(activeProfiles(profilesRef.current), parametersRef.current, curriculumRef.current);
  }, [loading, ai.workerAvailable, ai.initialize]);

  useEffect(() => {
    if (!ai.brains.length) return;
    const merged = mergeBrains(profilesRef.current, ai.brains);
    profilesRef.current = merged;
    setProfiles(merged);
  }, [ai.brains]);

  const persist = useCallback(async () => {
    try { await saveProfiles(profilesRef.current, parametersRef.current, curriculumRef.current); setStorageError(""); }
    catch { setStorageError("Battle results are in memory, but IndexedDB could not save them."); }
  }, []);

  useEffect(() => {
    if (ai.ready) void persist();
  }, [ai.ready, persist]);

  useEffect(() => {
    if (!ai.ready || productionAttemptedRef.current) return;
    productionAttemptedRef.current = true;
    void loadProductionModels().then(async (production) => {
      if (!production) return;
      const activeIds = new Set(activeProfiles(profilesRef.current).map((profile) => profile.snakeId));
      const bundles = Object.fromEntries(Object.entries(production.bundles).filter(([snakeId]) => activeIds.has(snakeId)));
      if (!Object.keys(bundles).length) return;
      await ai.useProductionBundles(bundles);
      setProductionRelease(production.manifest.releaseId);
    }).catch(() => { /* Local browser brains remain the safe fallback. */ });
  }, [ai.ready, ai.useProductionBundles]);

  const restart = useCallback(() => {
    if (postMatchTraining) return;
    const fresh = createGame({ snakes: activeProfiles(profilesRef.current).map(profileToDefinition) });
    gameRef.current = fresh; recordedRef.current = false; setPaused(false); setGame(fresh);
  }, [postMatchTraining]);

  useEffect(() => {
    if (!ai.ready || loading || paused || postMatchTraining || game.status === "finished") return;
    let cancelled = false;
    let timer = 0;
    const tick = async () => {
      const current = gameRef.current;
      if (current.status === "finished") return;
      const observations: Record<string, { data: number[], safeActions: import("@/game/types").RelativeAction[] }> = {};
      current.snakes.filter((snake) => snake.alive).forEach((snake) => {
        const data = Array.from(encodeObservation(current, snake));
        const lookahead = snake.buffs.hasteUntil > current.tick || snake.buffs.visionUntil > current.tick ? 2 : 1;
        const safeActions = ([0, 1, 2] as import("@/game/types").RelativeAction[]).filter(a => !isDangerAt(current, snake, relativeDirection(snake.direction, a), lookahead));
        observations[snake.id] = { data, safeActions };
      });
      const actions = await ai.act(observations, false);
      if (cancelled) return;
      const result = stepGame(current, actions);
      const experiences: Parameters<typeof ai.observe>[0] = {};
      const contexts: NonNullable<Parameters<typeof ai.observe>[1]> = {};
      for (const snake of current.snakes.filter((entry) => entry.alive)) {
        const nextSnake = result.state.snakes.find((entry) => entry.id === snake.id);
        if (!nextSnake) continue;
        experiences[snake.id] = {
          state: observations[snake.id].data, action: actions[snake.id] ?? 0, reward: result.rewards[snake.id] ?? 0,
          nextState: Array.from(encodeObservation(result.state, nextSnake)), terminal: !nextSnake.alive || result.state.status === "finished",
          rewardBreakdown: result.rewardBreakdowns[snake.id],
          deathCause: !nextSnake.alive ? (nextSnake.deathReason?.includes("zone") ? "zone" : nextSnake.deathReason?.includes("hazard") ? "hazard" : nextSnake.deathReason?.includes("wall") ? "wall" : nextSnake.deathReason?.includes("obstacle") ? "obstacle" : nextSnake.deathReason?.includes("head-on") ? "headOn" : nextSnake.deathReason?.includes("snake") ? "snakeBody" : "other") : undefined,
        };
        const claimed = nextSnake.powerUps > snake.powerUps;
        contexts[snake.id] = {
          scenario: "battle",
          opportunity: !current.powerUp && Boolean(result.state.powerUp),
          claimed,
          approachMiss: Boolean(current.powerUp) && !result.state.powerUp && !claimed,
        };
      }
      ai.observe(experiences, contexts);
      gameRef.current = result.state; setGame(result.state);
      if (result.state.status === "running") timer = window.setTimeout(tick, speed);
    };
    void tick();
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [ai.ready, loading, paused, postMatchTraining, speed, game.status, ai.act]);

  useEffect(() => {
    if (game.status !== "finished" || recordedRef.current) return;
    recordedRef.current = true;
    setPostMatchTraining(true);
    const outcomes = Object.fromEntries(game.snakes.map((snake) => [snake.id, { score: snake.score, food: snake.foodEaten, won: game.winnerId === snake.id }]));
    ai.finishEpisodes(outcomes);
    profilesRef.current = recordMatch(profilesRef.current, game);
    setProfiles(profilesRef.current);
    ai.trainAfterMatch(32);
  }, [game, ai.finishEpisodes, ai.trainAfterMatch]);

  useEffect(() => {
    if (!postMatchTraining || !ai.saveCounter) return;
    const merged = mergeBrains(profilesRef.current, ai.brains);
    profilesRef.current = merged; setProfiles(merged); setPostMatchTraining(false); void persist();
  }, [ai.saveCounter, ai.brains, postMatchTraining, persist]);

  const activeMeta = game.powerUp ? POWER_UP_META[game.powerUp.kind] : null;
  const alive = game.snakes.filter((snake) => snake.alive).length;

  if (loading || !ai.ready) return <main className="page-shell status-screen"><h1>Loading neural networks…</h1><p>{ai.error || "Opening each active snake’s local DQN before the battle begins."}</p></main>;

  return (
    <main className="page-shell battle-page">
      <section className="hero-row"><div><span className="eyebrow">Autonomous DQN combat simulation</span><h1>LAST SERPENT<br /><em>STANDING.</em></h1></div><p>{game.snakes.length} independent neural networks. {productionRelease ? `Promoted server release ${productionRelease.slice(0, 8)} is running read-only.` : "Local browser brains are active."}</p></section>
      {(storageError || ai.error) && <div className="system-banner" role="alert">{ai.error || storageError}</div>}
      <section className="battle-layout">
        <div className="arena-panel glass-card">
          <div className="arena-toolbar"><div className="round-readout"><span>{game.arena.phase.toUpperCase()} · {game.arena.mapArchetype.toUpperCase()}</span><strong>{postMatchTraining ? "UPDATING BRAINS" : game.status === "finished" ? "COMPLETE" : paused ? "PAUSED" : game.arena.objective ? "CORE ACTIVE" : game.arena.hazards.some((hazard) => hazard.activatesAt <= game.tick) ? "HAZARD ACTIVE" : "BATTLE ACTIVE"}</strong></div><div className="arena-stats"><span><b>{alive}</b> ALIVE</span><span><b>{game.tick}</b> TICK</span><span><b>{game.arena.seriesRound}/3</b> ROUND</span></div><div className="battle-controls"><button onClick={() => setPaused((value) => !value)} disabled={game.status === "finished"}>{paused ? "Resume" : "Pause"}</button><button className="icon-button" onClick={restart} disabled={postMatchTraining} title="New match">↻</button></div></div>
          <div className="arena-stage"><ArenaCanvas state={game} /><div className="arena-corners"><i /><i /><i /><i /></div>{game.status === "finished" && <div className="victory-overlay"><span>{postMatchTraining ? "CONSOLIDATING EXPERIENCE" : "BATTLE COMPLETE"}</span><h2>{game.winnerId ? game.snakes.find((snake) => snake.id === game.winnerId)?.name : "NO SURVIVORS"}</h2><p>{postMatchTraining ? "Running 32 post-match DQN updates…" : game.winnerId ? "Last snake standing" : "Mutual elimination"}</p><button className="primary-button" onClick={restart} disabled={postMatchTraining}>Launch next match <b>→</b></button></div>}{activeMeta && <div className="power-chip" style={{ "--power-color": activeMeta.color } as React.CSSProperties}><b>{activeMeta.icon}</b><span><small>POWER-UP ACTIVE</small>{activeMeta.name}</span></div>}</div>
          <div className="arena-footer"><label>Simulation speed <input type="range" min="45" max="180" value={225 - speed} onChange={(event) => setSpeed(225 - Number(event.target.value))} /></label><span>{game.arena.objective ? `CORE ${Math.max(0, game.arena.objective.expiresAt - game.tick)}T` : "CORE DORMANT"}</span><span>{game.tick >= game.arena.safeZone.telegraphAt && game.arena.safeZone.pendingInset > game.arena.safeZone.inset ? `ZONE CLOSE ${Math.max(0, game.arena.safeZone.closesAt - game.tick)}T` : "ZONE STABLE"}</span></div>
        </div>
        <div className="battle-sidebar"><LiveLeaderboard state={game} profiles={profiles} brains={ai.brains} /><div className="feed-card glass-card"><div className="panel-heading"><div><span className="eyebrow">Arena signal</span><h2>Battle feed</h2></div></div><div className="event-feed">{game.events.length ? game.events.map((event) => <div className={event.tone} key={event.id}><span>{String(event.tick).padStart(4, "0")}</span><p>{event.text}</p></div>) : <div><span>0000</span><p>Individual DQNs loaded. Match underway.</p></div>}</div></div></div>
      </section>
      <section className="power-catalogue"><div className="section-heading"><div><span className="eyebrow">Random arena drops</span><h2>Twelve ways to turn the fight</h2></div><p>Only one power-up can exist on the map at a time. Reach it before your rivals do.</p></div><div className="power-grid">{POWER_UP_KINDS.map((kind, index) => { const meta = POWER_UP_META[kind]; return <article key={kind} style={{ "--power-color": meta.color } as React.CSSProperties}><span className="power-number">{String(index + 1).padStart(2, "0")}</span><b>{meta.icon}</b><div><h3>{meta.name}</h3><p>{meta.description}</p></div></article>; })}</div></section>
    </main>
  );
}
