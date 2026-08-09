"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { loadAppData, activeProfiles, profileToDefinition } from "../game/storage";
import { SNAKES } from "../game/constants";
import type { SnakeDefinition } from "../game/types";

type Experiment = {
  id: string; status: string; requested_action: string | null; config: Record<string, any>;
  progress: Record<string, any>; error: string | null; checkpoint_key: string | null; created_at: string; started_at: string | null; finished_at: string | null;
};
type Release = { id: string; status: string; metrics: Record<string, any>; created_at: string };

const api = process.env.NEXT_PUBLIC_SERVER_TRAINING_API ?? "/server-api";
const editableExperimentFields = [
  "step_budget_per_seed",
  "seed_count",
  "master_seed",
  "actor_count",
  "replay_capacity_per_snake",
  "batch_size",
  "benchmark_matches",
] as const;

function formatApiError(payload: unknown, status: number): string {
  if (payload && typeof payload === "object" && "detail" in payload) {
    const detail = (payload as { detail?: unknown }).detail;
    if (typeof detail === "string" && detail.trim()) return detail;
    if (Array.isArray(detail)) {
      const messages = detail.map((item) => {
        if (!item || typeof item !== "object") return String(item);
        const entry = item as { loc?: unknown; msg?: unknown };
        const location = Array.isArray(entry.loc) ? entry.loc.join(".") : "request";
        return `${location}: ${typeof entry.msg === "string" ? entry.msg : JSON.stringify(item)}`;
      });
      if (messages.length) return messages.join("; ");
    }
    if (detail && typeof detail === "object") {
      try { return JSON.stringify(detail); } catch { /* fall through to the status message */ }
    }
  }
  return `Request failed (${status}).`;
}

function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  const total = Math.floor(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  return hours ? `${hours}h ${String(minutes).padStart(2, "0")}m ${String(secs).padStart(2, "0")}s` : `${minutes}m ${String(secs).padStart(2, "0")}s`;
}

function formatTimestamp(value: unknown): string {
  if (typeof value !== "string") return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

const phaseDescriptions: Record<string, string> = {
  queued: "Waiting for the local learner",
  preflight: "Preparing actors and checkpoints",
  training: "Collecting transitions and updating brains",
  benchmarking: "Evaluating this seed against the baseline",
  evaluating: "Combining seed benchmark results",
  paused: "Paused with a recoverable checkpoint",
  completed: "Finished and release created",
  failed: "Stopped because of an error",
  cancelled: "Cancelled",
};

export function ServerTraining() {
  const [authenticated, setAuthenticated] = useState<boolean | null>(null);
  const [csrf, setCsrf] = useState("");
  const [password, setPassword] = useState("");
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [releases, setReleases] = useState<Release[]>([]);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [roster, setRoster] = useState<SnakeDefinition[]>(SNAKES);
  const [draft, setDraft] = useState({ step_budget_per_seed: 1_000_000, seed_count: 5, master_seed: 42, actor_count: 8, replay_capacity_per_snake: 250_000, batch_size: 128, benchmark_matches: 200 });
  const [clock, setClock] = useState(() => Date.now());
  const prefilled = useRef(false);

  const request = useCallback(async <T,>(path: string, init: RequestInit = {}): Promise<T> => {
    const response = await fetch(`${api}${path}`, {
      ...init,
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", ...(csrf && init.method && init.method !== "GET" ? { "X-CSRF-Token": csrf } : {}), ...init.headers },
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({ detail: `Request failed (${response.status}).` }));
      throw new Error(formatApiError(payload, response.status));
    }
    return response.json() as Promise<T>;
  }, [csrf]);

  const refresh = useCallback(async () => {
    if (!authenticated) return;
    try {
      const [nextExperiments, nextReleases] = await Promise.all([request<Experiment[]>("/experiments"), request<Release[]>("/releases")]);
      setExperiments(nextExperiments); setReleases(nextReleases); setError("");
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Could not refresh local training status."); }
  }, [authenticated, request]);

  useEffect(() => {
    void fetch(`${api}/auth/me`, { credentials: "same-origin" }).then(async (response) => {
      if (!response.ok) { setAuthenticated(false); return; }
      const session = await response.json() as { csrfToken: string };
      setCsrf(session.csrfToken); setAuthenticated(true);
    }).catch(() => { setAuthenticated(false); setError("The local Docker training server is offline."); });
  }, []);

  useEffect(() => {
    if (!authenticated) return;
    void loadAppData().then((loaded) => {
      const active = activeProfiles(loaded.roster.profiles).map(profileToDefinition);
      if (active.length >= 2) setRoster(active);
    }).catch(() => setRoster(SNAKES));
    void refresh();
    const timer = window.setInterval(() => void refresh(), 2_000);
    return () => window.clearInterval(timer);
  }, [authenticated, refresh]);

  useEffect(() => {
    if (prefilled.current || !experiments.length) return;
    const source = experiments.find((entry) => entry.status === "cancelled") ?? experiments[0];
    if (source?.config) {
      // Only carry over the fields the form owns. Legacy v2 configs contain
      // fields such as training_spec_version and resume_checkpoint_key that
      // must never be sent when starting a fresh v3 experiment.
      setDraft((current) => {
        const next = { ...current };
        for (const key of editableExperimentFields) {
          const value = source.config[key];
          if (typeof value === "number" && Number.isFinite(value)) next[key] = value;
        }
        return next;
      });
    }
    prefilled.current = true;
  }, [experiments]);

  const active = experiments.find((entry) => ["queued", "preflight", "training", "evaluating", "paused"].includes(entry.status));
  useEffect(() => {
    if (!active) return;
    setClock(Date.now());
    const timer = window.setInterval(() => setClock(Date.now()), 1_000);
    return () => window.clearInterval(timer);
  }, [active?.id, active?.status, active?.progress.phase, active?.progress.timing?.phaseStartedAt]);

  const login = async (event: React.FormEvent) => {
    event.preventDefault(); setSubmitting(true); setError("");
    try {
      const session = await request<{ csrfToken: string }>("/auth/login", { method: "POST", body: JSON.stringify({ password }) });
      setCsrf(session.csrfToken); setAuthenticated(true); setPassword("");
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Login failed."); }
    finally { setSubmitting(false); }
  };

  const createExperiment = async () => {
    setSubmitting(true); setError("");
    try {
      await request("/experiments", {
        method: "POST",
        // Pin the current API contract even if an old experiment was used to
        // prefill the visible fields.
        body: JSON.stringify({ ...draft, training_spec_version: 3, roster, battle_size: Math.min(4, roster.length) }),
      });
      await refresh();
    }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Could not start training."); }
    finally { setSubmitting(false); }
  };

  const action = async (path: string) => {
    setSubmitting(true); setError("");
    try { await request(path, { method: "POST", body: "{}" }); await refresh(); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Action failed."); }
    finally { setSubmitting(false); }
  };

  const promote = async (release: Release) => {
    if (!release.metrics.eligible && !window.confirm("This release did not pass the five-seed safety gates. Promote it only for local testing?")) return;
    await action(`/releases/${release.id}/promote${release.metrics.eligible ? "" : "?override=true"}`);
  };

  const logout = async () => {
    setSubmitting(true); setError("");
    try { await request("/auth/logout", { method: "POST", body: "{}" }); setAuthenticated(false); setCsrf(""); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Could not sign out."); }
    finally { setSubmitting(false); }
  };

  if (authenticated === null) return <main className="page-shell status-screen"><h1>Connecting to local training…</h1></main>;
  if (!authenticated) return <main className="page-shell server-page"><section className="server-login glass-card"><span className="eyebrow">Private local control</span><h1>SERVER LAB</h1><p>Sign in with the administrator password created during Docker setup.</p><form onSubmit={login}><label>Administrator password<input type="password" autoComplete="current-password" value={password} onChange={(event) => setPassword(event.target.value)} minLength={12} required /></label><button className="primary-button" disabled={submitting}>{submitting ? "Signing in…" : "Open server lab"}</button></form>{error && <div className="system-banner" role="alert">{error}</div>}</section></main>;

  const legacySource = experiments.find((entry) =>
    ["cancelled", "failed"].includes(entry.status)
    && Boolean(entry.checkpoint_key)
    && Number(entry.progress.environmentSteps ?? 0) > 0
    && (entry.progress.metricsSchemaVersion === undefined || entry.progress.metricsSchemaVersion === 2),
  );
  const timing = (active?.progress.timing ?? {}) as Record<string, unknown>;
  const phase = String(active?.progress.phase ?? active?.status ?? "idle");
  const phaseStartedAt = typeof timing.phaseStartedAt === "string" ? timing.phaseStartedAt : active?.started_at;
  const phaseElapsed = phaseStartedAt ? Math.max(0, (clock - new Date(phaseStartedAt).getTime()) / 1_000) : 0;
  const benchmarkSeed = Number(timing.benchmarkSeedIndex ?? active?.progress.seedIndex ?? 0) + 1;
  const lastBenchmarkSeconds = Number(timing.lastBenchmarkSeconds ?? 0);
  return <main className="page-shell server-page">
    <section className="hero-row"><div><span className="eyebrow">Local PyTorch training service</span><h1>TRAIN<br /><em>OFFLINE.</em></h1></div><div><p>Eight local simulators feed one centralized learner. Models, checkpoints, queues, and benchmarks remain on this machine.</p><button onClick={logout} disabled={submitting}>Sign out</button></div></section>
    {error && <div className="system-banner" role="alert">{error}</div>}
    <section className="server-grid">
      <article className="glass-card server-create"><div className="panel-heading"><div><span className="eyebrow">New experiment</span><h2>Training budget</h2></div></div>
        <div className="server-fields">
          <label>Steps per seed<input type="number" min="1000" max="100000000" value={draft.step_budget_per_seed} onChange={(event) => setDraft({ ...draft, step_budget_per_seed: Number(event.target.value) })} /></label>
          <label>Independent seeds<input type="number" min="1" max="5" value={draft.seed_count} onChange={(event) => setDraft({ ...draft, seed_count: Number(event.target.value) })} /></label>
          <label>Simulation workers<input type="number" min="1" max="16" value={draft.actor_count} onChange={(event) => setDraft({ ...draft, actor_count: Number(event.target.value) })} /></label>
          <label>Replay per snake<input type="number" min="10000" max="1000000" value={draft.replay_capacity_per_snake} onChange={(event) => setDraft({ ...draft, replay_capacity_per_snake: Number(event.target.value) })} /></label>
          <label>Benchmark matches<input type="number" min="2" max="1000" value={draft.benchmark_matches} onChange={(event) => setDraft({ ...draft, benchmark_matches: Number(event.target.value) })} /></label>
        </div>
        <p className="server-note">Active roster: {roster.length} snakes ({roster.map((entry) => entry.name).join(", ")}). Every active snake receives its own learner, replay buffer, checkpoint, and evaluation bundle.</p>
        <button className="primary-button" onClick={createExperiment} disabled={submitting || Boolean(active)}>Start experiment</button>
        {active && <p className="server-note">Finish, pause, or cancel the current experiment before starting another.</p>}
      </article>
      <article className="glass-card server-status"><div className="panel-heading"><div><span className="eyebrow">Current workload</span><h2>{active ? active.status.toUpperCase() : "IDLE"}</h2></div></div>
        {active ? <><div className="server-metrics"><span><b>{Number(active.progress.environmentSteps ?? 0).toLocaleString()}</b> steps</span><span><b>{Number(active.progress.seedIndex ?? 0) + 1}/{active.config.seed_count}</b> seeds</span><span><b>v{active.progress.policyVersion ?? 1}</b> policy</span></div><div className="server-phase"><div><span className="eyebrow">Current phase</span><strong>{phase.toUpperCase()}</strong><small>{phaseDescriptions[phase] ?? "Working locally"}</small></div><div><span className="eyebrow">Phase elapsed</span><strong>{formatDuration(phaseElapsed)}</strong><small>Started {formatTimestamp(phaseStartedAt)}</small></div>{phase === "benchmarking" ? <div><span className="eyebrow">Seed benchmark</span><strong>{formatDuration(phaseElapsed)}</strong><small>Seed {benchmarkSeed} · timer resets per seed</small></div> : <div><span className="eyebrow">Last benchmark</span><strong>{lastBenchmarkSeconds ? formatDuration(lastBenchmarkSeconds) : "—"}</strong><small>{timing.lastBenchmarkSeedIndex === undefined ? "No completed seed benchmark yet" : `Seed ${Number(timing.lastBenchmarkSeedIndex) + 1}`}</small></div>}</div><p className="server-note server-timing-note">Training started {formatTimestamp(timing.trainingStartedAt)} · pauses {Number(timing.resumeCount ?? 0)} resumed · last resumed {formatTimestamp(timing.lastResumedAt)} · benchmark started {formatTimestamp(timing.benchmarkStartedAt)}</p><progress max={active.config.step_budget_per_seed} value={active.progress.environmentSteps ?? 0} /><div className="server-actions">{active.status === "paused" ? <button onClick={() => action(`/experiments/${active.id}/resume`)}>Resume</button> : <button onClick={() => action(`/experiments/${active.id}/pause`)}>Pause safely</button>}<button onClick={() => action(`/experiments/${active.id}/finish`)}>Finish now</button><button onClick={() => action(`/experiments/${active.id}/cancel`)}>Cancel</button></div><p className="server-note">Finish now benchmarks the current learner state and creates a partial candidate release{active.status === "paused" ? " from the saved checkpoint" : ""}.</p></> : <p>No local experiment is running.</p>}
      </article>
    </section>
    {legacySource && !releases.some((release) => release.status === "baseline") && <section className="glass-card server-list"><div className="panel-heading"><div><span className="eyebrow">Controlled comparison</span><h2>Complete the recoverable v2 baseline</h2></div></div><p>The preserved v2 checkpoint has {Number(legacySource.progress.environmentSteps ?? 0).toLocaleString()} of {Number(legacySource.config.step_budget_per_seed ?? 1_000_000).toLocaleString()} steps. Finish it once so every v3 seed can be compared against the same legacy policy.</p><button className="primary-button" disabled={submitting || Boolean(active)} onClick={() => action(`/legacy-baselines/${legacySource.id}/continue`)}>Complete v2 baseline</button></section>}
    <section className="glass-card server-list"><div className="panel-heading"><div><span className="eyebrow">Model registry</span><h2>Candidate and production releases</h2></div>{releases.some((release) => release.status === "production") && releases.some((release) => release.status === "archived") && <button disabled={submitting} onClick={() => action("/releases/rollback")}>Roll back</button>}</div>{releases.length ? releases.map((release) => <div className="server-row" key={release.id}><code>{release.id.slice(0, 8)}</code><strong>{release.status.toUpperCase()}</strong><span>{release.status === "baseline" ? "Controlled v2 comparison" : release.metrics.manualPromotion ? "Manual testing promotion" : release.metrics.eligible ? "Benchmark gate passed" : "Not yet eligible"}</span><Link href={`/intelligence?release=${release.id}`}>Analyze</Link>{!['production', 'baseline'].includes(release.status) && <button disabled={submitting} onClick={() => void promote(release)}>{release.metrics.eligible ? "Promote" : "Promote for testing"}</button>}</div>) : <p>No releases have been produced yet.</p>}</section>
  </main>;
}
