"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { Brain, CurriculumConfig, EvalReport, Experience, Hyperparameters, PersistedProfile, RelativeAction, SnakeDefinition, TrainingScenario } from "./types";
import type { WorkerRequest, WorkerResponse } from "./worker-protocol";
import type { BrainBundle } from "./model-bundle";

export type AiProgress = { scenario: TrainingScenario; totalEpisodes: number; totalSteps: number; elapsedMs: number };
type PendingRequest<T> = { resolve: (value: T) => void; reject: (reason?: unknown) => void };

export function useAiWorker() {
  const workerRef = useRef<Worker | null>(null);
  const requestId = useRef(0);
  const pending = useRef(new Map<number, (actions: Record<string, RelativeAction>) => void>());
  const pendingBundles = useRef(new Map<number, PendingRequest<BrainBundle>>());
  const pendingImports = useRef(new Map<number, PendingRequest<void>>());
  const pendingProduction = useRef(new Map<number, PendingRequest<void>>());
  const [workerAvailable, setWorkerAvailable] = useState(false);
  const [ready, setReady] = useState(false);
  const [backend, setBackend] = useState("");
  const [brains, setBrains] = useState<Brain[]>([]);
  const [progress, setProgress] = useState<AiProgress>({ scenario: "survival", totalEpisodes: 0, totalSteps: 0, elapsedMs: 0 });
  const [error, setError] = useState("");
  const [saveCounter, setSaveCounter] = useState(0);
  const [canUndoHive, setCanUndoHive] = useState(false);
  const [evalReport, setEvalReport] = useState<EvalReport | null>(null);

  useEffect(() => {
    const worker = new Worker(new URL("../workers/trainer.worker.ts", import.meta.url));
    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const message = event.data;
      if ("brains" in message) setBrains(message.brains);
      if (message.type === "ready") { setReady(true); setBackend(message.backend); }
      if (message.type === "actions") { pending.current.get(message.requestId)?.(message.actions); pending.current.delete(message.requestId); }
      if (message.type === "progress") setProgress({ scenario: message.scenario, totalEpisodes: message.totalEpisodes, totalSteps: message.totalSteps, elapsedMs: message.elapsedMs });
      if (message.type === "saved") setSaveCounter((value) => value + 1);
      if (message.type === "stopped") setSaveCounter((value) => value + 1);
      if (message.type === "hiveComplete") { setCanUndoHive(message.canUndo); setSaveCounter((value) => value + 1); }
      if (message.type === "evalReport") setEvalReport(message.report);
      if (message.type === "bundle") { pendingBundles.current.get(message.requestId)?.resolve(message.bundle); pendingBundles.current.delete(message.requestId); }
      if (message.type === "bundleImported") { pendingImports.current.get(message.requestId)?.resolve(); pendingImports.current.delete(message.requestId); setSaveCounter((value) => value + 1); }
      if (message.type === "productionBundlesLoaded") { pendingProduction.current.get(message.requestId)?.resolve(); pendingProduction.current.delete(message.requestId); }
      if (message.type === "error") {
        setError(message.message);
        if (message.requestId !== undefined) {
          pendingBundles.current.get(message.requestId)?.reject(new Error(message.message));
          pendingImports.current.get(message.requestId)?.reject(new Error(message.message));
          pendingProduction.current.get(message.requestId)?.reject(new Error(message.message));
          pendingBundles.current.delete(message.requestId);
          pendingImports.current.delete(message.requestId);
          pendingProduction.current.delete(message.requestId);
        }
      }
    };
    worker.onerror = (event) => setError(event.message || "The AI worker stopped unexpectedly.");
    workerRef.current = worker;
    setWorkerAvailable(true);
    return () => {
      pending.current.clear();
      pendingBundles.current.clear();
      pendingImports.current.clear();
      pendingProduction.current.clear();
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  const send = useCallback((message: WorkerRequest) => workerRef.current?.postMessage(message), []);
  const initialize = useCallback((profiles: PersistedProfile[], hyperparameters: Hyperparameters, curriculum: CurriculumConfig) => {
    setReady(false); setError("");
    send({ type: "init", profiles, hyperparameters, curriculum });
  }, [send]);
  const act = useCallback((observations: Record<string, { data: number[], safeActions: RelativeAction[] }>, explore: boolean) => new Promise<Record<string, RelativeAction>>((resolve) => {
    const id = ++requestId.current;
    pending.current.set(id, resolve);
    send({ type: "act", requestId: id, observations, explore });
  }), [send]);
  const exportBundle = useCallback((snakeId: string) => new Promise<BrainBundle>((resolve, reject) => {
    const id = ++requestId.current;
    pendingBundles.current.set(id, { resolve, reject });
    send({ type: "exportBundle", requestId: id, snakeId });
  }), [send]);
  const importBundle = useCallback((snakeId: string, bundle: BrainBundle) => new Promise<void>((resolve, reject) => {
    const id = ++requestId.current;
    pendingImports.current.set(id, { resolve, reject });
    send({ type: "importBundle", requestId: id, snakeId, bundle });
  }), [send]);
  const useProductionBundles = useCallback((bundles: Record<string, BrainBundle>) => new Promise<void>((resolve, reject) => {
    const id = ++requestId.current;
    pendingProduction.current.set(id, { resolve, reject });
    send({ type: "useProductionBundles", requestId: id, bundles });
  }), [send]);

  return {
    workerAvailable, ready, backend, brains, progress, error, saveCounter, canUndoHive, evalReport,
    initialize, act,
    observe: (experiences: Record<string, Experience>, contexts?: Extract<WorkerRequest, { type: "observe" }>["contexts"]) => send({ type: "observe", experiences, contexts }),
    finishEpisodes: (outcomes: Record<string, { score: number; food: number; won: boolean }>) => send({ type: "episode", outcomes }),
    startFast: (snakes: SnakeDefinition[]) => send({ type: "startFast", snakes }),
    evaluate: (runs: number) => { setEvalReport(null); send({ type: "evaluate", runs }); },
    stop: () => send({ type: "stop" }),
    checkpoint: () => send({ type: "checkpoint" }),
    updateParameters: (profiles: PersistedProfile[], hyperparameters: Hyperparameters) => send({ type: "updateParameters", profiles, hyperparameters }),
    reset: (snakeIds: string[]) => send({ type: "reset", snakeIds }),
    hiveSync: () => send({ type: "hiveSync" }),
    undoHive: () => send({ type: "undoHive" }),
    trainAfterMatch: (steps = 32) => send({ type: "trainAfterMatch", steps }),
    exportBundle,
    importBundle,
    useProductionBundles,
  };
}
