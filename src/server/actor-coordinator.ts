import { credentials, status } from "@grpc/grpc-js";
import { loadPackageDefinition } from "@grpc/grpc-js";
import { loadSync } from "@grpc/proto-loader";
import { resolve } from "node:path";
import { Worker } from "node:worker_threads";
import type { WirePolicy } from "./policy";

const target = process.env.LEARNER_GRPC_TARGET ?? "trainer:50051";
const capacity = Math.max(1, Math.min(16, Number(process.env.ACTOR_COUNT ?? 8)));
const definition = loadSync(resolve(process.cwd(), "proto/training.proto"), { keepCase: false, longs: Number, defaults: true, oneofs: true });
const grpcPackage = loadPackageDefinition(definition) as any;
const Client = grpcPackage.neon.training.v1.TrainingActors;
let workers: Worker[] = [];
let stream: any;
let sequence = 0;
let currentPolicyVersion = 0;
const pending = new Map<number, any>();
const groupId = `local-${process.pid}`;
let flowSignals: Int32Array[] = [];
let reconnectTimer: NodeJS.Timeout | null = null;
let generation = 0;
let lastLearnerWaitLog = 0;

function expectedDisconnect(error: Error & { code?: number }): boolean {
  if (error.code === status.UNAVAILABLE) return true;
  return error.code === status.INTERNAL && /RST_STREAM|Session closed/i.test(error.message);
}

function normalizePolicy(value: any): WirePolicy {
  return {
    snakeId: value.snakeId,
    version: Number(value.version),
    epsilon: value.epsilon,
    environmentSteps: Number(value.environmentSteps),
    scenarioSteps: {
      survival: Number(value.survivalSteps), powerup: Number(value.powerupSteps), safezone: Number(value.safeZoneSteps),
      hazard: Number(value.hazardSteps), objective: Number(value.objectiveSteps), battle: Number(value.battleSteps), series: Number(value.seriesSteps),
    },
    tensors: value.tensors.map((tensor: any) => ({ name: tensor.name, shape: tensor.shape.map(Number), values: tensor.values })),
  };
}

function normalizeRoster(value: any): Array<{ id: string; name: string; color: string; accent: string }> {
  const roster = Array.isArray(value) ? value : [];
  const normalized = roster.map((entry: any) => ({
    id: String(entry.snakeId ?? entry.id), name: String(entry.name ?? entry.snakeId ?? entry.id),
    color: String(entry.color ?? "#68f7c1"), accent: String(entry.accent ?? entry.color ?? "#d7fff1"),
  })).filter((entry: any) => entry.id);
  return normalized.length >= 2 ? normalized : [];
}

function setFlow(enabled: boolean): void {
  for (const signal of flowSignals) {
    Atomics.store(signal, 0, enabled ? 1 : 0);
    if (enabled) Atomics.notify(signal, 0);
  }
  workers.forEach((worker) => worker.postMessage({ type: "flow", enabled }));
}

function stopWorkers(): void {
  setFlow(true);
  workers.forEach((worker) => worker.postMessage({ type: "stop" }));
  workers.forEach((worker) => void worker.terminate());
  workers = [];
  flowSignals = [];
  pending.clear();
}

function startWorkers(start: any): void {
  stopWorkers();
  const count = Math.min(capacity, Number(start.actorCount));
  const policies = start.policies.map(normalizePolicy);
  currentPolicyVersion = Math.max(...policies.map((policy: WirePolicy) => policy.version));
  for (let index = 0; index < count; index += 1) {
    const flowBuffer = new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT);
    const flowSignal = new Int32Array(flowBuffer);
    Atomics.store(flowSignal, 0, 1);
    flowSignals.push(flowSignal);
    const worker = new Worker(resolve(__dirname, "actor-worker.js"), { workerData: { actorIndex: index, flowBuffer } });
    worker.on("message", (message) => {
      if (message.type === "error") { console.error(`Actor ${index}: ${message.message}`); return; }
      if (message.type !== "batch" || !message.transitions.length) return;
      const currentSequence = ++sequence;
      const payload = { batch: { actorId: `${groupId}:${message.actorId}`, sequence: currentSequence, policyVersion: currentPolicyVersion, transitions: message.transitions } };
      pending.set(currentSequence, payload);
      stream.write(payload);
      if (pending.size >= 8) setFlow(false);
    });
    worker.on("error", (error) => console.error(`Actor ${index} failed: ${error.message}`));
    worker.postMessage({
      type: "start",
      experimentId: start.experimentId,
      seed: Number(start.seed),
      policies,
      trainingSpecVersion: Number(start.trainingSpecVersion || 3),
      observationSize: Number(start.observationSize || 228),
      roster: normalizeRoster(start.roster),
      battleSize: Number(start.battleSize || 4),
    });
    workers.push(worker);
  }
}

function connect(): void {
  const connectionGeneration = ++generation;
  reconnectTimer = null;
  const client = new Client(target, credentials.createInsecure());
  const reconnect = () => {
    if (connectionGeneration !== generation || reconnectTimer) return;
    stopWorkers();
    reconnectTimer = setTimeout(connect, 2_000);
  };
  client.waitForReady(Date.now() + 2_000, (readyError?: Error) => {
    if (connectionGeneration !== generation) return;
    if (readyError) {
      const now = Date.now();
      if (now - lastLearnerWaitLog > 30_000) {
        console.warn(`Learner is not accepting jobs at ${target}; this is normal while training is paused or idle. Retrying in 2 seconds.`);
        lastLearnerWaitLog = now;
      }
      reconnectTimer = setTimeout(connect, 2_000);
      return;
    }
    stream = client.Connect();
    stream.on("data", (message: any) => {
      if (message.start) startWorkers(message.start);
      if (message.policy) {
        const policy = normalizePolicy(message.policy);
        currentPolicyVersion = Math.max(currentPolicyVersion, policy.version);
        workers.forEach((worker) => worker.postMessage({ type: "policy", policy }));
      }
      if (message.ack) {
        pending.delete(Number(message.ack.sequence));
        if (pending.size <= 4) setFlow(true);
      }
      if (message.stop) stopWorkers();
    });
    stream.on("error", (error: Error & { code?: number }) => {
      if (!expectedDisconnect(error)) console.error(error.message);
      reconnect();
    });
    stream.on("end", reconnect);
    stream.write({ hello: { groupId, capacity } });
  });
}

connect();
