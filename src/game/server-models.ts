"use client";

import { createStore, get, set } from "idb-keyval";
import { validateBrainBundle, type BrainBundle } from "./model-bundle";

type ProductionBrain = { snakeId: string; sha256: string; url: string };
export type ProductionManifest = {
  releaseId: string;
  trainingSpecVersion: number;
  observationSize: number;
  observationSpecHash: string;
  createdAt: string;
  metrics: Record<string, unknown>;
  brains: ProductionBrain[];
};
type SignedManifest = { payload: string; signature: string };

const apiBase = process.env.NEXT_PUBLIC_SERVER_TRAINING_API ?? "/server-api";
const cache = () => createStore("neon-serpents-server-models", "production");
type LoadedProduction = { manifest: ProductionManifest; bundles: Record<string, BrainBundle> };

function decodeBase64(value: string): ArrayBuffer {
  const binary = atob(value);
  return Uint8Array.from(binary, (character) => character.charCodeAt(0)).buffer;
}

async function verifyManifest(envelope: SignedManifest): Promise<ProductionManifest> {
  const response = await fetch(`${apiBase}/production/public-key`, { cache: "no-store" });
  if (!response.ok) throw new Error("Production signing key is unavailable.");
  const { publicKey } = await response.json() as { publicKey: string };
  const key = await crypto.subtle.importKey("raw", decodeBase64(publicKey), { name: "Ed25519" }, false, ["verify"]);
  const payload = decodeBase64(envelope.payload);
  const valid = await crypto.subtle.verify("Ed25519", key, decodeBase64(envelope.signature), payload);
  if (!valid) throw new Error("Production model manifest signature is invalid.");
  return JSON.parse(new TextDecoder().decode(payload)) as ProductionManifest;
}

async function sha256(value: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function fetchCurrent(): Promise<{ manifest: ProductionManifest; bundles: Record<string, BrainBundle> }> {
  const manifestResponse = await fetch(`${apiBase}/production/manifest`, { cache: "no-store" });
  if (!manifestResponse.ok) throw new Error("No promoted server models are available.");
  const envelope = await manifestResponse.json() as SignedManifest;
  const manifest = await verifyManifest(envelope);
  const bundles: Record<string, BrainBundle> = {};
  for (const brain of manifest.brains) {
    const response = await fetch(brain.url, { cache: "force-cache" });
    if (!response.ok) throw new Error(`Production brain ${brain.snakeId} could not be downloaded.`);
    const raw = await response.text();
    if (await sha256(raw) !== brain.sha256) throw new Error(`Production brain ${brain.snakeId} failed checksum validation.`);
    bundles[brain.snakeId] = validateBrainBundle(JSON.parse(raw), brain.snakeId);
  }
  await set("current", { manifest, bundles }, cache());
  return { manifest, bundles };
}

function validateCached(value: LoadedProduction | undefined): LoadedProduction | null {
  if (!value || value.manifest.trainingSpecVersion !== 3 || value.manifest.observationSize !== 228) return null;
  const expected = new Set(value.manifest.brains.map((brain) => brain.snakeId));
  if (expected.size !== value.manifest.brains.length || Object.keys(value.bundles).length !== expected.size) return null;
  const bundles: Record<string, BrainBundle> = {};
  for (const snakeId of expected) {
    if (!value.bundles[snakeId]) return null;
    bundles[snakeId] = validateBrainBundle(value.bundles[snakeId], snakeId);
  }
  return { manifest: value.manifest, bundles };
}

export async function loadProductionModels(): Promise<{ manifest: ProductionManifest; bundles: Record<string, BrainBundle> } | null> {
  try {
    return await fetchCurrent();
  } catch {
    return validateCached(await get<LoadedProduction>("current", cache()));
  }
}
