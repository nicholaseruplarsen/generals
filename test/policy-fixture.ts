// Shared loader for the policy parity fixture (champion-vs-hunter).

import { readFileSync } from "node:fs";
import type { Observation } from "../src/engine/types";

export interface RawObservation {
  armies: number[];
  generals: number[];
  cities: number[];
  mountains: number[];
  neutralCells: number[];
  ownedCells: number[];
  opponentCells: number[];
  fogCells: number[];
  structuresInFog: number[];
  ownedLandCount: number;
  ownedArmyCount: number;
  opponentLandCount: number;
  opponentArmyCount: number;
  timestep: number;
}

export interface RawStep {
  obs0: RawObservation;
  legalMask: number[];
  logits: number[];
  value: number;
  actionIdx: number;
  action: number[];
  hNorm: number;
  cNorm: number;
  /** Present on the first 8 steps only. */
  encoded?: number[];
}

export interface PolicyFixture {
  name: string;
  h: number;
  w: number;
  numSteps: number;
  finalWinner: number;
  steps: RawStep[];
}

export function loadPolicyFixture(): PolicyFixture {
  const url = new URL("./fixtures/policy/champion-vs-hunter.json", import.meta.url);
  return JSON.parse(readFileSync(url, "utf8")) as PolicyFixture;
}

/** Rebuild the engine Observation from a fixture step's obs0. */
export function toObservation(o: RawObservation): Observation {
  return {
    armies: Int32Array.from(o.armies),
    generals: Uint8Array.from(o.generals),
    cities: Uint8Array.from(o.cities),
    mountains: Uint8Array.from(o.mountains),
    neutralCells: Uint8Array.from(o.neutralCells),
    ownedCells: Uint8Array.from(o.ownedCells),
    opponentCells: Uint8Array.from(o.opponentCells),
    fogCells: Uint8Array.from(o.fogCells),
    structuresInFog: Uint8Array.from(o.structuresInFog),
    ownedLandCount: o.ownedLandCount,
    ownedArmyCount: o.ownedArmyCount,
    opponentLandCount: o.opponentLandCount,
    opponentArmyCount: o.opponentArmyCount,
    timestep: o.timestep,
  };
}
