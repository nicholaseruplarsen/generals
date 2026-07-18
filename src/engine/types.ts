// Shared engine types — the contract between engine (K1), model (K2), and UI (K3).
// Do not change shapes without updating briefs/; fixtures encode these exactly.

export const H = 10;
export const W = 10;
export const NUM_ACTIONS = H * W * 8 + 1; // 801; last index = pass

// Cell values in a generated grid (mirrors generals-bots grid encoding):
//   -2 mountain · 0 empty · 1 P0 general · 2 P1 general · 40..50 city army value
export type Grid = Int32Array; // length H*W, row-major

// Directions, in engine order: 0=up 1=down 2=left 3=right
export const DIRECTIONS: ReadonlyArray<readonly [number, number]> = [
  [-1, 0],
  [1, 0],
  [0, -1],
  [0, 1],
];

/** [pass, row, col, direction, split] — pass=1 ignores the rest. */
export type Action = readonly [number, number, number, number, number];

export interface GameState {
  armies: Int32Array; // H*W
  owner0: Uint8Array; // H*W booleans
  owner1: Uint8Array;
  neutral: Uint8Array;
  generals: Uint8Array;
  cities: Uint8Array;
  mountains: Uint8Array;
  generalPositions: [number, number, number, number]; // [r0,c0,r1,c1]
  time: number;
  winner: number; // -1 ongoing, 0/1 winner
}

/** Player view with fog applied — field-for-field mirror of Python Observation. */
export interface Observation {
  armies: Int32Array;
  generals: Uint8Array;
  cities: Uint8Array;
  mountains: Uint8Array;
  neutralCells: Uint8Array;
  ownedCells: Uint8Array;
  opponentCells: Uint8Array;
  fogCells: Uint8Array;
  structuresInFog: Uint8Array;
  ownedLandCount: number;
  ownedArmyCount: number;
  opponentLandCount: number;
  opponentArmyCount: number;
  timestep: number;
}

export interface StepInfo {
  army: [number, number];
  land: [number, number];
  isDone: boolean;
  winner: number;
  time: number;
}
