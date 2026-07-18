// Observation encoding — TypeScript port of spec/obs_encode.py.
// Mirrors the Python channel order, scaling, and action layout exactly; every
// scaled value passes through Math.fround so results are float32-exact.

import { DIRECTIONS, H, NUM_ACTIONS, W } from "../engine/types";
import type { Action, Observation } from "../engine/types";

export const NUM_CHANNELS = 18; // 14 observation planes + 4 move-validity planes
export { NUM_ACTIONS };

const CELLS = H * W;

/**
 * (4, H, W) direction-major legality mask — move from (r,c) in direction d is
 * legal iff the source is owned with armies > 1 and the destination is in
 * bounds and not a visible mountain / structure-in-fog.
 */
export function actionMask(obs: Observation): Uint8Array {
  const mask = new Uint8Array(4 * CELLS);
  for (let d = 0; d < 4; d++) {
    const dir = DIRECTIONS[d]!;
    const dr = dir[0];
    const dc = dir[1];
    const plane = d * CELLS;
    for (let r = 0; r < H; r++) {
      const r2 = r + dr;
      if (r2 < 0 || r2 >= H) continue;
      for (let c = 0; c < W; c++) {
        const c2 = c + dc;
        if (c2 < 0 || c2 >= W) continue;
        const i = r * W + c;
        if (obs.ownedCells[i] === 0 || obs.armies[i]! <= 1) continue;
        const j = r2 * W + c2;
        if (obs.mountains[j] !== 0 || obs.structuresInFog[j] !== 0) continue;
        mask[plane + i] = 1;
      }
    }
  }
  return mask;
}

function fillPlane(out: Float32Array, ch: number, value: number): void {
  out.fill(value, ch * CELLS, (ch + 1) * CELLS);
}

/** (NUM_CHANNELS, H, W) float32 policy input, channel-major (18·10·10). */
export function encode(obs: Observation): Float32Array {
  const out = new Float32Array(NUM_CHANNELS * CELLS);
  const armies = obs.armies;
  // Channel 0: log1p(armies) * 0.2
  for (let i = 0; i < CELLS; i++) {
    out[i] = Math.fround(Math.log1p(armies[i]!) * 0.2);
  }
  // Channels 1–8: binary planes — Uint8 0/1 converts exactly to float32.
  out.set(obs.generals, 1 * CELLS);
  out.set(obs.cities, 2 * CELLS);
  out.set(obs.mountains, 3 * CELLS);
  out.set(obs.neutralCells, 4 * CELLS);
  out.set(obs.ownedCells, 5 * CELLS);
  out.set(obs.opponentCells, 6 * CELLS);
  out.set(obs.fogCells, 7 * CELLS);
  out.set(obs.structuresInFog, 8 * CELLS);
  // Channels 9–13: scalar planes.
  fillPlane(out, 9, Math.fround(obs.ownedLandCount / CELLS));
  fillPlane(out, 10, Math.fround(Math.log1p(obs.ownedArmyCount) * 0.1));
  fillPlane(out, 11, Math.fround(obs.opponentLandCount / CELLS));
  fillPlane(out, 12, Math.fround(Math.log1p(obs.opponentArmyCount) * 0.1));
  fillPlane(out, 13, Math.fround(obs.timestep / 500));
  // Channels 14–17: move-validity planes — actionMask is already (4,H,W)
  // direction-major, i.e. contiguous channels 14..17.
  out.set(actionMask(obs), 14 * CELLS);
  return out;
}

/** (801,) flat legal-action mask: idx = (row*W+col)*8 + dir*2 + split; 800 = pass (always legal). */
export function flatMask(obs: Observation): Uint8Array {
  const m = actionMask(obs); // (4, H, W) direction-major
  const out = new Uint8Array(NUM_ACTIONS);
  for (let cell = 0; cell < CELLS; cell++) {
    const base = cell * 8;
    for (let d = 0; d < 4; d++) {
      const v = m[d * CELLS + cell]!;
      out[base + d * 2] = v; // split = 0
      out[base + d * 2 + 1] = v; // split = 1
    }
  }
  out[NUM_ACTIONS - 1] = 1;
  return out;
}

/** Discrete index -> engine action [pass, row, col, direction, split]. */
export function decodeAction(idx: number): Action {
  if (idx >= CELLS * 8) return [1, 0, 0, 0, 0];
  const cell = Math.floor(idx / 8);
  return [0, Math.floor(cell / W), cell % W, Math.floor((idx % 8) / 2), idx % 2];
}
