// Seeded 10x10 map generator — ports the construction algorithm of
// spec/grid.py generate_grid with a small local PRNG (mulberry32). It does
// NOT reproduce JAX's RNG stream, only the algorithm: generals first,
// mountains, connectivity carve (L-path) if needed, one castle within ~6 BFS
// of each general by converting a mountain bordering the reachable region
// (fallback: empty reachable cell), remaining cities convert mountains,
// generals re-asserted.

import { H, W, type Grid } from './types';

const N = H * W;

const MOUNTAIN_DENSITY: readonly [number, number] = [0.18, 0.26];
const NUM_CITIES_RANGE: readonly [number, number] = [9, 11];
const MIN_GENERALS_DISTANCE = 3;
const CASTLE_VAL_RANGE: readonly [number, number] = [40, 50]; // inclusive

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Uniform integer in [lo, hi] inclusive.
function randInt(rand: () => number, lo: number, hi: number): number {
  return lo + Math.floor(rand() * (hi - lo + 1));
}

// Uniformly sample one flat index where mask[i] is true. Returns -1 if empty.
function sampleFromMask(rand: () => number, mask: Uint8Array): number {
  let count = 0;
  for (let i = 0; i < mask.length; i++) count += mask[i]!;
  if (count === 0) return -1;
  let pick = Math.floor(rand() * count);
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] === 1 && pick-- === 0) return i;
  }
  return -1; // unreachable
}

// BFS distances from `start`; passable[i] must be 1 to traverse. -1 = unreachable.
function bfsDistances(passable: Uint8Array, start: number): Int32Array {
  const dist = new Int32Array(N).fill(-1);
  dist[start] = 0;
  let frontier = [start];
  while (frontier.length > 0) {
    const next: number[] = [];
    for (const idx of frontier) {
      const i = Math.floor(idx / W);
      const j = idx % W;
      const d = dist[idx]! + 1;
      if (i > 0 && passable[idx - W] === 1 && dist[idx - W] === -1) {
        dist[idx - W] = d;
        next.push(idx - W);
      }
      if (i < H - 1 && passable[idx + W] === 1 && dist[idx + W] === -1) {
        dist[idx + W] = d;
        next.push(idx + W);
      }
      if (j > 0 && passable[idx - 1] === 1 && dist[idx - 1] === -1) {
        dist[idx - 1] = d;
        next.push(idx - 1);
      }
      if (j < W - 1 && passable[idx + 1] === 1 && dist[idx + 1] === -1) {
        dist[idx + 1] = d;
        next.push(idx + 1);
      }
    }
    frontier = next;
  }
  return dist;
}

// carve_l_path: horizontal along pos_a's row to pos_b's column, then vertical
// to pos_b. Clears mountains/cities on the path, preserves generals.
function carveLPath(grid: Grid, posA: number, posB: number): void {
  const i1 = Math.floor(posA / W);
  const j1 = posA % W;
  const i2 = Math.floor(posB / W);
  const j2 = posB % W;
  for (let j = Math.min(j1, j2); j <= Math.max(j1, j2); j++) {
    const idx = i1 * W + j;
    if (grid[idx] === -2 || grid[idx]! > 2) grid[idx] = 0;
  }
  for (let i = Math.min(i1, i2); i <= Math.max(i1, i2); i++) {
    const idx = i * W + j2;
    if (grid[idx] === -2 || grid[idx]! > 2) grid[idx] = 0;
  }
}

export function generateMap(seed: number): Grid {
  const rand = mulberry32(seed);
  const grid = new Int32Array(N) as Grid;

  const numCities = randInt(rand, NUM_CITIES_RANGE[0], NUM_CITIES_RANGE[1]);
  const minMountains = Math.floor(MOUNTAIN_DENSITY[0] * N);
  const maxMountains = Math.floor(MOUNTAIN_DENSITY[1] * N);
  const numMountains = randInt(rand, minMountains, maxMountains);

  // Step 1: generals first on an empty grid.
  // valid_base_a_mask: a cell is valid if some corner is >= min distance away
  // (always true on 10x10 for distance 3, kept for parity with the spec).
  const firstValid = new Uint8Array(N);
  for (let i = 0; i < H; i++) {
    for (let j = 0; j < W; j++) {
      const maxCornerDist = Math.max(
        i + j,
        i + (W - 1 - j),
        H - 1 - i + j,
        H - 1 - i + (W - 1 - j),
      );
      firstValid[i * W + j] = maxCornerDist >= MIN_GENERALS_DISTANCE ? 1 : 0;
    }
  }
  const posFirst = sampleFromMask(rand, firstValid);

  const secondValid = new Uint8Array(N);
  const fi = Math.floor(posFirst / W);
  const fj = posFirst % W;
  for (let i = 0; i < H; i++) {
    for (let j = 0; j < W; j++) {
      secondValid[i * W + j] =
        Math.abs(i - fi) + Math.abs(j - fj) >= MIN_GENERALS_DISTANCE ? 1 : 0;
    }
  }
  const posSecond = sampleFromMask(rand, secondValid);

  // Randomly assign which position becomes P0 vs P1.
  const swap = rand() < 0.5;
  const posA = swap ? posSecond : posFirst;
  const posB = swap ? posFirst : posSecond;
  grid[posA] = 1;
  grid[posB] = 2;

  // Step 2: mountains on empty cells.
  const emptyMask = (): Uint8Array => {
    const m = new Uint8Array(N);
    for (let i = 0; i < N; i++) m[i] = grid[i] === 0 ? 1 : 0;
    return m;
  };
  const mountainPool = emptyMask();
  for (let placed = 0; placed < numMountains; placed++) {
    const idx = sampleFromMask(rand, mountainPool);
    if (idx === -1) break;
    grid[idx] = -2;
    mountainPool[idx] = 0;
  }

  // Step 3: connectivity carve through open ground (cities not yet placed;
  // passable = empty or general, i.e. values 0..2, mirroring flood_fill_connected).
  const openGround = new Uint8Array(N);
  for (let i = 0; i < N; i++) openGround[i] = grid[i]! >= 0 && grid[i]! <= 2 ? 1 : 0;
  const distFromA = bfsDistances(openGround, posA);
  if (distFromA[posB] === -1) {
    carveLPath(grid, posA, posB);
  }

  // Step 4: one castle within ~6 BFS of each general, converting a mountain
  // bordering the reachable region; fallback to an empty reachable cell.
  const placeCastle = (pos: number, castleVal: number): void => {
    const passable = new Uint8Array(N); // only mountains impassable
    for (let i = 0; i < N; i++) passable[i] = grid[i] !== -2 ? 1 : 0;
    const reach = bfsDistances(passable, pos);

    const frontierMountain = new Uint8Array(N);
    const emptyReach = new Uint8Array(N);
    for (let idx = 0; idx < N; idx++) {
      const d = reach[idx]!;
      // reach_incl includes the general's own cell (distance 0).
      if (d >= 0 && d <= 6) {
        // dilate4(reach_incl) & mountains
        const i = Math.floor(idx / W);
        const j = idx % W;
        if (i > 0 && grid[idx - W] === -2) frontierMountain[idx - W] = 1;
        if (i < H - 1 && grid[idx + W] === -2) frontierMountain[idx + W] = 1;
        if (j > 0 && grid[idx - 1] === -2) frontierMountain[idx - 1] = 1;
        if (j < W - 1 && grid[idx + 1] === -2) frontierMountain[idx + 1] = 1;
        // empty_reach excludes the start cell itself.
        if (d > 0 && grid[idx] === 0) emptyReach[idx] = 1;
      }
    }
    let anyMountain = false;
    for (let i = 0; i < N; i++) anyMountain = anyMountain || frontierMountain[i] === 1;
    const candidates = anyMountain ? frontierMountain : emptyReach;
    const posCastle = sampleFromMask(rand, candidates);
    if (posCastle !== -1) grid[posCastle] = castleVal;
  };
  placeCastle(posA, randInt(rand, CASTLE_VAL_RANGE[0], CASTLE_VAL_RANGE[1]));
  placeCastle(posB, randInt(rand, CASTLE_VAL_RANGE[0], CASTLE_VAL_RANGE[1]));

  // Step 5: remaining cities, also by converting mountains.
  const remainingCities = numCities - 2;
  const mountainMask = new Uint8Array(N);
  for (let i = 0; i < N; i++) mountainMask[i] = grid[i] === -2 ? 1 : 0;
  for (let placed = 0; placed < remainingCities; placed++) {
    const idx = sampleFromMask(rand, mountainMask);
    if (idx === -1) break;
    grid[idx] = randInt(rand, CASTLE_VAL_RANGE[0], CASTLE_VAL_RANGE[1]);
    mountainMask[idx] = 0;
  }

  // Step 6: re-assert generals as ground truth.
  grid[posA] = 1;
  grid[posB] = 2;

  // Step 7: small-grid guard. The spec's density range (0.18-0.26) is tuned
  // for 23x23; on 10x10 the carve + castle/city conversions can drain
  // mountains below 8% of the board. Top up by converting empty cells,
  // reverting any conversion that breaks open-ground connectivity between
  // the generals or pushes a general's nearest city beyond BFS distance 7.
  const isValid = (): boolean => {
    const open = new Uint8Array(N);
    const passable = new Uint8Array(N);
    for (let i = 0; i < N; i++) {
      open[i] = grid[i]! >= 0 && grid[i]! <= 2 ? 1 : 0;
      passable[i] = grid[i] !== -2 ? 1 : 0;
    }
    if (bfsDistances(open, posA)[posB] === -1) return false;
    for (const pos of [posA, posB]) {
      const dist = bfsDistances(passable, pos);
      let nearestCity = Infinity;
      for (let i = 0; i < N; i++) {
        if (grid[i]! > 2 && dist[i] !== -1) nearestCity = Math.min(nearestCity, dist[i]!);
      }
      if (nearestCity > 7) return false;
    }
    return true;
  };

  let mountains = 0;
  for (let i = 0; i < N; i++) if (grid[i] === -2) mountains++;
  if (mountains < 8) {
    const candidates = emptyMask();
    let idx = sampleFromMask(rand, candidates);
    while (mountains < 8 && idx !== -1) {
      candidates[idx] = 0;
      grid[idx] = -2;
      if (isValid()) {
        mountains++;
      } else {
        grid[idx] = 0;
      }
      idx = sampleFromMask(rand, candidates);
    }
  }

  return grid;
}
