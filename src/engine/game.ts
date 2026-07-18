// Bit-exact TypeScript port of the JAX engine in spec/game.py.
// All logic mirrors the Python line-for-line; arrays are flat row-major H*W.

import {
  H,
  W,
  DIRECTIONS,
  type Action,
  type GameState,
  type Grid,
  type Observation,
  type StepInfo,
} from './types';

const N = H * W;

// XLA gather semantics clamp out-of-bounds indices; mirror that for the
// speculative source/destination reads the Python performs even for passes
// and invalid actions.
function clampIndex(i: number, size: number): number {
  return i < 0 ? 0 : i >= size ? size - 1 : i;
}

export function createInitialState(grid: Grid): GameState {
  const armies = new Int32Array(N);
  const owner0 = new Uint8Array(N);
  const owner1 = new Uint8Array(N);
  const neutral = new Uint8Array(N);
  const generals = new Uint8Array(N);
  const cities = new Uint8Array(N);
  const mountains = new Uint8Array(N);
  const generalPositions: [number, number, number, number] = [-1, -1, -1, -1];

  for (let idx = 0; idx < N; idx++) {
    const v = grid[idx]!;
    const isGen0 = v === 1;
    const isGen1 = v === 2;
    const isGeneral = isGen0 || isGen1;
    const isMountain = v === -2;
    const isCity = v > 2;

    if (isGen0) {
      owner0[idx] = 1;
      generalPositions[0] = Math.floor(idx / W);
      generalPositions[1] = idx % W;
    }
    if (isGen1) {
      owner1[idx] = 1;
      generalPositions[2] = Math.floor(idx / W);
      generalPositions[3] = idx % W;
    }
    if (isGeneral) generals[idx] = 1;
    if (isCity) cities[idx] = 1;
    if (isMountain) mountains[idx] = 1;
    if (!isMountain && !isGeneral) neutral[idx] = 1;

    armies[idx] = isGeneral ? 1 : isCity ? v : 0;
  }

  return {
    armies,
    owner0,
    owner1,
    neutral,
    generals,
    cities,
    mountains,
    generalPositions,
    time: 0,
    winner: -1,
  };
}

function copyState(s: GameState): GameState {
  return {
    armies: s.armies.slice(),
    owner0: s.owner0.slice(),
    owner1: s.owner1.slice(),
    neutral: s.neutral.slice(),
    generals: s.generals,
    cities: s.cities,
    mountains: s.mountains,
    generalPositions: [s.generalPositions[0], s.generalPositions[1], s.generalPositions[2], s.generalPositions[3]],
    time: s.time,
    winner: s.winner,
  };
}

// _determine_move_order: computed on the PRE-move state, using each action's
// source/direction even when the action is a pass (no special-casing except
// only_p0_passes).
function determineMoveOrder(s: GameState, actions: [Action, Action]): 0 | 1 {
  const [pass0, si0raw, sj0raw, dir0raw] = actions[0];
  const [pass1, si1raw, sj1raw, dir1raw] = actions[1];

  const onlyP0Passes = pass0 === 1 && pass1 !== 1;

  const d0 = DIRECTIONS[clampIndex(dir0raw, 4)]!;
  const d1 = DIRECTIONS[clampIndex(dir1raw, 4)]!;

  const di0 = si0raw + d0[0];
  const dj0 = sj0raw + d0[1];
  const di1 = si1raw + d1[0];
  const dj1 = sj1raw + d1[1];

  const p0Chasing = di0 === si1raw && dj0 === sj1raw;
  const p1Chasing = di1 === si0raw && dj1 === sj0raw;

  // Clamped reads, mirroring XLA gather OOB behaviour.
  const dest0 = clampIndex(di0, H) * W + clampIndex(dj0, W);
  const dest1 = clampIndex(di1, H) * W + clampIndex(dj1, W);
  const src0 = clampIndex(si0raw, H) * W + clampIndex(sj0raw, W);
  const src1 = clampIndex(si1raw, H) * W + clampIndex(sj1raw, W);

  const p0Reinforcing = s.owner0[dest0] === 1;
  const p1Reinforcing = s.owner1[dest1] === 1;

  const army0 = s.armies[src0]!;
  const army1 = s.armies[src1]!;

  const p1WinsByChase = p1Chasing && !p0Chasing;
  const tieOnChase = p0Chasing === p1Chasing;
  const p1WinsByReinforce = tieOnChase && p1Reinforcing && !p0Reinforcing;
  const tieOnReinforce = p0Reinforcing === p1Reinforcing;
  const p1WinsByArmy = tieOnChase && tieOnReinforce && army1 > army0;

  return p1WinsByChase || p1WinsByReinforce || p1WinsByArmy || onlyP0Passes ? 1 : 0;
}

// execute_action + _execute_move + _apply_move, mutating `s` (already a copy).
function executeAction(s: GameState, player: 0 | 1, action: Action): void {
  const [passTurn, si, sj, dirRaw, splitArmy] = action;
  if (passTurn === 1) return;

  const inBounds = si >= 0 && si < H && sj >= 0 && sj < W;
  const d = DIRECTIONS[clampIndex(dirRaw, 4)]!;
  const di = si + d[0];
  const dj = sj + d[1];
  const destInBounds = di >= 0 && di < H && dj >= 0 && dj < W;

  const srcIdx = clampIndex(si, H) * W + clampIndex(sj, W);
  const dstIdx = clampIndex(di, H) * W + clampIndex(dj, W);

  const ownsSource = (player === 0 ? s.owner0[srcIdx] : s.owner1[srcIdx]) === 1;
  const sourceArmy = s.armies[srcIdx]!;

  let armyToMove = splitArmy === 1 ? Math.floor(sourceArmy / 2) : sourceArmy - 1;
  armyToMove = Math.max(0, Math.min(armyToMove, sourceArmy - 1));

  const validMove = inBounds && destInBounds && ownsSource && armyToMove > 0 && s.mountains[dstIdx] === 0;
  if (!validMove) return;

  const targetOwner0 = s.owner0[dstIdx] === 1;
  const targetOwner1 = s.owner1[dstIdx] === 1;
  const targetNeutral = s.neutral[dstIdx] === 1;

  const movingToOwn = (player === 0 && targetOwner0) || (player === 1 && targetOwner1);

  if (movingToOwn) {
    s.armies[dstIdx] = s.armies[dstIdx]! + armyToMove;
    s.armies[srcIdx] = s.armies[srcIdx]! - armyToMove;
    return;
  }

  const targetArmy = s.armies[dstIdx]!;
  const attackerWins = armyToMove > targetArmy;
  const remainingArmy = Math.abs(targetArmy - armyToMove);

  s.armies[dstIdx] = remainingArmy;
  s.armies[srcIdx] = s.armies[srcIdx]! - armyToMove;

  if (attackerWins) {
    if (player === 0) {
      s.owner0[dstIdx] = 1;
      if (targetOwner1) s.owner1[dstIdx] = 0;
    } else {
      s.owner1[dstIdx] = 1;
      if (targetOwner0) s.owner0[dstIdx] = 0;
    }
    if (targetNeutral) s.neutral[dstIdx] = 0;
  }

  const generalCaptured = attackerWins && s.generals[dstIdx] === 1;
  if (generalCaptured) s.winner = player;
}

// global_update: growth on the NEW time. Both increments can fire on one tick.
function globalUpdate(s: GameState): void {
  if (s.time % 50 === 0) {
    for (let i = 0; i < N; i++) {
      s.armies[i] = s.armies[i]! + s.owner0[i]! + s.owner1[i]!;
    }
  }
  if (s.time % 2 === 0) {
    for (let i = 0; i < N; i++) {
      if (s.generals[i] === 1 || s.cities[i] === 1) {
        s.armies[i] = s.armies[i]! + s.owner0[i]! + s.owner1[i]!;
      }
    }
  }
}

function transferLoserCellsToWinner(s: GameState): void {
  const winner = s.winner as 0 | 1;
  const loser = (1 - winner) as 0 | 1;
  const winOwn = winner === 0 ? s.owner0 : s.owner1;
  const loseOwn = loser === 0 ? s.owner0 : s.owner1;
  for (let i = 0; i < N; i++) {
    const loserCell = loseOwn[i]!;
    if (loserCell === 1) winOwn[i] = 1;
    loseOwn[i] = 0;
    if (loserCell === 1) s.neutral[i] = 0;
  }
}

function computeInfo(s: GameState): StepInfo {
  let army0 = 0;
  let army1 = 0;
  let land0 = 0;
  let land1 = 0;
  for (let i = 0; i < N; i++) {
    if (s.owner0[i] === 1) {
      army0 += s.armies[i]!;
      land0 += 1;
    }
    if (s.owner1[i] === 1) {
      army1 += s.armies[i]!;
      land1 += 1;
    }
  }
  return {
    army: [army0, army1],
    land: [land0, land1],
    isDone: s.winner >= 0,
    winner: s.winner,
    time: s.time,
  };
}

export function step(state: GameState, actions: [Action, Action]): { state: GameState; info: StepInfo } {
  const doneBefore = state.winner >= 0;
  const s = copyState(state);

  const firstPlayer = determineMoveOrder(s, actions);
  const secondPlayer = (1 - firstPlayer) as 0 | 1;

  executeAction(s, firstPlayer, actions[firstPlayer]);
  executeAction(s, secondPlayer, actions[secondPlayer]);

  if (!doneBefore) s.time += 1;

  if (s.winner >= 0) {
    transferLoserCellsToWinner(s);
  } else {
    globalUpdate(s);
  }

  return { state: s, info: computeInfo(s) };
}

// get_visibility: 3x3 max-pool (8-neighbour dilation) of the player's ownership.
function getVisibility(owned: Uint8Array): Uint8Array {
  const visible = new Uint8Array(N);
  for (let i = 0; i < H; i++) {
    for (let j = 0; j < W; j++) {
      let v = 0;
      for (let di = -1; di <= 1 && v === 0; di++) {
        const ni = i + di;
        if (ni < 0 || ni >= H) continue;
        for (let dj = -1; dj <= 1; dj++) {
          const nj = j + dj;
          if (nj < 0 || nj >= W) continue;
          if (owned[ni * W + nj] === 1) {
            v = 1;
            break;
          }
        }
      }
      visible[i * W + j] = v;
    }
  }
  return visible;
}

export function getObservation(state: GameState, player: 0 | 1): Observation {
  const owned = player === 0 ? state.owner0 : state.owner1;
  const opponent = player === 0 ? state.owner1 : state.owner0;
  const visible = getVisibility(owned);

  const armies = new Int32Array(N);
  const generals = new Uint8Array(N);
  const cities = new Uint8Array(N);
  const mountains = new Uint8Array(N);
  const neutralCells = new Uint8Array(N);
  const ownedCells = new Uint8Array(N);
  const opponentCells = new Uint8Array(N);
  const fogCells = new Uint8Array(N);
  const structuresInFog = new Uint8Array(N);

  for (let i = 0; i < N; i++) {
    if (visible[i] === 1) {
      armies[i] = state.armies[i]!;
      generals[i] = state.generals[i]!;
      cities[i] = state.cities[i]!;
      mountains[i] = state.mountains[i]!;
      neutralCells[i] = state.neutral[i]!;
      ownedCells[i] = owned[i]!;
      opponentCells[i] = opponent[i]!;
    } else {
      const structure = state.mountains[i] === 1 || state.cities[i] === 1;
      fogCells[i] = structure ? 0 : 1;
      structuresInFog[i] = structure ? 1 : 0;
    }
  }

  const info = computeInfo(state);
  return {
    armies,
    generals,
    cities,
    mountains,
    neutralCells,
    ownedCells,
    opponentCells,
    fogCells,
    structuresInFog,
    ownedLandCount: info.land[player]!,
    ownedArmyCount: info.army[player]!,
    opponentLandCount: info.land[1 - player]!,
    opponentArmyCount: info.army[1 - player]!,
    timestep: state.time,
  };
}
