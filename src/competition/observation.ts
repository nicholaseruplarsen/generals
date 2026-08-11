import { buildCost } from "./game";
import {
  DIRECTIONS,
  HISTORY,
  NUM_ACTIONS,
  PAD_TO,
  TEMPORAL_WINDOW,
  type Action,
  type Player,
  type PlayerView,
  type PolicyInput,
} from "./types";

const CELLS = PAD_TO * PAD_TO;
const CHANNELS = 31 + 2 * HISTORY;
const NEGATIVE_MASK = -1e9;

function planeIndex(channel: number, row: number, col: number): number {
  return channel * CELLS + row * PAD_TO + col;
}

export class PolicyMemory {
  /** Persistent obs-v2 memory, one independent instance per policy seat. */
  readonly armyStack = new Float32Array(HISTORY * CELLS);
  readonly enemyStack = new Float32Array(HISTORY * CELLS);
  readonly lastArmy = new Float32Array(CELLS);
  readonly lastEnemyArmy = new Float32Array(CELLS);
  readonly cities = new Uint8Array(CELLS);
  readonly generals = new Uint8Array(CELLS);
  readonly mountains = new Uint8Array(CELLS);
  readonly seen = new Uint8Array(CELLS);
  readonly enemySeen = new Uint8Array(CELLS);
  readonly lastEnemyValue = new Float32Array(CELLS);
  readonly lastEnemyTimestep = new Float32Array(CELLS);
  readonly ownGeneral = new Uint8Array(CELLS);
  readonly enemyGeneral = new Uint8Array(CELLS);
  readonly armyHistory = new Float32Array(TEMPORAL_WINDOW);
  readonly landHistory = new Float32Array(TEMPORAL_WINDOW);
}

function shiftStack(stack: Float32Array, current: Float32Array, previous: Float32Array): void {
  /** Shift seven frames backward and write the newest army delta at frame zero. */
  stack.copyWithin(CELLS, 0, (HISTORY - 1) * CELLS);
  for (let cell = 0; cell < CELLS; cell += 1) stack[cell] = current[cell]! - previous[cell]!;
  previous.set(current);
}

function shiftHistory(history: Float32Array, value: number): void {
  /** Append one scalar to the 512-turn temporal tape. */
  history.copyWithin(0, 1);
  history[history.length - 1] = value;
}

function totals(view: PlayerView): readonly [number, number, number, number] {
  /** Return own land/army and opponent land/army from privileged score totals. */
  let ownLand = 0;
  let ownArmy = 0;
  let enemyLand = 0;
  let enemyArmy = 0;
  for (let cell = 0; cell < view.state.owners.length; cell += 1) {
    if (view.state.owners[cell] === view.player) {
      ownLand += 1;
      ownArmy += view.state.armies[cell]!;
    } else if (view.state.owners[cell] === 1 - view.player) {
      enemyLand += 1;
      enemyArmy += view.state.armies[cell]!;
    }
  }
  return [ownLand, ownArmy, enemyLand, enemyArmy];
}

function enemyHalo(view: PlayerView): Uint8Array {
  /** Dilate currently visible enemy territory by one cell, matching obs-v2. */
  const out = new Uint8Array(CELLS);
  const { rows, cols, owners } = view.state;
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      let adjacent = false;
      for (let dr = -1; dr <= 1 && !adjacent; dr += 1) {
        for (let dc = -1; dc <= 1; dc += 1) {
          const rr = row + dr;
          const cc = col + dc;
          if (rr < 0 || rr >= rows || cc < 0 || cc >= cols) continue;
          const source = rr * cols + cc;
          if (view.visible[source] === 1 && owners[source] === 1 - view.player) {
            adjacent = true;
            break;
          }
        }
      }
      out[row * PAD_TO + col] = adjacent ? 1 : 0;
    }
  }
  return out;
}

function actionPenalty(view: PlayerView): Float32Array {
  /** Build the submission's exact kind-major 4,410-action legality penalty. */
  const penalty = new Float32Array(NUM_ACTIONS).fill(NEGATIVE_MASK);
  const { state, player, visible } = view;
  for (let row = 0; row < state.rows; row += 1) {
    for (let col = 0; col < state.cols; col += 1) {
      const source = row * state.cols + col;
      const padded = row * PAD_TO + col;
      if (state.owners[source] === player && state.armies[source]! > 1) {
        for (let direction = 0; direction < 4; direction += 1) {
          const rr = row + DIRECTIONS[direction]![0];
          const cc = col + DIRECTIONS[direction]![1];
          if (rr < 0 || rr >= state.rows || cc < 0 || cc >= state.cols) continue;
          const target = rr * state.cols + cc;
          if (visible[target] === 1 && state.mountains[target] === 1) continue;
          penalty[direction * CELLS + padded] = 0;
          penalty[(direction + 4) * CELLS + padded] = 0;
        }
      }
      const isGeneral = state.generals[0] === source || state.generals[1] === source;
      if (state.owners[source] === player && state.mountains[source] === 0 &&
          state.castles[source] === 0 && !isGeneral &&
          state.armies[source]! >= buildCost(state, player, row, col)) {
        penalty[9 * CELLS + padded] = 0;
      }
    }
  }
  penalty.fill(0, 8 * CELLS, 9 * CELLS);
  return penalty;
}

export function preparePolicyInput(view: PlayerView, memory: PolicyMemory): PolicyInput {
  /** Convert one legal player view into the normalized 45-channel g08 input. */
  const { state, player, visible } = view;
  const currentArmy = new Float32Array(CELLS);
  const currentEnemy = new Float32Array(CELLS);
  const currentTotal = new Float32Array(CELLS);
  const neutralArmy = new Float32Array(CELLS);
  const opponentHalo = enemyHalo(view);
  const [ownLand, ownArmy, enemyLand, enemyArmy] = totals(view);

  for (let row = 0; row < PAD_TO; row += 1) {
    for (let col = 0; col < PAD_TO; col += 1) {
      const padded = row * PAD_TO + col;
      const onMap = row < state.rows && col < state.cols;
      if (!onMap) {
        memory.mountains[padded] = 1;
        continue;
      }
      const source = row * state.cols + col;
      const owner = state.owners[source];
      const seenNow = visible[source] === 1;
      if (seenNow || owner === player) currentTotal[padded] = state.armies[source]!;
      if (owner === player) currentArmy[padded] = state.armies[source]!;
      if (seenNow && owner === 1 - player) currentEnemy[padded] = state.armies[source]!;
      if (seenNow && owner === -1) neutralArmy[padded] = state.armies[source]!;
      if (seenNow) {
        memory.seen[padded] = 1;
        if (state.mountains[source] === 1) memory.mountains[padded] = 1;
        if (state.castles[source] === 1) memory.cities[padded] = 1;
        if (state.generals[0] === source || state.generals[1] === source) memory.generals[padded] = 1;
        if (source === state.generals[player]) memory.ownGeneral[padded] = 1;
        if (source === state.generals[1 - player]) memory.enemyGeneral[padded] = 1;
      }
      if (opponentHalo[padded] === 1) memory.enemySeen[padded] = 1;
      if (currentEnemy[padded]! > 0) {
        memory.lastEnemyValue[padded] = currentEnemy[padded]!;
        memory.lastEnemyTimestep[padded] = 0;
      } else {
        memory.lastEnemyTimestep[padded] = memory.lastEnemyTimestep[padded]! + 1;
      }
    }
  }

  shiftStack(memory.armyStack, currentArmy, memory.lastArmy);
  shiftStack(memory.enemyStack, currentEnemy, memory.lastEnemyArmy);
  shiftHistory(memory.armyHistory, enemyArmy);
  shiftHistory(memory.landHistory, enemyLand);

  let ownGeneralIndex = 0;
  let enemyGeneralIndex = 0;
  let enemyKnown = false;
  for (let cell = 0; cell < CELLS; cell += 1) {
    if (memory.ownGeneral[cell] === 1) ownGeneralIndex = cell;
    if (memory.enemyGeneral[cell] === 1) {
      enemyGeneralIndex = cell;
      enemyKnown = true;
    }
  }
  let candidateCount = 0;
  const candidates = new Uint8Array(CELLS);
  for (let row = 0; row < PAD_TO; row += 1) {
    for (let col = 0; col < PAD_TO; col += 1) {
      const cell = row * PAD_TO + col;
      const distance = Math.abs(row - Math.floor(ownGeneralIndex / PAD_TO)) + Math.abs(col - ownGeneralIndex % PAD_TO);
      const eliminated = memory.seen[cell] === 1 && memory.generals[cell] === 0;
      const candidate = enemyKnown ? memory.enemyGeneral[cell] === 1 :
        row < state.rows && col < state.cols && memory.mountains[cell] === 0 && !eliminated && distance >= 17;
      candidates[cell] = candidate ? 1 : 0;
      if (candidate) candidateCount += 1;
    }
  }

  const obs = new Float32Array(CHANNELS * CELLS);
  const logCandidates = Math.log1p(candidateCount) / Math.log1p(CELLS);
  for (let row = 0; row < PAD_TO; row += 1) {
    for (let col = 0; col < PAD_TO; col += 1) {
      const cell = row * PAD_TO + col;
      const onMap = row < state.rows && col < state.cols;
      const source = onMap ? row * state.cols + col : 0;
      const seenNow = onMap && visible[source] === 1;
      const owner = onMap ? state.owners[source] : -1;
      const structureInFog = onMap && !seenNow && (memory.mountains[cell] === 1 || memory.cities[cell] === 1);
      const enemyDistance = enemyKnown ?
        (Math.abs(row - Math.floor(enemyGeneralIndex / PAD_TO)) + Math.abs(col - enemyGeneralIndex % PAD_TO)) / (2 * PAD_TO) : 0;
      const values = [
        currentTotal[cell]! / 50,
        currentArmy[cell]! / 50,
        currentEnemy[cell]! / 50,
        neutralArmy[cell]! / 50,
        memory.seen[cell]!,
        memory.enemySeen[cell]!,
        memory.ownGeneral[cell]!,
        memory.enemyGeneral[cell]!,
        memory.cities[cell]!,
        memory.mountains[cell]!,
        onMap && seenNow && owner === -1 && state.mountains[source] === 0 ? 1 : 0,
        onMap && owner === player ? 1 : 0,
        onMap && seenNow && owner === 1 - player ? 1 : 0,
        onMap && !seenNow ? 1 : 0,
        structureInFog ? 1 : 0,
        state.turn / 50,
        (state.turn % 50) / 50,
        ownLand / 50,
        ownArmy / 50,
        enemyLand / 50,
        enemyArmy / 50,
        memory.lastEnemyValue[cell]! / 50,
        Math.log1p(memory.lastEnemyTimestep[cell]!) / 5,
        col / (PAD_TO - 1),
        row / (PAD_TO - 1),
        enemyKnown ? 1 : 0,
        enemyDistance,
        candidates[cell]!,
        logCandidates,
        state.turn >= 800 ? 1 : 0,
        Math.min(state.turn / 800, 1),
      ];
      for (let channel = 0; channel < 31; channel += 1) obs[planeIndex(channel, row, col)] = values[channel]!;
      for (let frame = 0; frame < HISTORY; frame += 1) {
        obs[planeIndex(31 + frame, row, col)] = memory.armyStack[frame * CELLS + cell]! / 50;
        obs[planeIndex(31 + HISTORY + frame, row, col)] = memory.enemyStack[frame * CELLS + cell]! / 50;
      }
    }
  }
  return { obs, armyHistory: memory.armyHistory.slice(), landHistory: memory.landHistory.slice(), penalty: actionPenalty(view) };
}

export function decodeAction(actionIndex: number): Action {
  /** Decode the network's kind-major flat index into the competition protocol. */
  const kind = Math.floor(actionIndex / CELLS);
  const position = actionIndex % CELLS;
  const row = Math.floor(position / PAD_TO);
  const col = position % PAD_TO;
  if (kind === 9) return [2, row, col, 0, 0];
  if (kind === 8) return [1, 0, 0, 0, 0];
  return [0, row, col, kind % 4, kind >= 4 ? 1 : 0];
}
