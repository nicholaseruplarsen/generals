import {
  DIRECTIONS,
  type Action,
  type CompetitionMap,
  type CompetitionState,
  type Player,
  type PlayerView,
} from "./types";

const BUILD_BASE = 35;
const BUILD_PENALTY = 14;
const BUILD_DECAY = 2;

function index(state: CompetitionState, row: number, col: number): number {
  return row * state.cols + col;
}

function inBounds(state: CompetitionState, row: number, col: number): boolean {
  return row >= 0 && row < state.rows && col >= 0 && col < state.cols;
}

export function createState(map: CompetitionMap, swapSeats: boolean): CompetitionState {
  /** Create a fresh competition state from an immutable map header. */
  const cells = map.rows * map.cols;
  const armies = new Int32Array(cells);
  const owners = new Int8Array(cells).fill(-1);
  const mountains = new Uint8Array(cells);
  const castles = new Uint8Array(cells);
  const first = swapSeats ? map.generals[1] : map.generals[0];
  const second = swapSeats ? map.generals[0] : map.generals[1];
  const generals = [first[0] * map.cols + first[1], second[0] * map.cols + second[1]] as const;
  for (const [row, col] of map.mountains) mountains[row * map.cols + col] = 1;
  owners[generals[0]] = 0;
  owners[generals[1]] = 1;
  armies[generals[0]] = 1;
  armies[generals[1]] = 1;
  return { rows: map.rows, cols: map.cols, armies, owners, mountains, castles, generals, turn: 0, winner: -1 };
}

export function cloneState(state: CompetitionState): CompetitionState {
  /** Copy the mutable arrays while sharing immutable dimensions and generals. */
  return {
    ...state,
    armies: state.armies.slice(),
    owners: state.owners.slice(),
    mountains: state.mountains.slice(),
    castles: state.castles.slice(),
  };
}

export function buildCost(state: CompetitionState, player: Player, row: number, col: number): number {
  /** Return the exact crowding-priced castle cost for one owned target cell. */
  let cost = BUILD_BASE;
  for (let source = 0; source < state.owners.length; source += 1) {
    if (state.owners[source] !== player) continue;
    if (state.castles[source] === 0 && state.generals[player] !== source) continue;
    const sourceRow = Math.floor(source / state.cols);
    const sourceCol = source % state.cols;
    const distance = Math.abs(row - sourceRow) + Math.abs(col - sourceCol);
    cost += Math.max(0, BUILD_PENALTY - BUILD_DECAY * distance);
  }
  return cost;
}

export function isLegalBuild(state: CompetitionState, player: Player, row: number, col: number): boolean {
  /** Check ownership, terrain, structure, and live castle price. */
  if (!inBounds(state, row, col)) return false;
  const cell = index(state, row, col);
  const isGeneral = state.generals[0] === cell || state.generals[1] === cell;
  return state.owners[cell] === player && state.mountains[cell] === 0 &&
    state.castles[cell] === 0 && !isGeneral &&
    state.armies[cell]! >= buildCost(state, player, row, col);
}

function applyBuilds(state: CompetitionState, actions: readonly [Action, Action]): void {
  /** Competition builds resolve before either move and cannot conflict. */
  for (const player of [0, 1] as const) {
    const [kind, row, col] = actions[player];
    if (kind !== 2 || !isLegalBuild(state, player, row, col)) continue;
    const cell = index(state, row, col);
    state.armies[cell] = state.armies[cell]! - buildCost(state, player, row, col);
    state.castles[cell] = 1;
  }
}

function moveDestination(action: Action): readonly [number, number] {
  /** Resolve one action's destination without applying bounds checks. */
  const direction = DIRECTIONS[Math.max(0, Math.min(3, action[3]))]!;
  return [action[1] + direction[0], action[2] + direction[1]];
}

function moveOrder(state: CompetitionState, actions: readonly [Action, Action]): readonly [Player, Player] {
  /** Apply chasing, reinforcing, then smaller-source priority. */
  const a0 = actions[0];
  const a1 = actions[1];
  if (a0[0] !== 0 && a1[0] === 0) return [1, 0];
  if (a1[0] !== 0 && a0[0] === 0) return [0, 1];
  const d0 = moveDestination(a0);
  const d1 = moveDestination(a1);
  const chase0 = d0[0] === a1[1] && d0[1] === a1[2];
  const chase1 = d1[0] === a0[1] && d1[1] === a0[2];
  if (chase0 !== chase1) return chase0 ? [0, 1] : [1, 0];
  const reinforce0 = inBounds(state, d0[0], d0[1]) && state.owners[index(state, d0[0], d0[1])] === 0;
  const reinforce1 = inBounds(state, d1[0], d1[1]) && state.owners[index(state, d1[0], d1[1])] === 1;
  if (reinforce0 !== reinforce1) return reinforce0 ? [0, 1] : [1, 0];
  const source0 = inBounds(state, a0[1], a0[2]) ? state.armies[index(state, a0[1], a0[2])]! : 0;
  const source1 = inBounds(state, a1[1], a1[2]) ? state.armies[index(state, a1[1], a1[2])]! : 0;
  return source1 < source0 ? [1, 0] : [0, 1];
}

function executeMove(state: CompetitionState, player: Player, action: Action): void {
  /** Execute a legal move and resolve ordinary combat or deathtouch. */
  if (action[0] !== 0 || !inBounds(state, action[1], action[2])) return;
  const destination = moveDestination(action);
  if (!inBounds(state, destination[0], destination[1])) return;
  const source = index(state, action[1], action[2]);
  const target = index(state, destination[0], destination[1]);
  if (state.owners[source] !== player || state.armies[source]! <= 1 || state.mountains[target] === 1) return;
  const moving = action[4] === 1 ? Math.floor(state.armies[source]! / 2) : state.armies[source]! - 1;
  if (moving <= 0) return;
  state.armies[source] = state.armies[source]! - moving;
  if (state.owners[target] === player) {
    state.armies[target] = state.armies[target]! + moving;
    return;
  }
  const enemyGeneral = state.generals[1 - player];
  const deathtouch = state.turn >= 800 && target === enemyGeneral;
  if (deathtouch || moving > state.armies[target]!) {
    state.armies[target] = deathtouch ? Math.max(1, moving - state.armies[target]!) : moving - state.armies[target]!;
    state.owners[target] = player;
    if (target === enemyGeneral) state.winner = player;
  } else {
    state.armies[target] = state.armies[target]! - moving;
  }
}

function grow(state: CompetitionState): void {
  /** Apply owned-land and structure income after movement resolution. */
  for (let cell = 0; cell < state.owners.length; cell += 1) {
    if (state.owners[cell]! < 0) continue;
    if (state.turn % 50 === 0) state.armies[cell] = state.armies[cell]! + 1;
    if (state.turn % 2 === 0 &&
        (state.castles[cell] === 1 || state.generals[0] === cell || state.generals[1] === cell)) {
      state.armies[cell] = state.armies[cell]! + 1;
    }
  }
}

function transferLoser(state: CompetitionState, winner: Player): void {
  /** Transfer all remaining loser territory after a general capture. */
  for (let cell = 0; cell < state.owners.length; cell += 1) {
    if (state.owners[cell] === 1 - winner) state.owners[cell] = winner;
  }
}

export function step(state: CompetitionState, actions: readonly [Action, Action]): CompetitionState {
  /** Advance one simultaneous competition turn. */
  if (state.winner !== -1) return state;
  const next = cloneState(state);
  applyBuilds(next, actions);
  const order = moveOrder(next, actions);
  executeMove(next, order[0], actions[order[0]]);
  if (next.winner === -1) executeMove(next, order[1], actions[order[1]]);
  next.turn += 1;
  if (next.winner === 0 || next.winner === 1) transferLoser(next, next.winner);
  else if (next.turn >= 1200) next.winner = -2;
  else grow(next);
  return next;
}

export function visibility(state: CompetitionState, player: Player): Uint8Array {
  /** Return the player's 8-neighbour visibility halo. */
  const visible = new Uint8Array(state.owners.length);
  for (let row = 0; row < state.rows; row += 1) {
    for (let col = 0; col < state.cols; col += 1) {
      let seen = false;
      for (let dr = -1; dr <= 1 && !seen; dr += 1) {
        for (let dc = -1; dc <= 1; dc += 1) {
          const rr = row + dr;
          const cc = col + dc;
          if (inBounds(state, rr, cc) && state.owners[index(state, rr, cc)] === player) {
            seen = true;
            break;
          }
        }
      }
      visible[index(state, row, col)] = seen ? 1 : 0;
    }
  }
  return visible;
}

export function playerView(state: CompetitionState, player: Player): PlayerView {
  /** Package the state with the only visibility mask the player may use. */
  return { state, player, visible: visibility(state, player) };
}

export function totals(state: CompetitionState, player: Player): readonly [number, number] {
  /** Return [land, army] for one player. */
  let land = 0;
  let army = 0;
  for (let cell = 0; cell < state.owners.length; cell += 1) {
    if (state.owners[cell] !== player) continue;
    land += 1;
    army += state.armies[cell]!;
  }
  return [land, army];
}
