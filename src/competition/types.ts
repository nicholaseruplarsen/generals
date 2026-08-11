export const PAD_TO = 21;
export const HISTORY = 7;
export const TEMPORAL_WINDOW = 512;
export const NUM_ACTION_KINDS = 10;
export const NUM_ACTIONS = NUM_ACTION_KINDS * PAD_TO * PAD_TO;

export type Player = 0 | 1;
export type Action = readonly [kind: number, row: number, col: number, direction: number, half: number];

export const PASS_ACTION: Action = [1, 0, 0, 0, 0];
export const DIRECTIONS: ReadonlyArray<readonly [number, number]> = [
  [-1, 0],
  [1, 0],
  [0, -1],
  [0, 1],
];

export interface CompetitionMap {
  readonly rows: number;
  readonly cols: number;
  readonly mountains: ReadonlyArray<readonly [number, number]>;
  readonly generals: readonly [readonly [number, number], readonly [number, number]];
}

export interface CompetitionState {
  readonly rows: number;
  readonly cols: number;
  armies: Int32Array;
  owners: Int8Array;
  mountains: Uint8Array;
  castles: Uint8Array;
  readonly generals: readonly [number, number];
  turn: number;
  winner: -2 | -1 | Player;
}

export interface PlayerView {
  readonly state: CompetitionState;
  readonly player: Player;
  readonly visible: Uint8Array;
}

export interface PolicyInput {
  readonly obs: Float32Array;
  readonly armyHistory: Float32Array;
  readonly landHistory: Float32Array;
  readonly penalty: Float32Array;
}

export interface PolicyDecision {
  readonly action: Action;
  readonly actionIndex: number;
  readonly logits: Float32Array;
}
