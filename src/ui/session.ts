// Pure game session: engine state + history + per-player move queues.
// No DOM, no rendering — the app drives it from a setInterval loop and the
// renderer reads it from a requestAnimationFrame loop.

import { createInitialState, getObservation, step } from "../engine/game";
import { generateMap } from "../engine/map";
import type { Action, GameState, Observation, StepInfo } from "../engine/types";

export const PASS_ACTION: Action = [1, 0, 0, 0, 0];

export interface QueuedMove {
  row: number;
  col: number;
  dir: number; // engine direction: 0=up 1=down 2=left 3=right
  split: boolean;
}

export function moveToAction(m: QueuedMove): Action {
  return [0, m.row, m.col, m.dir, m.split ? 1 : 0];
}

// Bound memory in long bot-vs-bot games; the UI never replays old states.
const HISTORY_CAP = 1024;

export class GameSession {
  state: GameState;
  readonly seed: number;
  lastInfo: StepInfo | null = null;
  private readonly history: GameState[] = [];
  private readonly queues: [QueuedMove[], QueuedMove[]] = [[], []];

  constructor(seed: number) {
    this.seed = seed >>> 0;
    this.state = createInitialState(generateMap(this.seed));
  }

  get done(): boolean {
    return this.state.winner >= 0;
  }

  get winner(): number {
    return this.state.winner;
  }

  get turn(): number {
    return this.state.time;
  }

  obs(player: 0 | 1): Observation {
    return getObservation(this.state, player);
  }

  /**
   * State to render: once the game ends, the winning step has already
   * transferred every loser cell to the winner (engine semantics), which
   * reads as an instant reset. Show the last pre-capture frame instead.
   */
  viewState(): GameState {
    if (this.done && this.history.length > 0) {
      return this.history[this.history.length - 1]!;
    }
    return this.state;
  }

  step(actions: [Action, Action]): StepInfo {
    this.history.push(this.state);
    if (this.history.length > HISTORY_CAP) {
      this.history.splice(0, this.history.length - HISTORY_CAP);
    }
    const { state, info } = step(this.state, actions);
    this.state = state;
    this.lastInfo = info;
    return info;
  }

  enqueue(player: 0 | 1, move: QueuedMove): void {
    this.queues[player].push(move);
  }

  dequeue(player: 0 | 1): QueuedMove | undefined {
    return this.queues[player].shift();
  }

  clearQueue(player: 0 | 1): void {
    this.queues[player].length = 0;
  }

  /** Remove and return the most recently queued move (E = undo). */
  popQueue(player: 0 | 1): QueuedMove | undefined {
    return this.queues[player].pop();
  }

  queueOf(player: 0 | 1): readonly QueuedMove[] {
    return this.queues[player];
  }

  /** Full-map totals for the spectator HUD. Human mode uses obs counts instead. */
  totals(state: GameState = this.state): { army: [number, number]; land: [number, number] } {
    let a0 = 0;
    let a1 = 0;
    let l0 = 0;
    let l1 = 0;
    const { armies, owner0, owner1 } = state;
    for (let i = 0; i < armies.length; i++) {
      if (owner0[i] === 1) {
        a0 += armies[i]!;
        l0++;
      } else if (owner1[i] === 1) {
        a1 += armies[i]!;
        l1++;
      }
    }
    return { army: [a0, a1], land: [l0, l1] };
  }
}
