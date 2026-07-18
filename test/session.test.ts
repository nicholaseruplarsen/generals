// Sanity checks for the pure GameSession that drives the UI loop: map
// generation, queue semantics, stepping, and a long random self-play game
// exercising the enqueue/dequeue path every tick.

import { describe, expect, it } from "vitest";
import type { Action } from "../src/engine/types";
import { decodeAction, flatMask } from "../src/model/encode";
import { GameSession, PASS_ACTION, moveToAction } from "../src/ui/session";

// Deterministic PRNG (mulberry32) so failures are reproducible.
function prng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomLegalAction(mask: Uint8Array, rand: () => number): Action {
  const legal: number[] = [];
  for (let i = 0; i < mask.length; i++) if (mask[i] === 1) legal.push(i);
  return decodeAction(legal[Math.floor(rand() * legal.length)]!);
}

describe("GameSession", () => {
  it("starts from a generated map with two generals and empty queues", () => {
    const s = new GameSession(42);
    expect(s.seed).toBe(42);
    expect(s.turn).toBe(0);
    expect(s.done).toBe(false);
    expect(s.state.generals.reduce((a, b) => a + b, 0)).toBe(2);
    expect(s.queueOf(0)).toHaveLength(0);
    expect(s.queueOf(1)).toHaveLength(0);
  });

  it("queues are FIFO, clearable, and map to engine actions", () => {
    const s = new GameSession(7);
    s.enqueue(0, { row: 1, col: 1, dir: 3, split: false });
    s.enqueue(0, { row: 1, col: 2, dir: 1, split: true });
    expect(s.queueOf(0)).toHaveLength(2);
    expect(moveToAction(s.dequeue(0)!)).toEqual([0, 1, 1, 3, 0]);
    expect(moveToAction(s.dequeue(0)!)).toEqual([0, 1, 2, 1, 1]);
    expect(s.dequeue(0)).toBeUndefined();
    s.enqueue(0, { row: 0, col: 0, dir: 0, split: false });
    s.clearQueue(0);
    expect(s.queueOf(0)).toHaveLength(0);
  });

  it("steps advance time without mutating prior state; obs stays fogged", () => {
    const s = new GameSession(123);
    const before = s.state;
    const armiesBefore = Array.from(before.armies);
    s.step([PASS_ACTION, PASS_ACTION]);
    expect(s.turn).toBe(1);
    expect(s.state).not.toBe(before);
    expect(Array.from(before.armies)).toEqual(armiesBefore);
    const obs = s.obs(0);
    const hidden =
      obs.fogCells.reduce((a, b) => a + b, 0) + obs.structuresInFog.reduce((a, b) => a + b, 0);
    expect(hidden).toBeGreaterThan(0);
    expect(obs.ownedLandCount).toBe(1);
  });

  it("plays a long random game through the queue path without crashing", () => {
    const rand = prng(999);
    const s = new GameSession(999);
    let ticks = 0;
    while (!s.done && ticks < 1500) {
      // Player 0 acts only through its queue (enqueue then pop), like the UI.
      const m0 = randomLegalAction(flatMask(s.obs(0)), rand);
      s.enqueue(0, {
        row: m0[1],
        col: m0[2],
        dir: m0[3],
        split: m0[4] === 1,
      });
      const popped = s.dequeue(0);
      const a0 = popped ? moveToAction(popped) : PASS_ACTION;
      const a1 = randomLegalAction(flatMask(s.obs(1)), rand);
      s.step([a0, a1]);
      ticks++;
    }
    expect(s.turn).toBe(ticks);
    if (s.done) expect([0, 1]).toContain(s.winner);
  });
});
