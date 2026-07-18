// Map validity property tests for the seeded generator.

import { describe, expect, it } from 'vitest';

import { generateMap } from '../src/engine/map';
import { H, W } from '../src/engine/types';

const N = H * W;
const NUM_SEEDS = 250;

function bfs(grid: Int32Array, start: number, passable: (v: number) => boolean): Int32Array {
  const dist = new Int32Array(N).fill(-1);
  dist[start] = 0;
  let frontier = [start];
  while (frontier.length > 0) {
    const next: number[] = [];
    for (const idx of frontier) {
      const i = Math.floor(idx / W);
      const j = idx % W;
      const d = dist[idx]! + 1;
      const neighbours = [
        i > 0 ? idx - W : -1,
        i < H - 1 ? idx + W : -1,
        j > 0 ? idx - 1 : -1,
        j < W - 1 ? idx + 1 : -1,
      ];
      for (const n of neighbours) {
        if (n >= 0 && dist[n] === -1 && passable(grid[n]!)) {
          dist[n] = d;
          next.push(n);
        }
      }
    }
    frontier = next;
  }
  return dist;
}

describe('generateMap', () => {
  it('is deterministic per seed', () => {
    for (let seed = 0; seed < 10; seed++) {
      expect(Array.from(generateMap(seed))).toEqual(Array.from(generateMap(seed)));
    }
  });

  it('produces valid maps over many seeds', () => {
    for (let seed = 0; seed < NUM_SEEDS; seed++) {
      const grid = generateMap(seed);
      expect(grid.length).toBe(N);

      let count0 = 0;
      let count1 = 0;
      let pos0 = -1;
      let pos1 = -1;
      let cities = 0;
      let mountains = 0;
      for (let idx = 0; idx < N; idx++) {
        const v = grid[idx]!;
        const validValue = v === -2 || v === 0 || v === 1 || v === 2 || (v >= 40 && v <= 50);
        if (!validValue) throw new Error(`seed ${seed}: invalid cell value ${v} at ${idx}`);
        if (v === 1) {
          count0++;
          pos0 = idx;
        } else if (v === 2) {
          count1++;
          pos1 = idx;
        } else if (v === -2) {
          mountains++;
        } else if (v > 2) {
          cities++;
        }
      }

      // Exactly one general per player.
      expect(count0, `seed ${seed}: P0 general count`).toBe(1);
      expect(count1, `seed ${seed}: P1 general count`).toBe(1);

      // Manhattan distance between generals >= 3.
      const i0 = Math.floor(pos0 / W);
      const j0 = pos0 % W;
      const i1 = Math.floor(pos1 / W);
      const j1 = pos1 % W;
      const manhattan = Math.abs(i0 - i1) + Math.abs(j0 - j1);
      expect(manhattan, `seed ${seed}: general Manhattan distance`).toBeGreaterThanOrEqual(3);

      // Generals connected through open ground (cities NOT passable).
      const open = (v: number): boolean => v >= 0 && v <= 2;
      const distOpen = bfs(grid, pos0, open);
      expect(distOpen[pos1], `seed ${seed}: generals not connected through open ground`).not.toBe(-1);

      // Each general has a city within BFS distance 7 (mountains impassable).
      const noMountains = (v: number): boolean => v !== -2;
      for (const pos of [pos0, pos1]) {
        const dist = bfs(grid, pos, noMountains);
        let nearestCity = Infinity;
        for (let idx = 0; idx < N; idx++) {
          if (grid[idx]! > 2 && dist[idx] !== -1) nearestCity = Math.min(nearestCity, dist[idx]!);
        }
        expect(nearestCity, `seed ${seed}: no city within BFS 7 of general at ${pos}`).toBeLessThanOrEqual(7);
      }

      // Totals.
      expect(cities, `seed ${seed}: city count`).toBeGreaterThanOrEqual(7);
      expect(cities, `seed ${seed}: city count`).toBeLessThanOrEqual(11);
      const mountainFraction = mountains / N;
      expect(mountainFraction, `seed ${seed}: mountain fraction`).toBeGreaterThanOrEqual(0.08);
      expect(mountainFraction, `seed ${seed}: mountain fraction`).toBeLessThanOrEqual(0.3);
    }
  });
});
