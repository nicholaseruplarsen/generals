// Fixture replay tests: replay every recorded game bit-exactly and check both
// players' observations after every step.

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { describe, expect, it } from 'vitest';

import { createInitialState, getObservation, step } from '../src/engine/game';
import { H, W, type Action, type GameState, type Observation } from '../src/engine/types';

const here = dirname(fileURLToPath(import.meta.url));
const FIXTURES = ['expander-vs-hunter', 'hunter-vs-expander', 'random-vs-random'];

interface FixtureState {
  armies: number[];
  owner0: number[];
  owner1: number[];
  neutral: number[];
  time: number;
  winner: number;
}

interface FixtureObs {
  armies: number[];
  generals: number[];
  cities: number[];
  mountains: number[];
  neutralCells: number[];
  ownedCells: number[];
  opponentCells: number[];
  fogCells: number[];
  structuresInFog: number[];
  ownedLandCount: number;
  ownedArmyCount: number;
  opponentLandCount: number;
  opponentArmyCount: number;
  timestep: number;
}

interface Fixture {
  name: string;
  grid: number[];
  h: number;
  w: number;
  generals: number[];
  cities: number[];
  mountains: number[];
  generalPositions: number[];
  steps: Array<{ actions: [Action, Action]; state: FixtureState; obs0: FixtureObs; obs1: FixtureObs }>;
  finalWinner: number;
  numSteps: number;
}

function loadFixture(name: string): Fixture {
  const raw = readFileSync(join(here, 'fixtures', 'engine', `${name}.json`), 'utf8');
  return JSON.parse(raw) as Fixture;
}

function arr(a: ArrayLike<number>): number[] {
  return Array.from(a);
}

function expectState(actual: GameState, expected: FixtureState): void {
  expect(arr(actual.armies)).toEqual(expected.armies);
  expect(arr(actual.owner0)).toEqual(expected.owner0);
  expect(arr(actual.owner1)).toEqual(expected.owner1);
  expect(arr(actual.neutral)).toEqual(expected.neutral);
  expect(actual.time).toBe(expected.time);
  expect(actual.winner).toBe(expected.winner);
}

function expectObs(actual: Observation, expected: FixtureObs): void {
  expect(arr(actual.armies)).toEqual(expected.armies);
  expect(arr(actual.generals)).toEqual(expected.generals);
  expect(arr(actual.cities)).toEqual(expected.cities);
  expect(arr(actual.mountains)).toEqual(expected.mountains);
  expect(arr(actual.neutralCells)).toEqual(expected.neutralCells);
  expect(arr(actual.ownedCells)).toEqual(expected.ownedCells);
  expect(arr(actual.opponentCells)).toEqual(expected.opponentCells);
  expect(arr(actual.fogCells)).toEqual(expected.fogCells);
  expect(arr(actual.structuresInFog)).toEqual(expected.structuresInFog);
  expect(actual.ownedLandCount).toBe(expected.ownedLandCount);
  expect(actual.ownedArmyCount).toBe(expected.ownedArmyCount);
  expect(actual.opponentLandCount).toBe(expected.opponentLandCount);
  expect(actual.opponentArmyCount).toBe(expected.opponentArmyCount);
  expect(actual.timestep).toBe(expected.timestep);
}

describe.each(FIXTURES)('fixture replay: %s', (name) => {
  const fx = loadFixture(name);

  it('replays the game bit-exactly, with observations, every step', () => {
    expect(fx.h).toBe(H);
    expect(fx.w).toBe(W);
    expect(fx.steps.length).toBe(fx.numSteps);

    let state = createInitialState(new Int32Array(fx.grid));

    // Static masks live at the fixture top level.
    expect(arr(state.generals)).toEqual(fx.generals);
    expect(arr(state.cities)).toEqual(fx.cities);
    expect(arr(state.mountains)).toEqual(fx.mountains);
    expect(arr(state.generalPositions)).toEqual(fx.generalPositions);

    for (let t = 0; t < fx.steps.length; t++) {
      const fxStep = fx.steps[t]!;
      const before = state;
      const armiesBefore = arr(state.armies);
      const result = step(state, fxStep.actions);
      state = result.state;

      // step must never mutate its input (the UI keeps history).
      expect(arr(before.armies)).toEqual(armiesBefore);

      try {
        expectState(state, fxStep.state);
        expectObs(getObservation(state, 0), fxStep.obs0);
        expectObs(getObservation(state, 1), fxStep.obs1);
      } catch (err) {
        throw new Error(`mismatch at step ${t} of ${name}: ${(err as Error).message}`);
      }
    }

    expect(state.winner).toBe(fx.finalWinner);
  });
});
