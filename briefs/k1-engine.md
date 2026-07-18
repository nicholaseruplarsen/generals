# K1 — TypeScript engine port (bit-exact)

You are working in a git worktree of the `generals` web app repo. Your job:
port the generals.io 1v1 game engine from the Python/JAX reference in
`spec/game.py` (+ map generator in `spec/grid.py`) to TypeScript, and prove
it bit-exact against the golden fixtures in `test/fixtures/engine/`.

Run `npm install` once before starting. Acceptance = `npm test` green and
`npx tsc --noEmit` clean. Commit your work on the current branch with
conventional-commit messages (`feat(engine): …`).

## Deliverables

- `src/engine/game.ts` — state creation, step, observations
- `src/engine/map.ts` — seeded map generator
- `test/engine.test.ts` — fixture replay tests (all three fixture files)
- `test/map.test.ts` — map validity property tests

Use ONLY the types in `src/engine/types.ts` (this file is the contract with
the other agents — do NOT modify it, or any of: `package.json`,
`tsconfig.json`, `vite.config.ts`, `test/fixtures/**`, `briefs/**`,
`src/model/**`, `src/ui/**`). No new npm dependencies.

## Engine semantics (mirror `spec/game.py` exactly)

API to implement (exact names):

```ts
createInitialState(grid: Grid): GameState
step(state: GameState, actions: [Action, Action]): { state: GameState; info: StepInfo }
getObservation(state: GameState, player: 0 | 1): Observation
```

`step` may return a new state or mutate a copy — but must never mutate its
input (the UI keeps history).

Order of operations inside `step` (this ordering is what the fixtures encode):

1. `doneBefore = state.winner >= 0`.
2. Determine move order (`_determine_move_order`): p0 moves first UNLESS
   `p1_chasing & !p0_chasing`, else on chase-tie `p1_reinforcing &
   !p0_reinforcing`, else on reinforce-tie `army1 > army0`, else if
   `pass0 & !pass1`. ("chasing" = your destination is the opponent's source
   cell; "reinforcing" = destination is your own cell. All computed on the
   PRE-move state, using each action's source/direction even if the action
   is a pass — mirror the Python exactly, it does not special-case passes
   except `only_p0_passes`.)
3. Execute first player's action, then second player's, sequentially on the
   evolving state. A move: `army = split ? floor(a/2) : a-1`, clamped to
   `[0, source-1]`; invalid (out of bounds / unowned source / army<=0 /
   impassable dest) = silent no-op. Moving onto your own cell adds armies;
   onto enemy/neutral: attacker wins iff `army > target`, remaining =
   `|target - army|`, source decremented either way. Capturing a cell with
   a general (not your own) sets `winner`.
4. If NOT `doneBefore`: `time += 1`.
5. If `winner >= 0`: transfer ALL loser cells to winner (neutral untouched
   except loser cells leave neutral unchanged — see
   `_transfer_loser_cells_to_winner`). ELSE growth on the NEW time: if
   `time % 50 == 0` every owned cell +1; if `time % 2 == 0` every owned
   general/city cell +1 (both can fire on the same tick).
6. Info: army/land sums, isDone, winner, time.

Observation (`get_observation`): visibility = 3×3 dilation of own cells;
`armies/generals/cities/mountains/neutral/owned/opponent` are masked by
visibility; `fogCells = !visible & !(mountains|cities)`;
`structuresInFog = !visible & (mountains|cities)`; counts are TRUE totals
(not fog-limited); `timestep = state.time`.

## Fixture replay test

For each of the three files in `test/fixtures/engine/`:

```
state = createInitialState(Int32Array(fx.grid))
for (t, step) of fx.steps:
    state = step(state, step.actions).state
    assert state matches step.state          // bit-exact, every field
    assert getObservation(state, 0) matches step.obs0   // bit-exact
    assert getObservation(state, 1) matches step.obs1
```

Static masks (`generals`, `cities`, `mountains`, `generalPositions`) are at
the fixture top level. The random-vs-random fixture deliberately contains
invalid moves (e.g. into fogged mountains) — your no-op handling must match.

## Map generator (`src/engine/map.ts`)

Port the ALGORITHM of `spec/grid.py` `generate_grid` (10×10, mountain
density 0.18–0.26, 9–11 cities, min general Manhattan distance 3, castle
values 40–50 inclusive) with your own small seeded PRNG (e.g. mulberry32) —
it does NOT need to match JAX's RNG, only the construction: generals first,
mountains, connectivity carve (L-path) if needed, one castle within ~6 BFS
of each general placed by converting a mountain bordering the reachable
region (fallback: empty reachable cell), remaining cities convert mountains,
generals re-asserted. `generateMap(seed: number): Grid` must be
deterministic per seed.

`test/map.test.ts`: over ≥200 seeds assert — exactly one `1` and one `2`;
Manhattan distance ≥ 3; generals connected through open ground (cities are
NOT passable for this check); each general has a city within BFS distance 7
(mountains impassable); total cities in [7, 11]; mountain fraction in
[0.08, 0.30]; all values in {-2, 0, 1, 2, 40..50}.

## Report

End with a short summary: what passed, any semantic ambiguities you hit and
how you resolved them against the Python.
