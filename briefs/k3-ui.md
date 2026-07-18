# K3 — board UI, human input, game loop

You are working in a git worktree of the `generals` web app repo. The engine
(`src/engine/`) and model pipeline (`src/model/`) already exist and are
fixture-verified — build the playable UI on top of them. Acceptance:
`npm test` stays green, `npx tsc --noEmit` clean, `npm run build` succeeds,
and `npm run dev` serves a playable game.

Do NOT modify: `package.json`, `tsconfig.json`, `vite.config.ts`,
`src/engine/**`, `src/model/**` (import only), `test/fixtures/**`,
`briefs/**`. No new npm dependencies. Commit on the current branch
(`feat(ui): …`).

## Deliverables

- `src/ui/board.ts` — canvas renderer
- `src/ui/input.ts` — selection + move queue
- `src/ui/app.ts` — game loop + mode UI; wire it in `src/main.ts`
- `src/ui/style.css`

## Look & feel (generals.io-like)

Canvas grid, ~48px tiles, dark theme. Player 0 = blue (#4363d8),
player 1 = red (#e6194b), neutral gray, mountains dark with a ▲ glyph,
cities ⬤ with army count, generals ♛ (crown). Army counts centered in
white; fog = darker tile, structures-in-fog show a neutral obstacle glyph
(mountain and hidden city look IDENTICAL in fog, like real generals.io).
Human sees the game through `getObservation` for their player — never
render from the true state in human mode. In bot-vs-bot mode render the
full state (spectator view). Header bar: turn counter, army/land totals
per player, mode controls.

## Modes

1. **Play vs bot** — human is player 0, bot (worker, champion model) is
   player 1. Turn timer: 500 ms per tick (add a speed selector ×1 ×2 ×5).
   Human input, exactly generals.io-style:
   - click a tile you own to select; arrow keys / WASD queue a move from
     the selected cell in that direction, selection follows the queued
     destination; clicking another owned tile re-anchors.
   - moves enqueue (visualize the queued path with arrows); each tick pops
     the front of the queue and submits it; illegal pops are dropped
     (engine no-ops anyway). `q` clears the queue. Holding `z` when
     queueing marks the move as split (50%).
   - if the human queue is empty on a tick, submit pass.
2. **Bot vs bot** — champion vs spatial-v5 (dropdown to pick either model
   per seat, or the same model mirrored). Bots get fresh carries each game
   (`reset()`); each seat acts on ITS OWN observation each tick. Speed
   slider 1–20 ticks/sec, pause/step buttons.

Game over: overlay with winner, turn count, rematch button (new map seed).
Map: `generateMap(seed)` with a visible seed field + randomize button.

## Architecture notes

- Keep a pure `GameSession` object (state, histories, queues) separate from
  rendering; the loop is `setInterval`-driven at the chosen tick rate.
- The bot worker is async — request its action when the tick opens; if it
  hasn't answered by the next tick (it will, it's ~5 ms), submit pass.
- Render from a `requestAnimationFrame` loop reading the session.

## Report

Screenshot-worthy checklist at the end: what works, what's stubbed.
