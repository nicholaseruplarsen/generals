# K2 — observation encoding + ONNX inference (parity vs fixtures)

You are working in a git worktree of the `generals` web app repo. Your job:
port the neural policy's input pipeline from `spec/obs_encode.py` to
TypeScript and wrap the exported ONNX model for both tests (node) and the
browser (web worker), proving parity against
`test/fixtures/policy/champion-vs-hunter.json`.

Run `npm install` once before starting. Acceptance = `npm test` green and
`npx tsc --noEmit` clean. Commit on the current branch
(`feat(model): …` style messages).

Do NOT modify: `package.json`, `tsconfig.json`, `vite.config.ts`,
`src/engine/types.ts`, `test/fixtures/**`, `briefs/**`, `src/engine/**`
(beyond importing types), `src/ui/**`. Dependencies `onnxruntime-web`
(runtime) and `onnxruntime-node` (dev/tests) are already installed — no new
deps.

## Deliverables

- `src/model/encode.ts` — obs → Float32Array(18·10·10), mask, action codecs
- `src/model/infer.ts` — `GeneralsBot` class threading LSTM carry
- `src/model/worker.ts` — browser Web Worker wrapper
- `test/encode.test.ts`, `test/model.test.ts`

## Encoding (`src/model/encode.ts`) — mirror `spec/obs_encode.py`

Operating on the `Observation` type from `src/engine/types.ts`:

- `actionMask(obs): Uint8Array` — (4·H·W), direction-major exactly like the
  Python `(4, H, W)` stack: legal iff source owned & armies>1, destination
  in bounds and not (visible mountain | structure-in-fog).
- `encode(obs): Float32Array` — 18 channels of 10×10, channel-major, exact
  order and scaling from `spec/obs_encode.py` (log1p·0.2 armies, binary
  planes, scalar planes incl. log1p·0.1 army counts, timestep/500, then the
  4 mask planes). Use `Math.fround` after each scaled value so results are
  float32-exact.
- `flatMask(obs): Uint8Array` — length 801, layout
  `(row*W+col)*8 + dir*2 + split`, index 800 = pass (always legal).
- `decodeAction(idx): Action` and `NUM_ACTIONS = 801` re-exported.

## Inference (`src/model/infer.ts`)

```ts
class GeneralsBot {
  static async load(session: OrtSessionLike): Promise<GeneralsBot>
  reset(): void                       // zero the (1,256) h and c carries
  async act(obs: Observation): Promise<{ actionIdx: number; action: Action; value: number }>
}
```

Feed `{obs, h, c}`, read `{logits, value, h_out, c_out}`; store the new
carry; choose `argmax` of logits (the model masks illegal moves to -1e9
internally — but ALSO apply `flatMask` yourself before argmax as a
belt-and-braces step; fixture logits already have the mask applied).

`OrtSessionLike` = minimal interface (`run(feeds): Promise<results>` +
tensor construction) implemented by both `onnxruntime-node` and
`onnxruntime-web` `InferenceSession`; inject the session so tests use
onnxruntime-node while the browser worker uses onnxruntime-web. Model files:
`public/models/a100-i-draw2.onnx` (champion), `public/models/spatial-v5.onnx`.

`src/model/worker.ts`: a Web Worker speaking
`{type:"init", modelUrl}` / `{type:"reset"}` / `{type:"act", obs}` →
posts `{type:"ready"}` / `{type:"action", actionIdx, action, value}`.
Import `onnxruntime-web` and configure `ort.env.wasm.wasmPaths` relative to
`import.meta.env.BASE_URL` so it works under the `/generals/` base path.

## Parity tests

`test/encode.test.ts` — for the first 8 steps of the policy fixture (they
carry an `encoded` field): rebuild `Observation` from `obs0`, run your
`encode`, compare all 1800 floats within 1e-6. Also compare `flatMask` to
`legalMask` for EVERY step of the fixture.

`test/model.test.ts` — full-game replay: session over
`public/models/a100-i-draw2.onnx` (onnxruntime-node), carry starts at
zeros, for each of the 418 steps: encode `obs0` → run → compare logits to
fixture on legal entries (abs tol 1e-3), value within 1e-3, and the argmax
action index EXACTLY (`actionIdx`) — all 418 must match. Carry is never
reset mid-game.

## Report

End with a short summary: parity numbers achieved (max logit delta, argmax
match count), and anything surprising about ort-web/node API differences.
