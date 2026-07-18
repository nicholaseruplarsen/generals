# generals

Browser recreation of [generals.io](https://generals.io) for playing against
— and spectating — the RL bots trained in the private `generals-bots`
research repo. Fully static; deploys to
<https://nicholaseruplarsen.github.io/generals/>.

- `src/engine/` — TypeScript port of the JAX 1v1 engine (bit-exact against
  golden fixtures exported from the Python reference).
- `src/model/` — observation encoding + onnxruntime-web inference of the
  trained recurrent policies (`public/models/*.onnx`, LSTM carry threaded
  turn to turn).
- `src/ui/` — canvas board, generals.io-style move queue, play-vs-bot and
  bot-vs-bot modes.
- `spec/` — frozen copies of the Python reference implementations.
- `test/fixtures/` — golden fixtures; regenerate with
  `research/rl/export_web_fixtures.py` in generals-bots. ONNX models come
  from `research/rl/export_onnx.py` there.
- `briefs/` — task briefs for the coding agents that built each layer.

```bash
npm install
npm test        # fixture parity + property tests
npm run dev     # local play
npm run build   # tsc + vite build (what CI deploys)
```

Deployed by `.github/workflows/deploy.yml` on push to `main`.
