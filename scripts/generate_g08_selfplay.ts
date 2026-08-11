/** Generate compact g08-vs-g08 action traces for browser spectator mode. */

import { writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import * as ort from "onnxruntime-node";

import { createState, playerView, step } from "../src/competition/game";
import { REAL_COMPETITION_MAPS } from "../src/competition/maps.generated";
import { decodeAction, PolicyMemory, preparePolicyInput } from "../src/competition/observation";
import type { Action, CompetitionState, Player, PolicyInput } from "../src/competition/types";


async function act(
  session: ort.InferenceSession,
  state: CompetitionState,
  player: Player,
  memory: PolicyMemory,
): Promise<Action> {
  /** Run one exact greedy g08 decision with independent fog memory. */
  const input: PolicyInput = preparePolicyInput(playerView(state, player), memory);
  const outputs = await session.run({
    obs: new ort.Tensor("float32", input.obs, [1, 45, 21, 21]),
    army_history: new ort.Tensor("float32", input.armyHistory, [1, 512]),
    land_history: new ort.Tensor("float32", input.landHistory, [1, 512]),
    penalty: new ort.Tensor("float32", input.penalty, [1, 4410]),
  });
  const tensor = outputs[session.outputNames[0]!];
  if (tensor === undefined) throw new Error("g08 ONNX graph emitted no policy tensor");
  const logits = tensor.data as Float32Array;
  let actionIndex = 0;
  for (let index = 1; index < logits.length; index += 1) {
    if (logits[index]! > logits[actionIndex]!) actionIndex = index;
  }
  return decodeAction(actionIndex);
}


async function main(): Promise<void> {
  /** Generate four tournament-map trajectories and emit a typed TS asset. */
  const directory = dirname(fileURLToPath(import.meta.url));
  const modelPath = resolve(directory, "../public/models/g08-champion.onnx");
  const session = await ort.InferenceSession.create(modelPath);
  const replays: Array<{
    readonly mapIndex: number;
    readonly actions: Array<readonly [Action, Action]>;
    readonly winner: number;
  }> = [];

  for (const mapIndex of [0, 8, 16, 24]) {
    let state = createState(REAL_COMPETITION_MAPS[mapIndex]!, false);
    const memories = [new PolicyMemory(), new PolicyMemory()] as const;
    const actions: Array<readonly [Action, Action]> = [];
    while (state.winner === -1) {
      const pair = await Promise.all([
        act(session, state, 0, memories[0]),
        act(session, state, 1, memories[1]),
      ]) as [Action, Action];
      actions.push(pair);
      state = step(state, pair);
    }
    replays.push({ mapIndex, actions, winner: state.winner });
    process.stdout.write(`map ${mapIndex}: ${actions.length} turns, winner ${state.winner}\n`);
  }

  const destination = resolve(directory, "../src/competition/selfplay.generated.ts");
  const encoded = JSON.stringify(replays);
  await writeFile(
    destination,
    'import type { Action } from "./types";\n\n'
      + "export interface SelfplayReplay {\n"
      + "  readonly mapIndex: number;\n"
      + "  readonly actions: ReadonlyArray<readonly [Action, Action]>;\n"
      + "  readonly winner: number;\n"
      + "}\n\n"
      + "/** Offline greedy g08-vs-g08 trajectories on real tournament maps. */\n"
      + `export const SELFPLAY_REPLAYS = ${encoded} as const satisfies readonly SelfplayReplay[];\n`,
    "utf8",
  );
}


await main();
