// Browser Web Worker: runs GeneralsBot on onnxruntime-web (WASM backend).
// Protocol — in:  {type:"init", modelUrl} | {type:"reset"} | {type:"act", obs}
//           out:  {type:"ready"} | {type:"action", actionIdx, action, value}

// The wasm-only bundle: the default "onnxruntime-web" entry includes the
// WebGPU (jsep) backend and fetches ort-wasm-simd-threaded.jsep.mjs, which
// we don't ship in public/ort/. This one loads the plain wasm runtime.
import * as ort from "onnxruntime-web/wasm";
import type { Action, Observation } from "../engine/types";
import { GeneralsBot, ortSessionLike } from "./infer";

// onnxruntime-web downloads its WASM binaries at runtime; they are copied
// into public/ort/ and served under the vite base path (/generals/ in prod).
ort.env.wasm.wasmPaths = `${import.meta.env.BASE_URL}ort/`;

export type WorkerRequest =
  | { type: "init"; modelUrl: string }
  | { type: "reset" }
  | { type: "act"; obs: Observation };

export type WorkerResponse =
  | { type: "ready" }
  | { type: "action"; actionIdx: number; action: Action; value: number }
  | { type: "error"; message: string };

// Both DOM and WebWorker libs are in this project's tsconfig; pin `self` to
// the worker global scope explicitly.
const ctx = self as unknown as DedicatedWorkerGlobalScope;

let bot: GeneralsBot | null = null;

function post(msg: WorkerResponse): void {
  ctx.postMessage(msg);
}

ctx.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data;
  try {
    switch (msg.type) {
      case "init": {
        const session = await ort.InferenceSession.create(msg.modelUrl, {
          executionProviders: ["wasm"],
        });
        bot = await GeneralsBot.load(ortSessionLike(ort, session));
        post({ type: "ready" });
        break;
      }
      case "reset": {
        bot?.reset();
        break;
      }
      case "act": {
        if (!bot) throw new Error("worker used before init");
        const { actionIdx, action, value } = await bot.act(msg.obs);
        post({ type: "action", actionIdx, action, value });
        break;
      }
    }
  } catch (err) {
    post({ type: "error", message: err instanceof Error ? err.message : String(err) });
  }
};
