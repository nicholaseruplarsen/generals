import * as ort from "onnxruntime-web/wasm";
import wasmUrl from "@ort-wasm?url";
import { decodeAction } from "./observation";
import type { PolicyInput } from "./types";

ort.env.wasm.wasmPaths = { wasm: wasmUrl };
ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);

export type CompetitionWorkerRequest =
  | { readonly type: "init"; readonly modelUrl: string }
  | { readonly type: "act"; readonly input: PolicyInput };

export type CompetitionWorkerResponse =
  | { readonly type: "ready" }
  | { readonly type: "decision"; readonly actionIndex: number; readonly action: ReturnType<typeof decodeAction>; readonly logits: Float32Array }
  | { readonly type: "error"; readonly message: string };

const worker = self as unknown as DedicatedWorkerGlobalScope;
let session: ort.InferenceSession | null = null;

function post(message: CompetitionWorkerResponse, transfer: Transferable[] = []): void {
  /** Send one typed response to the UI thread. */
  worker.postMessage(message, transfer);
}

worker.onmessage = async (event: MessageEvent<CompetitionWorkerRequest>): Promise<void> => {
  try {
    if (event.data.type === "init") {
      session = await ort.InferenceSession.create(event.data.modelUrl, { executionProviders: ["wasm"] });
      post({ type: "ready" });
      return;
    }
    if (session === null) throw new Error("g08 worker acted before model initialization");
    const input = event.data.input;
    const outputs = await session.run({
      obs: new ort.Tensor("float32", input.obs, [1, 45, 21, 21]),
      army_history: new ort.Tensor("float32", input.armyHistory, [1, 512]),
      land_history: new ort.Tensor("float32", input.landHistory, [1, 512]),
      penalty: new ort.Tensor("float32", input.penalty, [1, 4410]),
    });
    const outputName = session.outputNames[0]!;
    const tensor = outputs[outputName];
    if (tensor === undefined) throw new Error(`g08 graph did not emit ${outputName}`);
    const logits = tensor.data as Float32Array;
    let actionIndex = 0;
    for (let index = 1; index < logits.length; index += 1) {
      if (logits[index]! > logits[actionIndex]!) actionIndex = index;
    }
    post({ type: "decision", actionIndex, action: decodeAction(actionIndex), logits }, [logits.buffer]);
  } catch (error) {
    post({ type: "error", message: String(error) });
  }
};
