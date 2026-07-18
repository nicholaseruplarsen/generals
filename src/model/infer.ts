// ONNX inference wrapper — feeds {obs, h, c}, reads {logits, value, h_out,
// c_out}, and threads the (1,256) LSTM carry across steps of a game.

import { H, NUM_ACTIONS, W } from "../engine/types";
import type { Action, Observation } from "../engine/types";
import { decodeAction, encode, flatMask, NUM_CHANNELS } from "./encode";

export const HIDDEN_SIZE = 256;

/** Minimal float32 tensor shape common to onnxruntime-node and onnxruntime-web. */
export interface OrtTensorLike {
  readonly data: Float32Array;
  readonly dims: readonly number[];
}

/**
 * What GeneralsBot needs from an ONNX runtime: run a session, and build
 * float32 tensors with *that* runtime's own Tensor class (each runtime
 * rejects foreign tensor objects). Adapt any onnxruntime-node /
 * onnxruntime-web module + InferenceSession pair with `ortSessionLike`.
 */
export interface OrtSessionLike {
  run(feeds: Record<string, OrtTensorLike>): Promise<Record<string, OrtTensorLike>>;
  tensor(data: Float32Array, dims: readonly number[]): OrtTensorLike;
}

interface OrtModuleLike {
  Tensor: new (type: "float32", data: Float32Array, dims: readonly number[]) => unknown;
}

interface RawSession {
  run(feeds: Record<string, unknown>): Promise<Record<string, unknown>>;
}

/** Adapt an onnxruntime-node / onnxruntime-web module + session to OrtSessionLike. */
export function ortSessionLike(ort: OrtModuleLike, session: RawSession): OrtSessionLike {
  return {
    tensor: (data, dims) => new ort.Tensor("float32", data, dims) as OrtTensorLike,
    run: async (feeds) => (await session.run(feeds)) as Record<string, OrtTensorLike>,
  };
}

export interface ActResult {
  actionIdx: number;
  action: Action;
  value: number;
}

export class GeneralsBot {
  private readonly h = new Float32Array(HIDDEN_SIZE);
  private readonly c = new Float32Array(HIDDEN_SIZE);
  /** Raw model logits from the most recent act() call; null before the first. */
  lastLogits: Float32Array | null = null;

  private constructor(private readonly session: OrtSessionLike) {}

  static async load(session: OrtSessionLike): Promise<GeneralsBot> {
    return new GeneralsBot(session);
  }

  /** Zero the (1,256) h and c carries — call once at game start. */
  reset(): void {
    this.h.fill(0);
    this.c.fill(0);
  }

  async act(obs: Observation): Promise<ActResult> {
    const s = this.session;
    const results = await s.run({
      obs: s.tensor(encode(obs), [1, NUM_CHANNELS, H, W]),
      h: s.tensor(this.h, [1, HIDDEN_SIZE]),
      c: s.tensor(this.c, [1, HIDDEN_SIZE]),
    });
    const logits = results["logits"]!.data;
    const value = results["value"]!.data[0]!;
    // Store the new carry (inputs were copied by run(), so this is safe).
    this.h.set(results["h_out"]!.data);
    this.c.set(results["c_out"]!.data);
    this.lastLogits = logits;
    // Argmax over legal actions — first max wins, matching jnp.argmax. The
    // model masks illegal logits to -1e9 internally; apply flatMask again as
    // a belt-and-braces step.
    const mask = flatMask(obs);
    let actionIdx = NUM_ACTIONS - 1;
    let best = -Infinity;
    for (let i = 0; i < NUM_ACTIONS; i++) {
      if (mask[i] === 0) continue;
      const v = logits[i]!;
      if (v > best) {
        best = v;
        actionIdx = i;
      }
    }
    return { actionIdx, action: decodeAction(actionIdx), value };
  }
}
