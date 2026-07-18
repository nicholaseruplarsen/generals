// Full-game replay parity: GeneralsBot over the champion ONNX model must
// reproduce the fixture's logits (legal entries), values, and argmax actions
// for all 418 steps, threading the LSTM carry from zeros without resets.

import * as ort from "onnxruntime-node";
import { describe, expect, it } from "vitest";
import { NUM_ACTIONS } from "../src/engine/types";
import { GeneralsBot, ortSessionLike } from "../src/model/infer";
import { loadPolicyFixture, toObservation } from "./policy-fixture";

const fixture = loadPolicyFixture();
const MODEL_PATH = new URL("../public/models/a100-i-draw2.onnx", import.meta.url).pathname;

describe("GeneralsBot full-game replay (champion-vs-hunter)", () => {
  it(
    `matches logits/value/argmax on all ${fixture.numSteps} steps`,
    { timeout: 180_000 },
    async () => {
      const session = await ort.InferenceSession.create(MODEL_PATH);
      const bot = await GeneralsBot.load(ortSessionLike(ort, session));
      bot.reset(); // carryInit: zeros

      let maxLogitDiff = 0;
      let maxValueDiff = 0;
      let argmaxMatches = 0;

      for (const [i, step] of fixture.steps.entries()) {
        const { actionIdx, value } = await bot.act(toObservation(step.obs0));
        const logits = bot.lastLogits!;
        expect(logits.length).toBe(NUM_ACTIONS);

        // Logit parity on legal entries only (illegal are pinned at -1e9).
        for (let j = 0; j < NUM_ACTIONS; j++) {
          if (step.legalMask[j] === 0) continue;
          const d = Math.abs(logits[j]! - step.logits[j]!);
          if (d > maxLogitDiff) maxLogitDiff = d;
        }
        maxValueDiff = Math.max(maxValueDiff, Math.abs(value - step.value));

        expect(actionIdx, `step ${i} argmax`).toBe(step.actionIdx);
        argmaxMatches++;
      }

      expect(maxLogitDiff, "max |logit diff| on legal entries").toBeLessThanOrEqual(1e-3);
      expect(maxValueDiff, "max |value diff|").toBeLessThanOrEqual(1e-3);
      expect(argmaxMatches).toBe(fixture.numSteps);
      console.log(
        `parity: argmax ${argmaxMatches}/${fixture.numSteps}, ` +
          `max logit diff ${maxLogitDiff.toExponential(3)}, max value diff ${maxValueDiff.toExponential(3)}`,
      );
    },
  );
});
