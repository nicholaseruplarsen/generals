// Parity of src/model/encode.ts against the policy fixture's golden tensors.

import { describe, expect, it } from "vitest";
import { H, NUM_ACTIONS, W } from "../src/engine/types";
import { encode, flatMask, NUM_CHANNELS } from "../src/model/encode";
import { loadPolicyFixture, toObservation } from "./policy-fixture";

const fixture = loadPolicyFixture();

describe("encode", () => {
  it("matches fixture `encoded` for the first 8 steps within 1e-6", () => {
    for (const [i, step] of fixture.steps.slice(0, 8).entries()) {
      expect(step.encoded, `step ${i} should carry an encoded field`).toBeDefined();
      const want = step.encoded!;
      expect(want.length).toBe(NUM_CHANNELS * H * W);
      const got = encode(toObservation(step.obs0));
      expect(got.length).toBe(want.length);
      let maxDiff = 0;
      for (let j = 0; j < want.length; j++) {
        const d = Math.abs(got[j]! - want[j]!);
        if (d > maxDiff) maxDiff = d;
      }
      expect(maxDiff, `step ${i} max |diff|`).toBeLessThanOrEqual(1e-6);
    }
  });

  it("flatMask matches fixture `legalMask` for every step", () => {
    for (const [i, step] of fixture.steps.entries()) {
      const got = flatMask(toObservation(step.obs0));
      expect(got.length).toBe(NUM_ACTIONS);
      expect(Array.from(got), `step ${i}`).toEqual(step.legalMask);
    }
  });
});
