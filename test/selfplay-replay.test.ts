import { describe, expect, it } from "vitest";

import { createState, step } from "../src/competition/game";
import { competitionMapAt } from "../src/competition/map";
import { SELFPLAY_REPLAYS } from "../src/competition/selfplay.generated";


describe("offline g08 self-play", () => {
  it("reaches each recorded terminal result exactly", () => {
    for (const replay of SELFPLAY_REPLAYS) {
      let state = createState(competitionMapAt(replay.mapIndex), false);
      for (const actions of replay.actions) state = step(state, actions);
      expect(state.turn).toBe(replay.actions.length);
      expect(state.winner).toBe(replay.winner);
    }
  });
});
