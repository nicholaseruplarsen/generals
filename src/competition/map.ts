import { REAL_COMPETITION_MAPS } from "./maps.generated";
import type { CompetitionMap } from "./types";

/** Official 21×18 tournament map from live match #198316. */
export const TOURNAMENT_MAP: CompetitionMap = {
  rows: 21,
  cols: 18,
  mountains: [
    [0,4],[0,7],[0,11],[0,16],[1,4],[1,5],[1,7],[1,10],[1,16],[1,17],
    [2,3],[2,11],[2,17],[3,4],[3,6],[3,8],[3,9],[3,14],[4,0],[4,3],
    [4,5],[4,14],[4,17],[5,7],[5,14],[6,1],[6,4],[6,5],[6,17],[7,0],
    [7,7],[7,11],[7,15],[8,2],[8,5],[8,8],[9,9],[9,10],[10,6],[10,16],
    [11,5],[11,10],[11,12],[12,1],[12,6],[12,11],[12,16],[12,17],[13,2],[13,13],
    [14,3],[14,5],[14,8],[15,0],[15,4],[15,8],[15,12],[15,16],[15,17],[16,0],
    [16,2],[16,6],[16,7],[16,9],[16,10],[16,14],[17,8],[17,11],[18,0],[18,5],
    [18,8],[18,9],[18,13],[18,16],[18,17],[19,1],[19,3],[19,4],[19,17],[20,4],
    [20,6],[20,13],[20,14],
  ],
  generals: [[19, 14], [2, 8]],
};

export function randomCompetitionMap(): CompetitionMap {
  /** Select a real archived tournament map without procedural generation. */
  const randomValue = new Uint32Array(1);
  crypto.getRandomValues(randomValue);
  return REAL_COMPETITION_MAPS[randomValue[0]! % REAL_COMPETITION_MAPS.length]!;
}

export function competitionMapAt(index: number): CompetitionMap {
  /** Resolve the archived map referenced by an offline self-play trace. */
  const map = REAL_COMPETITION_MAPS[index];
  if (map === undefined) throw new Error(`Missing archived competition map ${index}`);
  return map;
}
