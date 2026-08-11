"""Build a small browser map pool from real Generals Competition replays."""

from __future__ import annotations

import json
import random
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TypedDict, cast


BASE_URL = "https://www.generals.bot/api/leaderboard"
POOL_SIZE = 32


class CompetitionMap(TypedDict):
    """Map fields consumed by the TypeScript competition engine."""

    rows: int
    cols: int
    mountains: list[list[int]]
    generals: list[list[int]]


def request_json(parameters: dict[str, str]) -> dict[str, object]:
    """Fetch one public competition API payload."""
    url = f"{BASE_URL}?{urllib.parse.urlencode(parameters)}"
    request = urllib.request.Request(url, headers={"User-Agent": "generals-research-site/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return cast(dict[str, object], json.load(response))


def replay_map(payload: dict[str, object]) -> CompetitionMap:
    """Extract the immutable map header from one replay payload."""
    dims = cast(dict[str, int], payload["dims"])
    return {
        "rows": dims["rows"],
        "cols": dims["cols"],
        "mountains": cast(list[list[int]], payload["mountains"]),
        "generals": cast(list[list[int]], payload["generals"]),
    }


def main() -> None:
    """Sample unique real maps and emit a typed, compact TypeScript asset."""
    listing = request_json({"matches": "1", "player": "Nicholas"})
    matches = cast(list[dict[str, object]], listing["matches"])
    random.Random(20260812).shuffle(matches)
    maps: list[CompetitionMap] = []
    fingerprints: set[str] = set()
    for match in matches:
        payload = request_json({"replay": str(match["id"])})
        map_data = replay_map(payload)
        fingerprint = json.dumps(map_data, sort_keys=True, separators=(",", ":"))
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        maps.append(map_data)
        if len(maps) == POOL_SIZE:
            break
    if len(maps) != POOL_SIZE:
        raise RuntimeError(f"Expected {POOL_SIZE} unique maps, received {len(maps)}")

    destination = Path(__file__).resolve().parents[1] / "src" / "competition" / "maps.generated.ts"
    encoded = json.dumps(maps, separators=(",", ":"))
    destination.write_text(
        'import type { CompetitionMap } from "./types";\n\n'
        "/** Real maps sampled from Nicholas's public competition replays. */\n"
        f"export const REAL_COMPETITION_MAPS = {encoded} as const satisfies readonly CompetitionMap[];\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
