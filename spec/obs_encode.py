"""Shared observation encoding: Observation -> (C, H, W) float tensor + action mask.

This module is the single source of truth for the neural policy's input and
action space. The training env (research.rl.vecenv) and the eval-time JAX
policy (research.rl.jax_policy) both call these functions, so a policy trained
in torch sees exactly the same tensors when replayed in JAX.

Action space (Discrete):
    idx in [0, H*W*8):  cell = idx // 8, d = (idx % 8) // 2, split = idx % 2
    idx == H*W*8:       pass
Engine action layout: [pass, row, col, direction, split],
directions 0=up 1=down 2=left 3=right (game.DIRECTIONS).

All functions are pure/jit-safe and operate on a single Observation;
vmap for batches.
"""
import jax.numpy as jnp

from generals.core.observation import Observation

NUM_CHANNELS = 18  # 14 observation planes + 4 move-validity planes


def num_actions(h: int, w: int) -> int:
    return h * w * 8 + 1


def action_mask(obs: Observation) -> jnp.ndarray:
    """(4, H, W) bool — move from (i,j) in direction d is legal.

    Legal = source owned with army > 1, destination in bounds and not a
    visible mountain / structure in fog (same passability the scripted
    agents use).
    """
    movable = obs.owned_cells & (obs.armies > 1)
    blocked = obs.mountains | obs.structures_in_fog
    ok = ~blocked

    # dest_ok[d, i, j] == "cell (i,j) + DIRECTIONS[d] is in bounds and ok",
    # e.g. for d=0 (up) source (i,j) needs ok[i-1, j].
    up = jnp.pad(ok[:-1, :], ((1, 0), (0, 0)))     # d=0 (row-1)
    down = jnp.pad(ok[1:, :], ((0, 1), (0, 0)))    # d=1 (row+1)
    left = jnp.pad(ok[:, :-1], ((0, 0), (1, 0)))   # d=2 (col-1)
    right = jnp.pad(ok[:, 1:], ((0, 0), (0, 1)))   # d=3 (col+1)
    dest_ok = jnp.stack([up, down, left, right])
    return movable[None] & dest_ok


def encode(obs: Observation) -> jnp.ndarray:
    """(NUM_CHANNELS, H, W) float32 policy input."""
    h, w = obs.armies.shape
    f = lambda x: x.astype(jnp.float32)
    area = jnp.float32(h * w)
    scalar = lambda v: jnp.full((h, w), v, jnp.float32)
    mask = action_mask(obs)
    return jnp.stack(
        [
            jnp.log1p(f(obs.armies)) * 0.2,
            f(obs.generals),
            f(obs.cities),
            f(obs.mountains),
            f(obs.neutral_cells),
            f(obs.owned_cells),
            f(obs.opponent_cells),
            f(obs.fog_cells),
            f(obs.structures_in_fog),
            scalar(f(obs.owned_land_count) / area),
            scalar(jnp.log1p(f(obs.owned_army_count)) * 0.1),
            scalar(f(obs.opponent_land_count) / area),
            scalar(jnp.log1p(f(obs.opponent_army_count)) * 0.1),
            scalar(f(obs.timestep) / 500.0),
            f(mask[0]),
            f(mask[1]),
            f(mask[2]),
            f(mask[3]),
        ]
    )


def decode_action(idx: jnp.ndarray, h: int, w: int) -> jnp.ndarray:
    """Discrete index -> engine action [pass, row, col, direction, split]."""
    idx = idx.astype(jnp.int32)
    is_pass = idx >= h * w * 8
    cell = idx // 8
    return jnp.array(
        [
            is_pass.astype(jnp.int32),
            jnp.where(is_pass, 0, cell // w),
            jnp.where(is_pass, 0, cell % w),
            jnp.where(is_pass, 0, (idx % 8) // 2),
            jnp.where(is_pass, 0, idx % 2),
        ],
        dtype=jnp.int32,
    )


def flat_mask(obs: Observation) -> jnp.ndarray:
    """(num_actions,) bool mask over the flat Discrete space (pass always legal)."""
    m = action_mask(obs)                        # (4, H, W)
    per_cell = jnp.transpose(m, (1, 2, 0))      # (H, W, 4) — cell-major, dir-minor
    moves = jnp.repeat(per_cell.reshape(-1), 2)  # (H*W*8,) — split copies each move
    return jnp.concatenate([moves, jnp.ones((1,), bool)])
