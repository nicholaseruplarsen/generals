"""Torch policies over the (C, H, W) obs with invalid-action masking.

The flat action layout matches research.rl.obs_encode:
    idx = (row*W + col)*8 + direction*2 + split, and idx == H*W*8 is pass.
Channels 14..17 of the observation are the per-direction legality planes;
illegal moves get -1e9 logits so the sampled policy never plays them.

Architecture hyperparameters (channels/blocks/hidden) are exported alongside
the weights so research.rl.jax_policy can rebuild the identical network in JAX.
"""
import torch
import torch.nn as nn

MASK_CH = slice(14, 18)


class ConvPolicy(nn.Module):
    def __init__(self, env, channels: int = 64, blocks: int = 3, hidden_size: int = 128):
        super().__init__()
        c, h, w = env.single_observation_space.shape
        self.arch = dict(channels=channels, blocks=blocks, hidden_size=hidden_size,
                         in_channels=c, height=h, width=w)
        self.hidden_size = hidden_size

        convs, prev = [], c
        for _ in range(blocks):
            convs.append(nn.Conv2d(prev, channels, 3, padding=1))
            prev = channels
        self.convs = nn.ModuleList(convs)
        self.move_head = nn.Conv2d(channels, 8, 1)
        self.fc = nn.Linear(channels, hidden_size)
        self.pass_head = nn.Linear(hidden_size, 1)
        self.value_head = nn.Linear(hidden_size, 1)

        nn.init.orthogonal_(self.move_head.weight, 0.01)
        nn.init.zeros_(self.move_head.bias)
        nn.init.orthogonal_(self.pass_head.weight, 0.01)
        nn.init.zeros_(self.pass_head.bias)
        nn.init.orthogonal_(self.value_head.weight, 1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

    def forward_eval(self, observations, state=None):
        x = observations.float()
        mask = x[:, MASK_CH] > 0.5                       # (B, 4, H, W)
        for conv in self.convs:
            x = torch.relu(conv(x))
        b = x.shape[0]

        move_logits = self.move_head(x).permute(0, 2, 3, 1).reshape(b, -1)  # (B, H*W*8)
        move_mask = mask.permute(0, 2, 3, 1).repeat_interleave(2, dim=-1).reshape(b, -1)

        pooled = x.mean(dim=(2, 3))
        hid = torch.nn.functional.gelu(self.fc(pooled))
        pass_logit = self.pass_head(hid)                 # (B, 1)

        logits = torch.cat([move_logits, pass_logit], dim=1)
        full_mask = torch.cat([move_mask, torch.ones_like(pass_logit, dtype=torch.bool)], dim=1)
        logits = logits.masked_fill(~full_mask, -1e9)
        value = self.value_head(hid)
        return logits, value


class RecurrentConvPolicy(nn.Module):
    """CNN encoder followed by one LSTM layer and actor/critic heads.

    This implements PufferLib's recurrent policy contract directly. Flattening
    the final convolutional map (rather than globally pooling it) preserves
    spatial identity before the recurrent bottleneck. The recurrent state is
    cleared on episode boundaries during rollout collection.
    """

    def __init__(self, env, channels: int = 64, blocks: int = 3,
                 hidden_size: int = 256, lstm_input_size: int = 256,
                 spatial_decode: bool = False):
        super().__init__()
        c, h, w = env.single_observation_space.shape
        self.arch = dict(
            recurrent=True, channels=channels, blocks=blocks,
            hidden_size=hidden_size, lstm_input_size=lstm_input_size,
            spatial_decode=spatial_decode,
            in_channels=c, height=h, width=w,
        )
        self.hidden_size = hidden_size
        self.spatial_decode = spatial_decode
        self.is_continuous = False

        convs, prev = [], c
        for _ in range(blocks):
            convs.append(nn.Conv2d(prev, channels, 3, padding=1))
            prev = channels
        self.convs = nn.ModuleList(convs)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * h * w, lstm_input_size),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(lstm_input_size, hidden_size, num_layers=1)
        self.cell = nn.LSTMCell(lstm_input_size, hidden_size)
        # Share parameters between the fast single-step cell and batched LSTM.
        self.cell.weight_ih = self.lstm.weight_ih_l0
        self.cell.weight_hh = self.lstm.weight_hh_l0
        self.cell.bias_ih = self.lstm.bias_ih_l0
        self.cell.bias_hh = self.lstm.bias_hh_l0
        if spatial_decode:
            # Decode moves spatially like the feed-forward winner: broadcast
            # the LSTM hidden back onto the conv map so per-cell move logits
            # keep spatial structure instead of squeezing 8*H*W+1 actions out
            # of the hidden-size bottleneck.
            self.hidden_proj = nn.Linear(hidden_size, channels)
            self.fuse = nn.Conv2d(channels, channels, 3, padding=1)
            self.move_head = nn.Conv2d(channels, 8, 1)
            self.pass_head = nn.Linear(hidden_size, 1)
        else:
            self.action_head = nn.Linear(hidden_size, h * w * 8 + 1)
        self.value_head = nn.Linear(hidden_size, 1)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name and param.ndim >= 2:
                nn.init.orthogonal_(param)
        if spatial_decode:
            nn.init.orthogonal_(self.move_head.weight.flatten(1), 0.01)
            nn.init.orthogonal_(self.pass_head.weight, 0.01)
        else:
            nn.init.orthogonal_(self.action_head.weight, 0.01)
        nn.init.orthogonal_(self.value_head.weight, 1.0)

    def _encode(self, observations):
        """-> (conv feature map, flat LSTM input)."""
        x = observations.float()
        for conv in self.convs:
            x = torch.relu(conv(x))
        return x, self.encoder(x)

    @staticmethod
    def _mask(observations):
        mask = observations[..., MASK_CH, :, :] > 0.5
        prefix = mask.shape[:-3]
        move = mask.movedim(-3, -1).repeat_interleave(2, dim=-1)
        move = move.reshape(*prefix, -1)
        return torch.cat(
            [move, torch.ones(*prefix, 1, dtype=torch.bool, device=move.device)],
            dim=-1,
        )

    def _decode(self, hidden, feat_map, observations):
        """hidden (B, hidden), feat_map (B, channels, H, W) -> logits, value."""
        if self.spatial_decode:
            fused = torch.relu(feat_map + self.hidden_proj(hidden)[:, :, None, None])
            fused = torch.relu(self.fuse(fused))
            move = self.move_head(fused)                       # (B, 8, H, W)
            # Flat layout must match obs_encode: (row*W+col)*8 + dir*2 + split.
            move = move.permute(0, 2, 3, 1).reshape(move.shape[0], -1)
            logits = torch.cat([move, self.pass_head(hidden)], dim=-1)
        else:
            logits = self.action_head(hidden)
        logits = logits.masked_fill(~self._mask(observations), -1e9)
        return logits, self.value_head(hidden)

    def forward_eval(self, observations, state):
        feat_map, encoded = self._encode(observations)
        h, c = state.get("lstm_h"), state.get("lstm_c")
        if h is not None:
            # PufferLib provides the previous transition's done flag.
            done = state.get("done")
            if done is not None:
                keep = (~done.bool()).to(h.dtype).unsqueeze(-1)
                h, c = h * keep, c * keep
            hidden, c = self.cell(encoded, (h, c))
        else:
            hidden, c = self.cell(encoded)
        state["lstm_h"], state["lstm_c"] = hidden, c
        return self._decode(hidden, feat_map, observations)

    def forward(self, observations, state):
        space_ndim = 3
        if observations.ndim == space_ndim + 1:
            observations = observations[:, None]
        if observations.ndim != space_ndim + 2:
            raise ValueError(f"invalid recurrent observation shape {observations.shape}")
        batch, time = observations.shape[:2]
        flat_obs = observations.reshape(batch * time, *observations.shape[-3:])
        feat_map, encoded = self._encode(flat_obs)
        encoded = encoded.reshape(batch, time, -1).transpose(0, 1)

        # BPTT must clear the carry at episode boundaries inside a segment,
        # exactly as forward_eval does via state["done"] during rollout;
        # otherwise recomputed log-probs diverge from behavior-policy outputs
        # and PPO ratios are wrong after every in-segment reset. PuffeRL gives
        # training no done mask, but the encoded timestep plane (channel 13,
        # timestep/500) is exactly zero only on a fresh episode's first
        # observation, so the reset points are recoverable from the obs.
        reset = (observations[:, :, 13, 0, 0] < 1e-6).transpose(0, 1)  # (time, batch)
        h, c = state.get("lstm_h"), state.get("lstm_c")
        if h is None:
            h = encoded.new_zeros(batch, self.hidden_size)
            c = encoded.new_zeros(batch, self.hidden_size)
        else:
            h, c = h.reshape(batch, -1), c.reshape(batch, -1)
        steps = []
        for t in range(time):
            keep = (~reset[t]).to(h.dtype).unsqueeze(-1)
            h, c = self.cell(encoded[t], (h * keep, c * keep))
            steps.append(h)
        hidden = torch.stack(steps).transpose(0, 1)
        state["lstm_h"], state["lstm_c"] = h.detach(), c.detach()

        hidden_flat = hidden.reshape(batch * time, -1)
        logits, values = self._decode(hidden_flat, feat_map, flat_obs)
        return logits, values.reshape(batch, time)
