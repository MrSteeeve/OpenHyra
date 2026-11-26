import torch
from torch import nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel,
)


class CandidateLOBWorldModel(LOBWorldModel):
    def __init__(self):
        super().__init__()
        offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            offsets.append(offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), 8)
        self.input_projection = nn.Linear(75, 48)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(48, 48, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(48 + 8 * stage, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        self.volume_residual = nn.Sequential(
            nn.Linear(74, 8), nn.SiLU(), nn.Linear(8, BUCKET_COUNTS[3])
        )
        nn.init.zeros_(self.volume_residual[-1].weight)
        nn.init.zeros_(self.volume_residual[-1].bias)

    def _encode(self, observation):
        ticks = observation.book_ticks
        dtype = self.input_projection.weight.dtype
        shares = observation.book_shares.to(dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(-1) / present.sum(-1).clamp_min(1)
        relative = ticks - mid[..., None, None]
        occupied = (ticks > 0) & (shares > 0)
        lookup_ticks = relative.masked_fill(~occupied, float("inf"))
        price = (relative / 20).clamp(-10, 10).masked_fill(ticks <= 0, 0)
        depth = (shares / 100).log1p()
        bid, ask = touch.unbind(-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(-1), 0).log1p()
        bid_depth, ask_depth = shares.sum(-1).unbind(-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        own = observation.is_own.reshape_as(imbalance)
        previous = self.token_embed(observation.previous_event.long() + self.token_offsets)
        features = torch.cat((
            price.flatten(-2), depth.flatten(-2),
            torch.stack((spread, imbalance, own), -1), previous.flatten(-2)
        ), -1)
        return self.input_projection(features.to(dtype)).tanh(), depth, lookup_ticks

    def _context(self, observation, state, single):
        x, depth, ticks = self._encode(observation)
        out, hidden = self.gru(
            x.unsqueeze(1) if single else x,
            None if state is None else state.unsqueeze(0),
        )
        if single:
            out = out[:, 0]
        context = torch.cat((out, depth.flatten(-2), ticks.flatten(-2)), -1)
        return context, hidden[0]

    def sequence_context(self, observation):
        return self._context(observation, None, False)[0]

    def step_context(self, observation, state):
        return self._context(observation, state, True)

    def head_logits(self, context, stage, prior):
        base = context[..., :48].to(self.token_embed.weight.dtype)
        if prior:
            tokens = torch.stack(prior, -1).long() + self.token_offsets[:stage]
            base = torch.cat((base, self.token_embed(tokens).flatten(-2)), -1)
        logits = self.heads[stage](base)
        if stage != 3:
            return logits

        _, action, price = prior
        # Action order: add bid/ask, cancel bid/ask, trade buy/sell.
        active = (action >= 2) & (action <= 5)
        ask_side = (action == 3) | (action == 4)
        depth = context[..., 48:68].unflatten(-1, (2, 10))
        ticks = context[..., 68:88].unflatten(-1, (2, 10))
        side = ask_side.long()[..., None, None].expand(*action.shape, 1, 10)
        depth = depth.gather(-2, side).squeeze(-2)
        ticks = ticks.gather(-2, side).squeeze(-2)

        # Signed offsets are centered at bucket 12; bucket 25 is non-exact.
        offset = price - 12
        match = (ticks.round() == offset[..., None]) & (price[..., None] < 25)
        unique = match.sum(-1) == 1
        available = depth.masked_fill(~match, 0).sum(-1).masked_fill(~unique, 0)
        lookup = torch.stack((available, (~unique).to(available.dtype)), -1)
        residual = self.volume_residual(torch.cat((base, lookup.to(base.dtype)), -1))
        return logits + residual.masked_fill(~active[..., None], 0)
