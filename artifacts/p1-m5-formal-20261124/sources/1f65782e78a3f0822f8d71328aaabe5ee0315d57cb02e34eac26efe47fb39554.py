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
        self.wait_residual = nn.Sequential(
            nn.Linear(52, 16), nn.SiLU(), nn.Linear(16, BUCKET_COUNTS[0])
        )
        nn.init.zeros_(self.wait_residual[-1].weight)
        nn.init.zeros_(self.wait_residual[-1].bias)

    def _features(self, observation):
        ticks = observation.book_ticks
        dtype = self.input_projection.weight.dtype
        shares = observation.book_shares.to(dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(-1) / present.sum(-1).clamp_min(1)
        price = ((ticks - mid[..., None, None]) / 20).clamp(-10, 10)
        price = price.masked_fill(ticks <= 0, 0)
        depth = (shares / 100).log1p()

        bid, ask = touch.unbind(-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(-1), 0).log1p()
        bid_depth, ask_depth = shares.sum(-1).unbind(-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        own = observation.is_own.reshape_as(spread)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        features = torch.cat((
            price.flatten(-2), depth.flatten(-2),
            torch.stack((spread, imbalance, own), dim=-1), previous.flatten(-2),
        ), dim=-1)

        bid_size, ask_size = shares[..., 0].unbind(-1)
        bid_log, ask_log = depth[..., 0].unbind(-1)
        touch_imbalance = (bid_size - ask_size) / (bid_size + ask_size).clamp_min(1)
        book = torch.stack((spread, bid_log, ask_log, touch_imbalance), dim=-1)
        return self.input_projection(features.to(dtype)).tanh(), book.to(dtype)

    def _context(self, observation, state, step):
        encoded, book = self._features(observation)
        prior = None if state is None else state.unsqueeze(0)
        out, hidden = self.gru(encoded.unsqueeze(1) if step else encoded, prior)
        if step:
            out = out[:, 0]
        return torch.cat((out, book.to(out.dtype)), dim=-1), hidden[0]

    def sequence_context(self, observation):
        return self._context(observation, None, False)[0]

    def step_context(self, observation, state):
        return self._context(observation, state, True)

    def head_logits(self, context, stage, prior):
        recurrent = context[..., :48]
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            recurrent = torch.cat((recurrent, prefix), dim=-1)
        logits = self.heads[stage](recurrent)
        if stage == 0:
            logits = logits + self.wait_residual(context)
        return logits
