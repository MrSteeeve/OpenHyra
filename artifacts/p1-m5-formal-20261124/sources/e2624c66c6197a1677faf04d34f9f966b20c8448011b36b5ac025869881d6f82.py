import torch
from torch import nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Parent model with a training-loop hook for wait calibration.

    The inherited forward methods do not consume training metadata or add
    losses. The trainer must add wait_calibration_loss to its likelihood
    objective, supplying the rollout decoder's bucket means and a mask of
    complete, unpadded, single-session 64-event blocks.
    """

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
        own = observation.is_own.reshape_as(imbalance)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        features = torch.cat((
            price.flatten(-2), depth.flatten(-2),
            torch.stack((spread, imbalance, own), dim=-1),
            previous.flatten(-2),
        ), dim=-1)
        return self.input_projection(features.to(dtype)).tanh()

    def _run(self, features, state):
        hidden = None if state is None else state.unsqueeze(0)
        out, hidden = self.gru(features, hidden)
        return out, hidden[0]

    def sequence_context(self, observation):
        return self._run(self._features(observation), None)[0]

    def step_context(self, observation, state):
        out, state = self._run(self._features(observation).unsqueeze(1), state)
        return out[:, 0], state

    def head_logits(self, context, stage, prior):
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            context = torch.cat((context, self.token_embed(tokens).flatten(-2)), dim=-1)
        return self.heads[stage](context)

    def wait_calibration_loss(self, context, targets, bucket_durations, block_mask):
        """Return the weighted auxiliary term, with gradients only to head 0.

        context: [B,T,48]; targets: [B,T,4].
        bucket_durations: [16] decoder means in base time units, including zero.
        block_mask: bool [B,T//64], true only for eligible contiguous blocks
        starting at positions 0, 64, ... . For session-aligned blocks, callers
        may first pack eligible context and target slices into [N,64,...].
        """
        b, t, c = context.shape
        n = t // 64
        blocks = context.detach()[:, :64 * n].reshape(b, n, 64, c)[block_mask]
        wait = targets[:, :64 * n, 0].reshape(b, n, 64)[block_mask].long()
        logits = self.head_logits(blocks, 0, ())
        if blocks.shape[0] == 0:
            return logits.sum()
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        durations = torch.as_tensor(
            bucket_durations, device=logits.device, dtype=logits.dtype
        ).detach()
        predicted = (logits.softmax(-1) * durations).sum(dim=(-1, -2))
        observed = durations[wait].sum(-1)
        return 0.1 * (predicted.log1p() - observed.log1p()).square().mean()
