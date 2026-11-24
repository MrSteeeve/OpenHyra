import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)
from stochopia.neural.primitives import parallel_affine_scan, affine_scan_step


class CandidateLOBWorldModel(LOBWorldModel):
    def __init__(self) -> None:
        super().__init__()
        offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            offsets.append(offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(offsets), persistent=False)
        # Action buckets: bid/ask adds, bid/ask cancels, buy/sell executions.
        self.register_buffer(
            "execution_sign", torch.tensor([0., 0., 0., 0., 1., -1.]),
            persistent=False,
        )
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), 8)
        self.input_projection = nn.Linear(75, 48)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(48, 48, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(48 + 8 * stage, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )

    def _inputs(self, obs: StepObservation) -> tuple[Tensor, Tensor]:
        ticks = obs.book_ticks
        dtype = self.input_projection.weight.dtype
        shares = obs.book_shares.to(dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(-1) / present.sum(-1).clamp_min(1)
        price = ((ticks - mid[..., None, None]) / 20).clamp(-10, 10)
        price = price.masked_fill(ticks <= 0, 0)
        depth = (shares / 100).log1p()
        bid, ask = touch.unbind(-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(-1), 0).log1p()
        bd, ad = shares.sum(-1).unbind(-1)
        imbalance = (bd - ad) / (bd + ad).clamp_min(1)
        own = obs.is_own.reshape(spread.shape).to(dtype)
        previous = self.token_embed(obs.previous_event + self.token_offsets)
        features = torch.cat((
            price.flatten(-2), depth.flatten(-2),
            torch.stack((spread, imbalance, own), -1), previous.flatten(-2),
        ), -1)
        sign = self.execution_sign[obs.previous_event[..., 1]]
        drive = (0.05 * own.clamp(0, 1) * sign).unsqueeze(-1)
        return self.input_projection(features.to(dtype)).tanh(), drive

    def sequence_context(self, observation: StepObservation) -> Tensor:
        features, drive = self._inputs(observation)
        out, _ = self.gru(features)
        trace = parallel_affine_scan(drive.new_full((1,), 0.95), drive)
        return torch.cat((out, trace), -1)

    def step_context(
        self, observation: StepObservation, state: State
    ) -> tuple[Tensor, Tensor]:
        features, drive = self._inputs(observation)
        if state is None:
            hidden = None
            trace = torch.zeros_like(drive)
        else:
            hidden = state[..., :48].unsqueeze(0).contiguous()
            trace = state[..., 48:]
        _, hidden = self.gru(features.unsqueeze(1), hidden)
        trace = affine_scan_step(drive.new_full((1,), 0.95), drive, trace)
        context = torch.cat((hidden[0], trace), -1)
        return context, context

    def head_logits(
        self, context: Tensor, stage: int, prior: tuple[Tensor, ...]
    ) -> Tensor:
        features = context[..., :48]
        if prior:
            tokens = torch.stack(prior, -1) + self.token_offsets[:stage]
            features = torch.cat((features, self.token_embed(tokens).flatten(-2)), -1)
        logits = self.heads[stage](features)
        if stage == 1:
            logits = logits + 2 * context[..., 48:] * self.execution_sign
        return logits
