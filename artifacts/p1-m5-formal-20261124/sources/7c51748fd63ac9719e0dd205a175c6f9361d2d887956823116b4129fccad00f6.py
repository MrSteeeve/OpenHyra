import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)
from stochopia.neural.primitives import parallel_affine_scan


class CandidateLOBWorldModel(LOBWorldModel):
    def __init__(self) -> None:
        super().__init__()
        offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            offsets.append(offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(offsets), persistent=False)
        self.register_buffer("trace_decay", torch.tensor([0.8, 0.95, 0.99, 0.998]))
        # Action order: add bid/ask, cancel bid/ask, trade buy/sell.
        self.register_buffer("trade_sign", torch.tensor([0, 0, 0, 0, 1, -1]))
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), 8)
        self.input_projection = nn.Linear(75, 48)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(48, 48, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(48 + stage * 8, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        self.action_trace = nn.Linear(4, BUCKET_COUNTS[1], bias=False)
        nn.init.zeros_(self.action_trace.weight)

    def _features(self, observation: StepObservation) -> Tensor:
        dtype = self.input_projection.weight.dtype
        ticks = observation.book_ticks
        shares = observation.book_shares.to(dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        price = ((ticks - mid[..., None, None]) / 20).clamp(-10, 10)
        price = price.masked_fill(ticks <= 0, 0)
        depth = (shares / 100).log1p()
        bid, ask = touch.unbind(dim=-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(dim=-1), 0).log1p()
        bid_depth, ask_depth = shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        own = observation.is_own.reshape_as(spread)
        previous = self.token_embed(observation.previous_event.long() + self.token_offsets)
        features = torch.cat((
            price.flatten(-2), depth.flatten(-2),
            torch.stack((spread, imbalance, own), dim=-1), previous.flatten(-2),
        ), dim=-1)
        return self.input_projection(features.to(dtype)).tanh()

    def _run(
        self, observation: StepObservation, state: State, step: bool,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        features = self._features(observation)
        sign = self.trade_sign[observation.previous_event[..., 1].long()]
        if step:
            features = features.unsqueeze(1)
            sign = sign.unsqueeze(1)
        hidden, trace = (None, None) if state is None else state
        output, hidden = self.gru(features, None if hidden is None else hidden.unsqueeze(0))
        decay = self.trace_decay
        inputs = sign.to(decay.dtype).unsqueeze(-1) * (1 - decay)
        traces = parallel_affine_scan(decay, inputs, h0=trace)
        context = torch.cat((output, traces.to(output.dtype)), dim=-1)
        return context, (hidden[0], traces[:, -1])

    def sequence_context(self, observation: StepObservation) -> Tensor:
        return self._run(observation, None, False)[0]

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, State]:
        context, next_state = self._run(observation, state, True)
        return context[:, 0], next_state

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        base, traces = context.split((48, 4), dim=-1)
        if prior:
            tokens = torch.stack(prior, dim=-1).long() + self.token_offsets[:stage]
            base = torch.cat((base, self.token_embed(tokens).flatten(-2)), dim=-1)
        logits = self.heads[stage](base)
        return logits + self.action_trace(traces) if stage == 1 else logits
