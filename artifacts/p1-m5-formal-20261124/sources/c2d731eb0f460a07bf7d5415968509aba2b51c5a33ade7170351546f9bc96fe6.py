import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)
from stochopia.neural.primitives import affine_scan_step, parallel_affine_scan


class CandidateLOBWorldModel(LOBWorldModel):
    def __init__(self) -> None:
        super().__init__()
        offsets = [sum(BUCKET_COUNTS[:i]) for i in range(4)]
        self.register_buffer("token_offsets", torch.tensor(offsets), persistent=False)
        self.register_buffer(
            "trace_decay", (-1 / torch.tensor([4., 16., 64., 256., 1024.])).exp(),
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
        for head in self.heads:
            head.weight = nn.Parameter(
                torch.cat((head.weight.detach(), head.weight.new_zeros(head.out_features, 5)), dim=1)
            )
            head.in_features += 5

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

    def _trace_terms(self, observation: StepObservation, ref: Tensor) -> tuple[Tensor, Tensor]:
        action = observation.previous_event[..., 1]
        sign = (action == 4).to(ref.dtype) - (action == 5).to(ref.dtype)
        decay = self.trace_decay.to(ref)
        return decay, sign.unsqueeze(-1) * (1 - decay)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        out, _ = self.gru(self._features(observation))
        decay, inputs = self._trace_terms(observation, out)
        traces = parallel_affine_scan(decay, inputs)
        return torch.cat((out, traces), dim=-1)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, State]:
        hidden, traces = (None, None) if state is None else state
        out, hidden = self.gru(
            self._features(observation).unsqueeze(1),
            None if hidden is None else hidden.unsqueeze(0),
        )
        out = out[:, 0]
        decay, inputs = self._trace_terms(observation, out)
        if traces is None:
            traces = torch.zeros_like(inputs)
        traces = affine_scan_step(decay.expand_as(inputs), inputs, traces)
        return torch.cat((out, traces), dim=-1), (hidden[0], traces)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1).long() + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context[..., :48], prefix, context[..., 48:]), dim=-1)
        return self.heads[stage](context)
