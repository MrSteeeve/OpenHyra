import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)
from stochopia.neural.primitives import affine_scan_step, parallel_affine_scan


class CandidateLOBWorldModel(LOBWorldModel):
    def __init__(self) -> None:
        super().__init__()
        offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            offsets.append(offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(offsets), persistent=False)
        self.register_buffer(
            "flow_decay",
            torch.tensor([2.0 ** (-1.0 / h) for h in (2, 8, 32, 128)]),
            persistent=False,
        )
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), 8)
        self.input_projection = nn.Linear(75, 48)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(48, 48, batch_first=True)
        self.flow_residual = nn.Sequential(nn.Linear(8, 32), nn.Tanh(), nn.Linear(32, 48))
        nn.init.zeros_(self.flow_residual[-1].weight)
        nn.init.zeros_(self.flow_residual[-1].bias)
        self.heads = nn.ModuleList(
            nn.Linear(48 + stage * 8, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )

    def _encode(self, obs: StepObservation) -> tuple[Tensor, Tensor, Tensor]:
        dtype = self.input_projection.weight.dtype
        ticks = obs.book_ticks
        shares = obs.book_shares.to(dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        price = ((ticks - mid[..., None, None]) / 20).clamp(-10, 10)
        price = price.masked_fill(ticks <= 0, 0).to(dtype)
        depth = (shares / 100).log1p()
        bid, ask = touch.unbind(-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(-1), 0).log1p()
        bid_depth, ask_depth = shares.sum(-1).unbind(-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        own = obs.is_own.reshape_as(imbalance)
        previous = self.token_embed(obs.previous_event + self.token_offsets)
        features = torch.cat((
            price.flatten(-2), depth.flatten(-2),
            torch.stack((spread, imbalance, own), dim=-1).to(dtype),
            previous.flatten(-2),
        ), dim=-1)
        action = obs.previous_event[..., 1]
        sign = (action == 4).to(dtype) - (action == 5).to(dtype)
        flow_input = sign.unsqueeze(-1) * (1 - self.flow_decay)
        liquidity = torch.cat((depth[..., 0], present.to(dtype)), dim=-1)
        return self.input_projection(features).tanh(), flow_input, liquidity

    def _context(self, hidden: Tensor, traces: Tensor, liquidity: Tensor) -> Tensor:
        return hidden + self.flow_residual(torch.cat((traces, liquidity), dim=-1))

    def sequence_context(self, observation: StepObservation) -> Tensor:
        features, flow_input, liquidity = self._encode(observation)
        hidden, _ = self.gru(features)
        traces = parallel_affine_scan(self.flow_decay, flow_input)
        return self._context(hidden, traces, liquidity)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, State]:
        features, flow_input, liquidity = self._encode(observation)
        if state is None:
            prior_hidden = None
            prior_traces = torch.zeros_like(flow_input)
        else:
            prior_hidden, prior_traces = state
            prior_hidden = prior_hidden.unsqueeze(0)
        _, hidden = self.gru(features.unsqueeze(1), prior_hidden)
        hidden = hidden.squeeze(0)
        traces = affine_scan_step(self.flow_decay, flow_input, prior_traces)
        return self._context(hidden, traces, liquidity), (hidden, traces)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            context = torch.cat((context, self.token_embed(tokens).flatten(-2)), dim=-1)
        return self.heads[stage](context)
