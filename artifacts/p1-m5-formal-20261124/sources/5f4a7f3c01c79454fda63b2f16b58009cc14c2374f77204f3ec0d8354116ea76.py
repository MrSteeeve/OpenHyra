from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Book encoder + GRU with a signed-flow impact trace feeding stage heads."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8) -> None:
        super().__init__()
        self.state_dim = state_dim
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        self.input_projection = nn.Linear(43 + 4 * embedding_dim, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        # Impact trace: bounded scalar from context, decayed accumulator.
        self.impact_readout = nn.Linear(state_dim, 1)
        nn.init.zeros_(self.impact_readout.weight)
        nn.init.zeros_(self.impact_readout.bias)
        self.impact_decay_logit = nn.Parameter(torch.tensor(2.0))
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + 2 + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )

    def _features(self, observation: StepObservation) -> Tensor:
        ticks = observation.book_ticks
        book_shares = observation.book_shares.to(self.input_projection.weight.dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        relative = (ticks - mid[..., None, None]) / 20
        price_feature = relative.clamp(-10, 10).masked_fill(ticks <= 0, 0)
        depth_feature = (book_shares / 100).log1p()

        bid, ask = touch.unbind(dim=-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(dim=-1), 0).log1p()
        bid_depth, ask_depth = book_shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        features = torch.cat((price_feature.flatten(-2), depth_feature.flatten(-2),
                              torch.stack((spread, imbalance, observation.is_own), dim=-1),
                              previous.flatten(-2)), dim=-1)
        return self.input_projection(features.to(self.input_projection.weight.dtype)).tanh()

    def sequence_context(self, observation: StepObservation) -> Tensor:
        out, _ = self.gru(self._features(observation))
        # Impact trace scan along time.
        drive = self.impact_readout(out).squeeze(-1)  # [B,T]
        decay = torch.sigmoid(self.impact_decay_logit)
        traces = []
        acc = torch.zeros_like(drive[:, 0])
        for t in range(drive.shape[1]):
            acc = decay * acc + drive[:, t]
            traces.append(acc)
        trace = torch.stack(traces, dim=1)  # [B,T]
        fast = trace - torch.cat((torch.zeros_like(trace[:, :1]), trace[:, :-1]), dim=1)
        return torch.cat((out, trace.unsqueeze(-1), fast.unsqueeze(-1)), dim=-1)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, State]:
        if state is None or isinstance(state, Tensor) and state.dim() != 2:
            h_prev = None
            trace = None
        else:
            if isinstance(state, tuple):
                h_prev, trace = state[0], state[1]
            else:
                h_prev, trace = state, None
        prior = None if h_prev is None else h_prev.unsqueeze(0)
        out, hidden = self.gru(self._features(observation).unsqueeze(1), prior)
        out = out[:, 0]
        hidden = hidden[0]
        drive = self.impact_readout(out).squeeze(-1)  # [B]
        decay = torch.sigmoid(self.impact_decay_logit)
        if trace is None:
            trace = drive
        else:
            trace = decay * trace + drive
        fast = trace - drive if False else drive  # step-local difference -> drive
        ctx = torch.cat((out, trace.unsqueeze(-1), fast.unsqueeze(-1)), dim=-1)
        return ctx, (hidden, trace)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
