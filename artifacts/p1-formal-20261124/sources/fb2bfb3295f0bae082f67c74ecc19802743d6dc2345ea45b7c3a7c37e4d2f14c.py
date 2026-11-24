from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """GRU with a leaky-integrated slow context feeding the first two heads."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8, slow_decay: float = 0.9) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.slow_decay = slow_decay
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        self.input_projection = nn.Linear(43 + 4 * embedding_dim, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        self.slow_projection = nn.Linear(state_dim, state_dim, bias=False)
        self.heads = nn.ModuleList(
            nn.Linear(
                state_dim + stage * embedding_dim + (state_dim if stage < 2 else 0),
                count,
            )
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
        features = torch.cat(
            (
                price_feature.flatten(-2),
                depth_feature.flatten(-2),
                torch.stack((spread, imbalance, observation.is_own), dim=-1),
                previous.flatten(-2),
            ),
            dim=-1,
        )
        return self.input_projection(features.to(self.input_projection.weight.dtype)).tanh()

    def _slow_scan(self, fast: Tensor, slow0: Tensor | None) -> Tensor:
        decay = self.slow_decay
        proj = self.slow_projection(fast).tanh()
        b, t, c = proj.shape
        if slow0 is None:
            slow0 = torch.zeros(b, c, dtype=proj.dtype, device=proj.device)
        out = torch.empty_like(proj)
        h = slow0
        for i in range(t):
            h = decay * h + (1.0 - decay) * proj[:, i]
            out[:, i] = h
        return out

    def _slow_scan_step(self, fast: Tensor, slow_prev: Tensor | None) -> Tensor:
        decay = self.slow_decay
        proj = self.slow_projection(fast).tanh()
        if slow_prev is None:
            slow_prev = torch.zeros_like(proj)
        return decay * slow_prev + (1.0 - decay) * proj

    def sequence_context(self, observation: StepObservation) -> Tensor:
        fast, _ = self.gru(self._features(observation))
        slow = self._slow_scan(fast, None)
        return torch.cat((fast, slow), dim=-1)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, State]:
        if state is None:
            hidden_prev = None
            slow_prev = None
        else:
            hidden_prev = state[0].unsqueeze(0)
            slow_prev = state[1]
        fast_seq, hidden = self.gru(self._features(observation).unsqueeze(1), hidden_prev)
        fast = fast_seq[:, 0]
        slow = self._slow_scan_step(fast, slow_prev)
        return torch.cat((fast, slow), dim=-1), (hidden[0], slow)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        s = self.state_dim
        fast = context[..., :s]
        slow = context[..., s:]
        parts = [fast]
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            parts.append(self.token_embed(tokens).flatten(-2))
        if stage < 2:
            parts.append(slow)
        return self.heads[stage](torch.cat(parts, dim=-1))
