from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Ten-level book encoder with a GRU plus a slow leaky context for impact heads."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8,
                 slow_dim: int = 16) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.slow_dim = slow_dim
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        self.input_projection = nn.Linear(43 + 4 * embedding_dim, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)

        # Slow leaky context: learned per-dimension leak logit, slow leak init.
        # leak = sigmoid(leak_logit).  Init to sigmoid(-3)~0.047 => decay ~0.95.
        self.slow_leak_logit = nn.Parameter(torch.full((state_dim,), -3.0))
        self.slow_projection = nn.Linear(state_dim, slow_dim)
        nn.init.zeros_(self.slow_projection.weight)
        nn.init.zeros_(self.slow_projection.bias)

        self.slow_dim = slow_dim
        # Only the impact-relevant heads read the slow context: stages 1 (aggr)
        # and 3 (impact) get the slow term prepended.
        self._slow_stages = (1, 3)
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + stage * embedding_dim
                      + (slow_dim if stage in self._slow_stages else 0), count)
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

    def _leak(self) -> Tensor:
        return torch.sigmoid(self.slow_leak_logit)

    def _slow_init(self, batch_shape) -> Tensor:
        return torch.zeros(*batch_shape, self.state_dim)

    def _combine(self, h: Tensor) -> Tensor:
        return self.slow_projection(h)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        features = self._features(observation)
        out, _ = self.gru(features)
        leak = self._leak()
        B, T, _ = out.shape
        c = self._slow_init((B,))
        slowdowns = []
        for t in range(T):
            c = c + leak * (out[:, t] - c)
            slowdowns.append(c)
        slow = torch.stack(slowdowns, dim=1)
        return torch.cat((out, self._combine(slow)), dim=-1)

    def step_context(self, observation: StepObservation, state: State):
        # state: (gru_h, slow_c) tuple, each [B, state_dim]
        if state is None:
            prior = None
            c = None
        else:
            h_state, c_state = state
            prior = h_state.unsqueeze(0)
            c = c_state
        feats = self._features(observation)
        out, hidden = self.gru(feats.unsqueeze(1), prior)
        h = out[:, 0]
        leak = self._leak()
        if c is None:
            c = self._slow_init(h.shape[:-1])
        c = c + leak * (h - c)
        slow = self._combine(c)
        ctx = torch.cat((h, slow), dim=-1)
        return ctx, (hidden[0], c)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if stage in self._slow_stages:
            h = context[..., :self.state_dim]
            slow = context[..., self.state_dim:]
        else:
            h = context
            slow = None
        parts = [h]
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            parts.append(prefix)
        if slow is not None:
            parts.append(slow)
        return self.heads[stage](torch.cat(parts, dim=-1))

    def forward_sequence(self, batch):
        logits = super().forward_sequence(batch)
        return logits

    def forward_step(self, step_obs, hidden_state=None):
        # Ensure the StepSampler loop sees the same slow-context state carried by
        # step_context.  The inherited forward_step using step_context is correct,
        # but we route through it explicitly so the (h, c) tuple is what carries.
        sampler = self._make_sampler(step_obs)
        ctx, new_state = self.step_context(step_obs, hidden_state)
        return _BoundSampler(self, ctx, new_state, step_obs), new_state

    # ``forward_step`` may be overridden by the base class reliance; we keep the
    # inherited forward_sequence but implement a small sampler shim so the step
    # path reuses step_context directly.
    def _make_sampler(self, step_obs):
        return None


class _BoundSampler:
    """Minimal StepSampler-compatible wrapper used by forward_step."""

    def __init__(self, model: CandidateLOBWorldModel, context: Tensor,
                 state, observation: StepObservation) -> None:
        self._model = model
        self._context = context
        self._state = state
        self._obs = observation

    def conditioned(self, targets: Tensor):
        prior = targets[..., :0]
        stage_outputs = []
        stages = [0, 1, 2, 3]
        for stage in stages:
            prior_tuple = tuple(targets[..., i] for i in range(stage))
            stage_outputs.append(self._model.head_logits(self._context, stage, prior_tuple))
        return tuple(stage_outputs)

    def sample(self, generator: torch.Generator | None = None) -> Tensor:
        samples = []
        stages = [0, 1, 2, 3]
        for stage in stages:
            prior_tuple = tuple(samples[i] for i in range(stage))
            logits = self._model.head_logits(self._context, stage, prior_tuple)
            probs = torch.softmax(logits, dim=-1)
            flat = probs.reshape(-1, probs.shape[-1])
            idx = torch.multinomial(flat, 1, generator=generator).reshape(
                probs.shape[:-1])
            samples.append(idx)
        return torch.stack(samples, dim=-1)
