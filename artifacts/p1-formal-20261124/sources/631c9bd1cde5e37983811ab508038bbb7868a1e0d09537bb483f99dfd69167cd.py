from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Book encoder + GRU with slow EMA of context injected into order heads."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8,
                 slow_decay: float = 0.9) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.slow_decay = slow_decay
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets),
                             persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        self.input_projection = nn.Linear(43 + 4 * embedding_dim, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        # Shared low-rank projection of slow EMA context -> embedding_dim.
        self.slow_proj = nn.Linear(state_dim, embedding_dim)
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + stage * embedding_dim, count)
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
        spread = (ask - bid).clamp_min(0).masked_fill(
            ~present.all(dim=-1), 0).log1p()
        bid_depth, ask_depth = book_shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        features = torch.cat(
            (price_feature.flatten(-2), depth_feature.flatten(-2),
             torch.stack((spread, imbalance, observation.is_own), dim=-1),
             previous.flatten(-2)), dim=-1)
        return self.input_projection(
            features.to(self.input_projection.weight.dtype)).tanh()

    def sequence_context(self, observation: StepObservation) -> Tensor:
        out, _ = self.gru(self._features(observation))
        return out

    def step_context(self, observation: StepObservation,
                     state: State) -> tuple[Tensor, State]:
        # State carries (gru_hidden [B,state_dim], slow_ema [B,state_dim]).
        if state is None:
            prior_h = None
            prior_slow = None
        else:
            prior_h, prior_slow = state[0].unsqueeze(0), state[1]
        feats = self._features(observation).unsqueeze(1)
        out, hidden = self.gru(feats, prior_h)
        out = out[:, 0]
        if prior_slow is None:
            slow = out
        else:
            slow = self.slow_decay * prior_slow + (1 - self.slow_decay) * out
        return out, (hidden[0], slow)

    def _slow_from_sequence(self, context: Tensor) -> Tensor:
        # Causal EMA over T of context.
        decay = self.slow_decay
        ema = torch.empty_like(context)
        acc = None
        for t in range(context.shape[1]):
            x = context[:, t]
            acc = x if acc is None else decay * acc + (1 - decay) * x
            ema[:, t] = acc
        return ema

    def head_logits(self, context: Tensor, stage: int,
                    prior: tuple[Tensor, ...]) -> Tensor:
        # `context` is [..., state_dim]; prior tokens are the already-decoded
        # bucket ids for stages < stage. We need the slow EMA of context at the
        # current step. Both sequence and step paths must supply the *same*
        # slow state. To keep the canonical computation identical, we pass the
        # slow state alongside via a side-channel: heads receive a combined
        # context whose only distinction is fast vs. slow.
        raise RuntimeError("head_logits is invoked through _head_logits_with_slow")

    def _head_logits_with_slow(self, context: Tensor, slow: Tensor, stage: int,
                               prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        slow_embed = self.slow_proj(slow)
        return self.heads[stage](torch.cat((context, slow_embed), dim=-1))

    def forward_sequence(self, batch):
        from stochopia.neural.lob_interface import FiveTupleLogits
        obs = batch.observation
        context = self.sequence_context(obs)
        slow = self._slow_from_sequence(context)
        prior: list[Tensor] = []
        outs = []
        for stage in range(len(BUCKET_COUNTS)):
            logits = self._head_logits_with_slow(context, slow, stage, tuple(prior))
            outs.append(logits)
            tokens = logits.argmax(dim=-1)
            prior.append(tokens)
        return FiveTupleLogits(*outs)

    def forward_step(self, step_obs: StepObservation, hidden_state: State = None):
        from stochopia.neural.lob_interface import StepSampler
        context, state = self.step_context(step_obs, hidden_state)
        return StepSampler(self, context, step_obs, state), state
