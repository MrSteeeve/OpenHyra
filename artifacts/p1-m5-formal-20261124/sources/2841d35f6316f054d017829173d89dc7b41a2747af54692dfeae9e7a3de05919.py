from __future__ import annotations
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Book encoder + GRU with a decayed impact accumulator channel."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim

        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)

        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)

        # input: 43 book features + 4*emb previous tokens + 1 impact scalar
        in_dim = 43 + 4 * embedding_dim + 1
        self.input_projection = nn.Linear(in_dim, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])

        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)

        self.heads = nn.ModuleList(
            nn.Linear(state_dim + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )

        # Impact accumulator parameters.
        self.impact_decay_logit = nn.Parameter(torch.tensor(0.0))
        self.impact_gain = nn.Parameter(torch.tensor(1.0))

    # ------------------------------------------------------------------
    # feature construction
    # ------------------------------------------------------------------
    def _base_features(self, observation: StepObservation):
        """Return (linear_input_without_impact, impact_inst) both [...,].

        linear_input_without_impact: [..., in_dim-1]
        impact_inst: [...] raw signed pressure for the current step.
        """
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

        base = torch.cat(
            (
                price_feature.flatten(-2),
                depth_feature.flatten(-2),
                torch.stack((spread, imbalance, observation.is_own), dim=-1),
                previous.flatten(-2),
            ),
            dim=-1,
        )

        # Raw signed own-order pressure for impact accumulation.
        own = observation.is_own.to(base.dtype)
        impact_inst = (self.impact_gain * own * imbalance).tanh()

        return base, impact_inst

    # ------------------------------------------------------------------
    # sequence path
    # ------------------------------------------------------------------
    def sequence_context(self, observation: StepObservation) -> Tensor:
        base, impact_inst = self._base_features(observation)
        decay = torch.sigmoid(self.impact_decay_logit)
        T = base.shape[1]
        accum = torch.zeros(base.shape[:-1], dtype=base.dtype, device=base.device)
        seq = []
        for t in range(T):
            accum = decay * accum + impact_inst[:, t]
            seq.append(accum)
        accum_seq = torch.stack(seq, dim=1).unsqueeze(-1)  # [...,T,1]
        features = torch.cat((base, accum_seq), dim=-1)
        inp = self.input_projection(features).tanh()
        out, _ = self.gru(inp)
        return out

    # ------------------------------------------------------------------
    # step path (canonical equivalent computation)
    # ------------------------------------------------------------------
    def step_context(self, observation: StepObservation, state: State):
        base, impact_inst = self._base_features(observation)
        decay = torch.sigmoid(self.impact_decay_logit)

        if state is None:
            prev_hidden = None
            prev_accum = torch.zeros(base.shape[:-1], dtype=base.dtype, device=base.device)
        else:
            if isinstance(state, tuple):
                prev_hidden = state[0] if state[0] is not None else None
                prev_accum = state[1]
            else:
                prev_hidden = state
                prev_accum = torch.zeros(base.shape[:-1], dtype=base.dtype, device=base.device)

        accum = decay * prev_accum + impact_inst
        features = torch.cat((base, accum.unsqueeze(-1)), dim=-1)
        inp = self.input_projection(features).tanh().unsqueeze(1)
        prior = None if prev_hidden is None else prev_hidden.unsqueeze(0)
        out, hidden = self.gru(inp, prior)
        return out[:, 0], (hidden[0], accum)

    # ------------------------------------------------------------------
    # head
    # ------------------------------------------------------------------
    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
