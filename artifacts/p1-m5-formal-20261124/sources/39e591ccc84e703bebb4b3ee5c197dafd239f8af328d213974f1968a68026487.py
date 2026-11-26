from __future__ import annotations
import math
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Ten-level book encoder + GRU with an impact-memory integrator and cross-stage heads."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8,
                 impact_dim: int = 8) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        self.impact_dim = impact_dim
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        # impact magnitude bucket embedding (previous event stage-0 token acts as magnitude proxy)
        self.magnitude_bins = max(1, BUCKET_COUNTS[0])
        self.magnitude_embed = nn.Embedding(self.magnitude_bins, impact_dim)
        self.input_projection = nn.Linear(43 + 4 * embedding_dim, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        # Learnable own-flag gain initialized near-zero but nonzero (allows own impact to drive state).
        self.own_gain = nn.Parameter(torch.tensor(0.05))
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        # Impact memory: scalar decayed leaky integrator (real scalars in state).
        # log_decay_init ~ log(0.9) via sigmoid(logit).
        self.impact_decay_logit = nn.Parameter(torch.tensor(2.2))  # sigmoid ~0.9
        self.impact_gain = nn.Parameter(torch.tensor(0.1))
        # Heads see context + cross-stage prefix embeddings + impact slot + magnitude emb.
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + (2 * impact_dim) + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )

    # ---- impact memory plumbing -------------------------------------------------
    def _impact_state(self) -> Tensor:
        return torch.sigmoid(self.impact_decay_logit)

    def _signed_own_size(self, observation: StepObservation) -> Tensor:
        """Per-step signed own-trade size proxy (real scalar per prefix)."""
        own = observation.is_own.to(self.own_gain.dtype)
        # Use share magnitude of the touch-level own order as impact proxy.
        book_shares = observation.book_shares.to(self.own_gain.dtype)
        # total shares at touch across sides, normalized
        size = book_shares[..., 0, :].sum(dim=-1) / 100.0
        return own * size * self.own_gain

    def _impact_bucket(self, observation: StepObservation) -> Tensor:
        prev = observation.previous_event
        mag = prev[..., 0].clamp(min=0, max=self.magnitude_bins - 1)
        return self.magnitude_embed(mag)

    # ---- shared feature construction -------------------------------------------
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
        own = observation.is_own.to(self.input_projection.weight.dtype) * self.own_gain
        features = torch.cat((price_feature.flatten(-2), depth_feature.flatten(-2),
                              torch.stack((spread, imbalance, own), dim=-1),
                              previous.flatten(-2)), dim=-1)
        return self.input_projection(features.to(self.input_projection.weight.dtype)).tanh()

    # ---- sequence path ---------------------------------------------------------
    def sequence_context(self, observation: StepObservation) -> Tensor:
        feats = self._features(observation)
        out, _ = self.gru(feats)
        # Run impact integrator in parallel along time using a simple cumsum-based scan.
        own_sz = self._signed_own_size(observation) * self.impact_gain
        decay = self._impact_state()
        # leaky integrator: h_t = decay * h_{t-1} + own_sz_t
        B, T = own_sz.shape
        # closed-form via discounted cumsum
        steps = torch.arange(T, device=own_sz.device, dtype=own_sz.dtype)
        powers = decay.pow(steps)  # [T]
        scaled = own_sz / powers.clamp_min(1e-8)
        cum = torch.cumsum(scaled, dim=1) * powers
        return torch.cat((out, cum.unsqueeze(-1)), dim=-1)

    def step_context(self, observation: StepObservation, state: State):
        feats = self._features(observation)
        if state is None:
            hidden = None
            impact = torch.zeros(feats.shape[0], device=feats.device, dtype=feats.dtype)
        else:
            if isinstance(state, tuple):
                hidden, impact = state[0], state[1]
            else:
                hidden, impact = state, torch.zeros(feats.shape[0], device=feats.device,
                                                    dtype=feats.dtype)
            hidden = hidden.unsqueeze(0)
        out, new_hidden = self.gru(feats.unsqueeze(1), hidden)
        own_sz = self._signed_own_size(observation) * self.impact_gain
        new_impact = self._impact_state() * impact + own_sz
        return out[:, 0], (new_hidden[0], new_impact)

    # ---- heads -----------------------------------------------------------------
    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        # Split context back into GRU state and impact scalar.
        gru_part = context[..., : self.state_dim]
        impact = context[..., self.state_dim: self.state_dim + 1]
        # impact slot: broadcast to impact_dim
        impact_slot = impact.expand(*impact.shape[:-1], self.impact_dim)
        # magnitude embedding from the first-stage prefix (event magnitude proxy)
        if prior:
            mag = prior[0].clamp(min=0, max=self.magnitude_bins - 1)
            mag_emb = self.magnitude_embed(mag)
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            full = torch.cat((gru_part, impact_slot, mag_emb, prefix), dim=-1)
        else:
            mag_emb = torch.zeros(*context.shape[:-1], self.impact_dim,
                                  device=context.device, dtype=context.dtype)
            full = torch.cat((gru_part, impact_slot, mag_emb), dim=-1)
        return self.heads[stage](full)
