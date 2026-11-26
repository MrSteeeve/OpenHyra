from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Ten-level book encoder, GRU, impact-memory scalars, and conditional heads."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8,
                 impact_buckets: int = 4) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.impact_buckets = impact_buckets
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        self.impact_embed = nn.Embedding(impact_buckets, embedding_dim)
        # Base features: 40 (price+depth) + 3 (spread,imbalance,is_own) = 43
        # + 4*embedding_dim (previous tokens) + embedding_dim (impact bucket)
        self.input_projection = nn.Linear(43 + 4 * embedding_dim + embedding_dim, state_dim)
        # Learnable scale on own-flag feature (near-zero init but parametric).
        self.own_scale = nn.Parameter(torch.tensor(0.01))
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        # Heads receive state + stage * embedding_dim (prefix) + embedding_dim (impact).
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + stage * embedding_dim + embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        # Impact leaky-integration decay (real scalar in State).
        self.impact_decay_logit = nn.Parameter(torch.tensor(0.0))

    # ---- impact-memory state helpers ----
    def _impact_decay(self) -> Tensor:
        return torch.sigmoid(self.impact_decay_logit)

    def _signed_size_bucket(self, observation: StepObservation) -> Tensor:
        # previous_event last column is size bucket index (0..BUCKET_COUNTS[3]-1)
        # Sign from is_own vs side of book? Use is_own as magnitude proxy indicator;
        # sign from sign of (previous size) with own flag; we bucket the magnitude.
        size_bucket = observation.previous_event[..., 3]  # [..., B]
        own = observation.is_own
        # Signed own-trade magnitude proxy: own * (bucket - mean)/norm
        # Continuous signed drive in [-1, 1]-ish.
        max_b = float(BUCKET_COUNTS[3] - 1)
        size = (size_bucket.to(torch.float32) / max_b).clamp(0.0, 1.0)
        drive = own.to(torch.float32) * size
        return drive

    def _impact_bucket(self, impact: Tensor) -> Tensor:
        b = self.impact_buckets
        # impact in (-1, 1); map to bucket index
        idx = ((impact.clamp(-1.0, 1.0) + 1.0) * 0.5 * (b - 1)).round().long()
        return idx.clamp(0, b - 1)

    def _impact_scan(self, observation: StepObservation) -> Tensor:
        drive = self._signed_size_bucket(observation)  # [..., T]
        decay = self._impact_decay()
        T = drive.shape[-1]
        out = torch.empty_like(drive)
        h = torch.zeros_like(drive[..., 0])
        for t in range(T):
            h = decay * h + drive[..., t]
            out[..., t] = h
        return out

    def _features(self, observation: StepObservation, impact: Tensor) -> Tensor:
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
        own = observation.is_own.to(self.input_projection.weight.dtype)
        own_scaled = own * self.own_scale
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        impact_idx = self._impact_bucket(impact)
        impact_emb = self.impact_embed(impact_idx)
        features = torch.cat((price_feature.flatten(-2), depth_feature.flatten(-2),
                              torch.stack((spread, imbalance, own_scaled), dim=-1),
                              previous.flatten(-2), impact_emb), dim=-1)
        return self.input_projection(
            features.to(self.input_projection.weight.dtype)).tanh()

    def sequence_context(self, observation: StepObservation) -> Tensor:
        impact = self._impact_scan(observation)
        out, _ = self.gru(self._features(observation, impact))
        return out

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        # State: (gru_hidden [B, state_dim], impact [B])
        if state is None:
            h_prev = None
            impact_prev = observation.book_ticks.new_zeros(observation.is_own.shape, 1).squeeze(-1)
        else:
            h_prev, impact_prev = state
            h_prev = h_prev.unsqueeze(0)
        drive = self._signed_size_bucket(observation)[..., 0]
        impact = self._impact_decay() * impact_prev + drive
        feats = self._features(observation, impact.unsqueeze(-1))
        out, hidden = self.gru(feats.unsqueeze(1), h_prev)
        return out[:, 0], (hidden[0], impact)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        # Embed impact from context via zero-coupling: heads use a learned impact-embed
        # look-up driven by a linear read of the context.
        # Simpler: append a stage-aware prefix only; impact info already in context.
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
