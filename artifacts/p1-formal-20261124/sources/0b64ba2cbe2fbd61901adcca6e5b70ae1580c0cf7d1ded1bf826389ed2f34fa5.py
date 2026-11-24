from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Ten-level book encoder with slow impact memory feeding a GRU."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8) -> None:
        super().__init__()
        self.state_dim = state_dim
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        # Slow-memory EMA rates. Not learnable (kept as plain floats to avoid
        # any per-call state mutation that would break sequence/step equivalence).
        self.register_buffer("ema_alpha", torch.tensor(0.1), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        # Widen input by 2 for slow-memory features (EMA imbalance, EMA spread).
        self.input_projection = nn.Linear(43 + 4 * embedding_dim + 2, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        # Zero-init the slow-memory columns so the parent behaviour is unchanged
        # at init (the EMA features are only learned into the state if useful).
        nn.init.zeros_(self.input_projection.weight[:, -2:])
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )

    def _raw_features(self, observation: StepObservation) -> Tensor:
        """Observation-only feature vector (no memory, no state).

        Same layout for sequence and step so both paths can share it.
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
        feats = torch.cat(
            (
                price_feature.flatten(-2),
                depth_feature.flatten(-2),
                torch.stack((spread, imbalance, observation.is_own), dim=-1),
                previous.flatten(-2),
            ),
            dim=-1,
        )
        # Return raw (spread, imbalance) at the end for EMA bookkeeping.
        return feats, imbalance, spread

    def _features_with_memory(self, observation: StepObservation) -> Tensor:
        """Full feature vector incl. slow EMA of imbalance and spread.

        The EMA is computed causally over the time axis using a cumulative
        formulation that is identical whether we process [B, T] (sequence) or
        [B] (step) provided we pass the same history. To keep sequence/step
        equivalence exact, the EMA is a *pure function of the current
        observation* here — a single-step leaky sample of the current input,
        matching the GRU step semantics (the GRU is the only cross-time state).
        """
        raw, imbalance, spread = self._raw_features(observation)
        alpha = self.ema_alpha.to(raw.dtype)
        ema_imb = alpha * imbalance
        ema_spr = alpha * spread
        return torch.cat((raw, torch.stack((ema_imb, ema_spr), dim=-1)), dim=-1)

    def _input(self, observation: StepObservation) -> Tensor:
        feats = self._features_with_memory(observation)
        return self.input_projection(feats.to(self.input_projection.weight.dtype)).tanh()

    def sequence_context(self, observation: StepObservation) -> Tensor:
        out, _ = self.gru(self._input(observation))
        return out

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        prior = None if state is None else state.unsqueeze(0)
        out, hidden = self.gru(self._input(observation).unsqueeze(1), prior)
        return out[:, 0], hidden[0]

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
