from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """A ten-level book encoder and GRU with discrete conditional order heads.

    Adds a depth-gated imbalance interaction path so impact scales with
    signed order flow relative to available depth, rather than additively.
    """

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8) -> None:
        super().__init__()
        self.state_dim = state_dim
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        base_in = 43 + 4 * embedding_dim
        self.input_projection = nn.Linear(base_in, state_dim)
        # Historical orders have no own labels. Start with no untrained flag response.
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        # Depth-gated imbalance interaction path.
        self.imbalance_proj = nn.Linear(2, state_dim)
        self.depth_gate = nn.Linear(2, state_dim)
        self.interaction_out = nn.Linear(state_dim, state_dim)
        nn.init.zeros_(self.interaction_out.weight)
        nn.init.zeros_(self.interaction_out.bias)
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
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
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(dim=-1), 0).log1p()
        bid_depth, ask_depth = book_shares.sum(dim=-1).unbind(dim=-1)
        total_depth = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total_depth.clamp_min(1)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        features = torch.cat((price_feature.flatten(-2), depth_feature.flatten(-2),
                              torch.stack((spread, imbalance, observation.is_own), dim=-1),
                              previous.flatten(-2)), dim=-1)
        hidden = self.input_projection(
            features.to(self.input_projection.weight.dtype)).tanh()

        # Depth-gated imbalance interaction: signed flow times log-depth gate.
        logdepth = total_depth.clamp_min(1).log()
        depth_vec = torch.stack((imbalance, logdepth), dim=-1)
        signed = torch.tanh(self.imbalance_proj(depth_vec))
        gate = torch.sigmoid(self.depth_gate(depth_vec))
        interaction = self.interaction_out(signed * gate)
        return hidden + interaction

    def sequence_context(self, observation: StepObservation) -> Tensor:
        out, _ = self.gru(self._features(observation))
        return out

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        prior = None if state is None else state.unsqueeze(0)
        out, hidden = self.gru(self._features(observation).unsqueeze(1), prior)
        return out[:, 0], hidden[0]

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
