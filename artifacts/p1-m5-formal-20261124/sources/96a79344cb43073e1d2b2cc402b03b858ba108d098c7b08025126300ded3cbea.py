from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """GRU encoder with a signed transient-memory decay path feeding stage heads."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        self.input_projection = nn.Linear(43 + 4 * embedding_dim, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        # Signed transient memory: (transient sign, decayed magnitude) from GRU output.
        self.transient_proj = nn.Linear(state_dim, 2)
        with torch.no_grad():
            self.transient_proj.bias.fill_(0.0)
            # Negative gate bias -> sigmoid ~0.3, fast transient decay.
            self.transient_proj.weight.mul_(0.05)
            self.transient_proj.bias[1] = -0.85
        # Head-side correction: remove transient signed-flow from stage logits.
        self.transient_correct = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.transient_correct.weight.fill_(0.0)
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        self._stage_scale = nn.Parameter(torch.linspace(0.0, 1.0, len(BUCKET_COUNTS)))

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
        features = self._features(observation)
        out, _ = self.gru(features)
        return self._augment_with_transient(out)

    def _augment_with_transient(self, out: Tensor) -> Tensor:
        # Maintain a decaying signed transient memory over time via affine scan.
        raw = self.transient_proj(out)  # [..., 2]
        sign_obs = torch.tanh(raw[..., 0])
        gate = torch.sigmoid(raw[..., 1])  # decay factor per step
        # Running signed memory m_t = sign_obs_t + gate_t * m_{t-1}.
        B, T, _ = out.shape
        mem = torch.empty_like(sign_obs)
        h = torch.zeros_like(sign_obs[:, 0])
        for t in range(T):
            h = sign_obs[:, t] + gate[:, t] * h
            mem[:, t] = h
        transient = torch.stack((mem, mem.abs()), dim=-1)  # [B,T,2]
        return torch.cat((out, transient), dim=-1)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        prior_gru = None
        prior_mem = None
        if state is not None:
            if isinstance(state, tuple):
                prior_gru, prior_mem = state
            else:
                prior_gru = state
        prior = None if prior_gru is None else prior_gru.unsqueeze(0)
        features = self._features(observation).unsqueeze(1)
        out, hidden = self.gru(features, prior)
        out = out[:, 0]
        raw = self.transient_proj(out)
        sign_obs = torch.tanh(raw[..., 0])
        gate = torch.sigmoid(raw[..., 1])
        if prior_mem is None:
            prior_mem = torch.zeros_like(sign_obs)
        mem = sign_obs + gate * prior_mem
        transient = torch.stack((mem, mem.abs()), dim=-1)
        ctx = torch.cat((out, transient), dim=-1)
        return ctx, (hidden[0], mem)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        core = context[..., : self.state_dim]
        transient = context[..., self.state_dim:]
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            core = torch.cat((core, prefix), dim=-1)
        logits = self.heads[stage](core)
        correction = self.transient_correct(transient) * self._stage_scale[stage]
        return logits - correction
