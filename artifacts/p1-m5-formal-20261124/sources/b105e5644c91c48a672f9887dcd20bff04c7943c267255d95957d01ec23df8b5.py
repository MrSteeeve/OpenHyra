from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """GRU with a two-timescale EMA side-channel feeding only the heads."""

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
        # Two-timescale EMA side-channel.
        self.ema_logit = nn.Parameter(torch.tensor([0.5, 2.0]))
        # Heads consume [context ; ema_side]. Bias only biases; EMA columns zero-init.
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + 2 * state_dim + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        for head in self.heads:
            head.weight.data[:, state_dim:3 * state_dim] = 0.0

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

    def _ema_gates(self) -> Tensor:
        # Two scalar gates from log-sigmoid-able logits, broadcast to [state_dim].
        g = torch.sigmoid(self.ema_logit)  # (2,)
        return g

    def _sequence_ema(self, feats: Tensor) -> Tensor:
        # feats [B,T,C], returns [B,T,2C]
        g = self._ema_gates()  # (2,)
        B, T, C = feats.shape
        outs = []
        for k in range(2):
            gk = g[k]
            h = feats.new_zeros(B, C)
            acc = []
            for t in range(T):
                h = gk * h + (1.0 - gk) * feats[:, t]
                acc.append(h)
            outs.append(torch.stack(acc, dim=1))
        return torch.cat(outs, dim=-1)

    def _step_ema(self, feats: Tensor, state: State) -> tuple[Tensor, Tensor]:
        # feats [B,C]; state is (gru_h[B,C], ema_side[B,2C]).
        g = self._ema_gates()
        gru_h, ema_side = state
        new_side = []
        for k in range(2):
            gk = g[k]
            h = gk * ema_side[:, k * feats.shape[-1]:(k + 1) * feats.shape[-1]] + (1.0 - gk) * feats
            new_side.append(h)
        return torch.cat(new_side, dim=-1), (gru_h, torch.cat(new_side, dim=-1))

    def _pack_state(self, gru_h: Tensor, ema_side: Tensor) -> Tensor:
        return torch.cat((gru_h, ema_side), dim=-1)

    def _unpack_state(self, state: State) -> tuple[Tensor, Tensor]:
        if state is None:
            return None, None
        C = self.state_dim
        return state[:, :C], state[:, C:]

    def sequence_context(self, observation: StepObservation) -> Tensor:
        feats = self._features(observation)
        out, _ = self.gru(feats)
        ema = self._sequence_ema(feats)
        return torch.cat((out, ema), dim=-1)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        feats = self._features(observation)
        gru_h, ema_side = self._unpack_state(state)
        prior = None if gru_h is None else gru_h.unsqueeze(0)
        out, hidden = self.gru(feats.unsqueeze(1), prior)
        gru_out = out[:, 0]
        if gru_h is None:
            ema_prev = feats.new_zeros(feats.shape[0], 2 * self.state_dim)
        else:
            ema_prev = ema_side
        _, (new_gru, new_ema) = self._step_ema(feats, (hidden[0], ema_prev))
        del new_ema
        new_ema_val = self._step_ema_compute(feats, ema_prev)
        ctx = torch.cat((gru_out, new_ema_val), dim=-1)
        return ctx, self._pack_state(hidden[0], new_ema_val)

    def _step_ema_compute(self, feats: Tensor, ema_prev: Tensor) -> Tensor:
        g = self._ema_gates()
        C = feats.shape[-1]
        new_side = []
        for k in range(2):
            gk = g[k]
            h = gk * ema_prev[:, k * C:(k + 1) * C] + (1.0 - gk) * feats
            new_side.append(h)
        return torch.cat(new_side, dim=-1)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
