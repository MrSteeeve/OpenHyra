from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """A ten-level book encoder and GRU with discrete conditional order heads.

    Adds a leaky slow-context accumulator over the GRU output, injected into the
    stage-0 head via a zero-initialized residual so initial logits are unchanged.
    """

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
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        # Slow-context accumulator: EMA with a learned (sigmoid-parameterized) decay.
        self.slow_decay_logit = nn.Parameter(torch.tensor(2.0))  # sigmoid ~0.88
        self.slow_residual = nn.Linear(state_dim, state_dim)
        nn.init.zeros_(self.slow_residual.weight)
        nn.init.zeros_(self.slow_residual.bias)

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

    def _slow_scan(self, gru_out: Tensor, h0: Tensor | None = None) -> Tensor:
        # Leaky EMA over time: z_t = (1-a) z_{t-1} + a * gru_out_t, a = sigmoid(logit).
        a = torch.sigmoid(self.slow_decay_logit)
        z = None
        outs = []
        for t in range(gru_out.shape[1]):
            x = gru_out[:, t]
            z = a * x if z is None else (1.0 - a) * z + a * x
            outs.append(z)
        return torch.stack(outs, dim=1)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        gru_out, _ = self.gru(self._features(observation))
        slow = self._slow_scan(gru_out)
        # stage-0 head sees context + slow residual; heads 1+ see only gru_out.
        stage0_input = gru_out + self.slow_residual(slow)
        return torch.cat((stage0_input.unsqueeze(0), gru_out.unsqueeze(0)), dim=0)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, State]:
        if state is None:
            prior = None
            slow_prev = None
        else:
            prior = state[0].unsqueeze(0)
            slow_prev = state[1]
        x = self._features(observation).unsqueeze(1)
        gru_out, hidden = self.gru(x, prior)
        h_t = hidden[0]
        a = torch.sigmoid(self.slow_decay_logit)
        g = gru_out[:, 0]
        if slow_prev is None:
            z = a * g
        else:
            z = (1.0 - a) * slow_prev + a * g
        stage0_input = g + self.slow_residual(z)
        ctx = torch.cat((stage0_input, g), dim=-1)
        return ctx, (h_t, z)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if context.shape[-1] == 2 * self.state_dim:
            context = context[..., :self.state_dim] if stage == 0 else context[..., self.state_dim:]
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)

    def forward_sequence(self, batch):
        observation = batch.observation
        ctx = self.sequence_context(observation)  # [2, B, T, D]
        # Reorder to [B, T, 2, D] to split per stage.
        ctx = ctx.permute(1, 2, 0, 3)
        outputs = []
        prior: tuple[Tensor, ...] = ()
        for stage in range(len(BUCKET_COUNTS)):
            logits = self.head_logits(ctx[..., stage, :], stage, prior)
            outputs.append(logits)
            prior = prior + (logits.argmax(dim=-1),)
        return tuple(outputs)
