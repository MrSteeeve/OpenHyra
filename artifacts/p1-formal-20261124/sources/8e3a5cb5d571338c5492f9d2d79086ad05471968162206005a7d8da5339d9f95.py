import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    def __init__(self) -> None:
        super().__init__()
        offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            offsets.append(offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), 8)
        self.input_projection = nn.Linear(75, 48)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(48, 48, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(48 + 8 * stage, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        base_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if base_count + 192 <= 28000:
            self.queue_projection = nn.Linear(4, 48, bias=False)
            nn.init.zeros_(self.queue_projection.weight)
        else:
            self.queue_projection = None
            self.queue_gain = nn.Parameter(torch.zeros(()))
        assert sum(p.numel() for p in self.parameters() if p.requires_grad) <= 28000

    def _encode(self, obs: StepObservation) -> tuple[Tensor, Tensor]:
        ticks = obs.book_ticks
        shares = obs.book_shares.to(self.input_projection.weight.dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(-1) / present.sum(-1).clamp_min(1)
        price = ((ticks - mid[..., None, None]) / 20).clamp(-10, 10)
        price = price.masked_fill(ticks <= 0, 0)
        depth = (shares / 100).log1p()
        bid, ask = touch.unbind(-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(-1), 0).log1p()
        bid_depth, ask_depth = shares.sum(-1).unbind(-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        own = obs.is_own.reshape_as(imbalance)
        previous = self.token_embed(obs.previous_event.long() + self.token_offsets)
        features = torch.cat((
            price.flatten(-2), depth.flatten(-2),
            torch.stack((spread, imbalance, own), dim=-1), previous.flatten(-2),
        ), dim=-1).to(shares.dtype)

        bid_queue, ask_queue = shares[..., :2].masked_fill(
            ticks[..., :2] <= 0, 0
        ).unbind(-2)
        total = bid_queue + ask_queue
        queues = torch.cat((
            (bid_queue - ask_queue) / total.clamp_min(1),
            (total / 100).log1p(),
        ), dim=-1)
        if self.queue_projection is None:
            residual = torch.nn.functional.pad(self.queue_gain * queues, (0, 44))
        else:
            residual = self.queue_projection(queues)
        return self.input_projection(features).tanh(), residual.tanh()

    def _run(
        self, obs: StepObservation, state: State, *, step: bool
    ) -> tuple[Tensor, Tensor]:
        inputs, residual = self._encode(obs)
        if step:
            inputs = inputs.unsqueeze(1)
        out, hidden = self.gru(inputs, None if state is None else state.unsqueeze(0))
        return (out[:, 0] if step else out) + residual, hidden[0]

    def sequence_context(self, observation: StepObservation) -> Tensor:
        return self._run(observation, None, step=False)[0]

    def step_context(
        self, observation: StepObservation, state: State
    ) -> tuple[Tensor, Tensor]:
        return self._run(observation, state, step=True)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1).long() + self.token_offsets[:stage]
            context = torch.cat((context, self.token_embed(tokens).flatten(-2)), dim=-1)
        return self.heads[stage](context)
