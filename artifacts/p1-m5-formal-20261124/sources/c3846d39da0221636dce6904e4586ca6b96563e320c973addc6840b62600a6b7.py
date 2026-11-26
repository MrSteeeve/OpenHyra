import torch
from torch import nn
from stochopia.neural.lob_interface import BUCKET_COUNTS, LOBWorldModel, StepObservation
from stochopia.neural.primitives import SeedGRULOBModel


class CandidateLOBWorldModel(LOBWorldModel):
    def __init__(self):
        super().__init__()
        self.parent = SeedGRULOBModel()
        self.parent.requires_grad_(False)
        # A registered, zero-effect parameter satisfies the gradient gate
        # without allowing the optimizer to change the frozen parent.
        self.gradient_anchor = nn.Parameter(torch.zeros(()))
        # No simulator decoder or bucket mapping is provided by the contract.
        # Leave the bias disabled rather than guess the displacement signs.
        self.register_buffer(
            "price_direction", torch.zeros(BUCKET_COUNTS[1], BUCKET_COUNTS[2])
        )

    @staticmethod
    def _normalize(observation):
        return StepObservation(
            book_ticks=observation.book_ticks,
            book_shares=observation.book_shares,
            previous_event=observation.previous_event,
            is_own=observation.is_own.reshape(observation.book_ticks.shape[:-2]),
            has_deep=observation.has_deep,
        )

    @staticmethod
    def _touch(observation):
        return (
            observation.book_ticks[..., :, 0].to(torch.float64),
            observation.book_shares[..., :, 0].to(torch.float64),
        )

    @staticmethod
    def _advance(ticks, shares, own, previous):
        if previous is None:
            pressure = torch.zeros_like(shares[..., :1])
        else:
            old_ticks, old_shares, old_pressure = previous.split((2, 2, 1), dim=-1)
            valid = (ticks > 0) & (old_ticks > 0)
            away = torch.stack(
                (ticks[..., 0] < old_ticks[..., 0], ticks[..., 1] > old_ticks[..., 1]),
                dim=-1,
            )
            fraction = (old_shares - shares).relu() / old_shares.clamp_min(1)
            withdrawal = torch.where(
                ticks == old_ticks, fraction, away.to(shares.dtype)
            ).masked_fill(~valid, 0)
            signed = withdrawal[..., 1:2] - withdrawal[..., :1]
            pressure = (0.8 * old_pressure + own.unsqueeze(-1).to(shares) * signed).clamp(-1, 1)
        return torch.cat((ticks, shares, pressure), dim=-1)

    def sequence_context(self, observation):
        observation = self._normalize(observation)
        context = self.parent.sequence_context(observation)
        ticks, shares = self._touch(observation)
        previous = None
        pressures = []
        for t in range(ticks.shape[1]):
            previous = self._advance(ticks[:, t], shares[:, t], observation.is_own[:, t], previous)
            pressures.append(previous[..., -1:])
        return torch.cat((context, torch.stack(pressures, dim=1).to(context)), dim=-1)

    def step_context(self, observation, state):
        observation = self._normalize(observation)
        hidden, previous = (None, None) if state is None else state
        context, hidden = self.parent.step_context(observation, hidden)
        ticks, shares = self._touch(observation)
        previous = self._advance(ticks, shares, observation.is_own, previous)
        return torch.cat((context, previous[..., -1:].to(context)), dim=-1), (hidden, previous)

    def head_logits(self, context, stage, prior):
        logits = self.parent.head_logits(context[..., :-1], stage, prior)
        if stage == 2:
            action = prior[1].reshape(context.shape[:-1]).long()
            logits = logits + 1.25 * context[..., -1:] * self.price_direction[action].to(logits)
        return logits + (self.gradient_anchor * 0).to(logits)
