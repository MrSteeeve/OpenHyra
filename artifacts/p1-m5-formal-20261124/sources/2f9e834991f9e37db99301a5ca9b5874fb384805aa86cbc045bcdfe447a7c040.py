import torch
from torch import nn
from stochopia.neural.lob_interface import BUCKET_COUNTS, LOBWorldModel
from stochopia.neural.primitives import SeedGRULOBModel


class CandidateLOBWorldModel(LOBWorldModel):
    def __init__(self):
        super().__init__()
        self.parent = SeedGRULOBModel()
        self.parent.requires_grad_(False)

        # The interface does not expose the simulator's bucket decoder.
        # Populate with sign(decoded_tick - reference_tick), leaving invalid
        # buckets and non-price actions at zero. Default to no price bias.
        self.register_buffer(
            "price_direction", torch.zeros(BUCKET_COUNTS[1], BUCKET_COUNTS[2])
        )
        # A disconnected leaf permits backward without updating frozen weights.
        self.register_buffer("_backward_leaf", torch.zeros(()), persistent=False)

    @staticmethod
    def _pressure(ticks, shares, own, previous):
        if previous is None:
            return shares[..., 0].new_zeros(shares.shape[:-1])
        old_ticks, old_shares, old_pressure = previous.split((2, 2, 1), dim=-1)
        valid = (ticks > 0) & (old_ticks > 0)
        fraction = (old_shares - shares).relu() / old_shares.clamp_min(1)
        away = torch.stack(
            (ticks[..., 0] < old_ticks[..., 0], ticks[..., 1] > old_ticks[..., 1]),
            dim=-1,
        )
        withdrawal = torch.where(ticks == old_ticks, fraction, away.to(shares.dtype))
        withdrawal = withdrawal.masked_fill(~valid, 0)
        signed = withdrawal[..., 1] - withdrawal[..., 0]
        return (0.8 * old_pressure.squeeze(-1) + own.to(shares.dtype) * signed).clamp(-1, 1)

    @staticmethod
    def _touch(observation):
        # Keep tick comparisons exact for ordinary integer tick coordinates,
        # independently of the GRU's activation dtype.
        ticks = observation.book_ticks[..., :, 0].to(torch.float64)
        shares = observation.book_shares[..., :, 0].to(ticks.dtype)
        return ticks, shares

    def sequence_context(self, observation):
        context = self.parent.sequence_context(observation)
        ticks, shares = self._touch(observation)
        own = observation.is_own.reshape(ticks.shape[:-1])
        previous = None
        pressures = []
        for t in range(ticks.shape[1]):
            tick, share = ticks[:, t], shares[:, t]
            p = self._pressure(tick, share, own[:, t], previous).unsqueeze(-1)
            pressures.append(p)
            previous = torch.cat((tick, share, p), dim=-1)
        pressure = (
            torch.stack(pressures, dim=1).to(context)
            if pressures else context.new_empty((*context.shape[:-1], 1))
        )
        return torch.cat((context, pressure), dim=-1)

    def step_context(self, observation, state):
        if state is None:
            hidden, previous = None, None
        else:
            hidden, previous = state[..., :-5], state[..., -5:]
            hidden = hidden.to(next(self.parent.parameters()).dtype)
        context, hidden = self.parent.step_context(observation, hidden)
        ticks, shares = self._touch(observation)
        own = observation.is_own.reshape(ticks.shape[:-1])
        p = self._pressure(ticks, shares, own, previous).unsqueeze(-1)
        return (
            torch.cat((context, p.to(context)), dim=-1),
            torch.cat((hidden.to(ticks.dtype), ticks, shares, p), dim=-1),
        )

    def head_logits(self, context, stage, prior):
        logits = self.parent.head_logits(context[..., :-1], stage, prior)
        if stage == 2:
            action = prior[1].reshape(context.shape[:-1]).long()
            logits = logits + 1.25 * context[..., -1:] * self.price_direction[action].to(logits)
        if torch.is_grad_enabled() and not logits.requires_grad:
            leaf = self._backward_leaf.detach().requires_grad_(True)
            logits = logits + (0 * leaf).to(logits)
        return logits
