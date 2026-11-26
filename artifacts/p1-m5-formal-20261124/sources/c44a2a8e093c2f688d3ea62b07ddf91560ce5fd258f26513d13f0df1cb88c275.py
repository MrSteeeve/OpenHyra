import torch
from torch import nn
from stochopia.neural.lob_interface import LOBWorldModel
from stochopia.neural.primitives import SeedGRULOBModel


class CandidateLOBWorldModel(LOBWorldModel):
    def __init__(self):
        super().__init__()
        self.parent = SeedGRULOBModel()
        self.parent.requires_grad_(False)

        # Fallback encoding: paired bid/ask actions, with the market pair
        # last, and 26 side-relative offsets indexed from -13 to +12.
        offsets = torch.arange(26) - 13
        side = torch.tensor([-1, 1, -1, 1, 0, 0])
        self.register_buffer(
            "price_direction", (side[:, None] * offsets[None, :]).sign(),
            persistent=False,
        )

    @staticmethod
    def _pressure(ticks, shares, own, previous, pressure):
        old_ticks, old_shares = previous
        valid = (ticks > 0) & (old_ticks > 0)
        same = ticks == old_ticks
        fraction = (old_shares - shares).clamp_min(0) / old_shares.clamp_min(1)
        retreat = torch.stack(
            (ticks[..., 0] < old_ticks[..., 0],
             ticks[..., 1] > old_ticks[..., 1]), dim=-1,
        )
        withdrawal = torch.where(same, fraction, retreat.to(shares.dtype))
        withdrawal = withdrawal.masked_fill(~valid, 0)
        bid, ask = withdrawal.unbind(-1)
        return (0.85 * pressure + own.to(shares.dtype) * (ask - bid)).clamp(-1, 1)

    @staticmethod
    def _touch(observation, context):
        ticks = observation.book_ticks[..., :, 0]
        shares = observation.book_shares[..., :, 0].to(context.dtype)
        own = observation.is_own.reshape(context.shape[:-1])
        return ticks, shares, own

    def sequence_context(self, observation):
        context = self.parent.sequence_context(observation)
        ticks, shares, own = self._touch(observation, context)
        if context.size(1) == 0:
            return torch.cat((context, context[..., :1]), dim=-1)
        pressure = context.new_zeros(context.size(0))
        previous = (torch.zeros_like(ticks[:, 0]), torch.zeros_like(shares[:, 0]))
        history = []
        for t in range(context.size(1)):
            pressure = self._pressure(ticks[:, t], shares[:, t], own[:, t],
                                      previous, pressure)
            previous = (ticks[:, t], shares[:, t])
            history.append(pressure)
        return torch.cat((context, torch.stack(history, dim=1).unsqueeze(-1)), dim=-1)

    def step_context(self, observation, state):
        hidden = None if state is None else state[0]
        context, hidden = self.parent.step_context(observation, hidden)
        ticks, shares, own = self._touch(observation, context)
        if state is None:
            previous = (torch.zeros_like(ticks), torch.zeros_like(shares))
            pressure = context.new_zeros(context.shape[:-1])
        else:
            previous = (state[1], state[2])
            pressure = state[3]
        pressure = self._pressure(ticks, shares, own, previous, pressure)
        return (torch.cat((context, pressure.unsqueeze(-1)), dim=-1),
                (hidden, ticks, shares, pressure))

    def head_logits(self, context, stage, prior):
        logits = self.parent.head_logits(context[..., :-1], stage, prior)
        if stage != 2:
            return logits
        action = prior[1].long().reshape(context.shape[:-1])
        direction = self.price_direction[action].to(logits.dtype)
        return logits + (0.75 * context[..., -1:]) * direction
